# app/api/twilio_endpoints.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from app.services.ai_service import generate_ai_response_live, transcribe_audio, summarize_conversation
from app.services.zendesk_client import ZendeskClient
from app.services.shopify_client import ShopifyClient
from app.core.db_client import DatabaseClient
from twilio.twiml.voice_response import VoiceResponse, Connect
from typing import Optional
import base64
import os
import time
import traceback
import asyncio
import audioop
import struct
from dotenv import load_dotenv
import json

load_dotenv()
router = APIRouter()

# Initialize clients - all optional, app works without them
zendesk_client = None
shopify_client = None
db_client = None

try:
    from app.services.zendesk_client import ZendeskClient
    zendesk_client = ZendeskClient()
    print("✅ ZendeskClient initialized")
except Exception as e:
    print(f"ℹ️  ZendeskClient not available: {e}")

try:
    from app.services.shopify_client import ShopifyClient
    shopify_client = ShopifyClient()
    print("✅ ShopifyClient initialized")
except Exception as e:
    print(f"ℹ️  ShopifyClient not available: {e}")

try:
    from app.core.db_client import DatabaseClient
    db_client = DatabaseClient()
    if db_client.enabled:
        print("✅ DatabaseClient initialized")
    else:
        print("ℹ️  DatabaseClient disabled (no database configured)")
except Exception as e:
    print(f"ℹ️  DatabaseClient not available: {e}")

# ============================================================
# AUDIO CONVERSION UTILITIES
# ============================================================

def mulaw_to_linear_pcm(mulaw_data: bytes) -> bytes:
    """Convert μ-law to 16-bit linear PCM"""
    try:
        if not mulaw_data or len(mulaw_data) == 0:
            return b''
        linear_pcm = audioop.ulaw2lin(mulaw_data, 2)
        return linear_pcm
    except Exception as e:
        print(f"❌ μ-law to PCM error: {e}")
        return b''

def linear_pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit linear PCM to μ-law"""
    try:
        if not pcm_data or len(pcm_data) == 0:
            return b''
        
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]
        
        if len(pcm_data) == 0:
            return b''
        
        mulaw_data = audioop.lin2ulaw(pcm_data, 2)
        return mulaw_data
    except Exception as e:
        print(f"❌ PCM to μ-law error: {e}")
        return b''

def create_wav_header(
    data_size: int,
    sample_rate: int = 8000,
    channels: int = 1,
    bits_per_sample: int = 16
) -> bytes:
    """Create proper WAV file header"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    header = b'RIFF'
    header += struct.pack('<I', data_size + 36)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)
    header += struct.pack('<H', 1)
    header += struct.pack('<H', channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', byte_rate)
    header += struct.pack('<H', block_align)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)
    
    return header

# ============================================================
# TWILIO WEBHOOKS
# ============================================================

@router.post("/webhook/voice")
async def voice_webhook(request: Request):
    """Handle incoming call and return TwiML with WebSocket stream"""
    try:
        form_data = await request.form()
        caller_number = form_data.get("From", "Unknown")
        called_number = form_data.get("To", "Unknown")
        call_sid = form_data.get("CallSid", "Unknown")
        
        print(f"📞 Incoming call: {caller_number} → {called_number} (SID: {call_sid})")
        
        websocket_url = os.getenv(
            "WEBSOCKET_URL",
            "wss://your-ngrok-url.ngrok.io/api/twilio/media-stream"
        )
        print(f"🌐 Using WebSocket URL: {websocket_url}")
        
        response = VoiceResponse()
        connect = Connect()
        connect.stream(url=websocket_url)
        response.append(connect)
        
        print(f"✅ TwiML generated for call SID {call_sid}")
        return Response(content=str(response), media_type="application/xml")
        
    except Exception as e:
        print(f"❌ Error in voice_webhook: {e}")
        traceback.print_exc()
        error_response = VoiceResponse()
        error_response.say("Sorry, something went wrong. Please try again later.")
        return Response(content=str(error_response), media_type="application/xml")

@router.post("/webhook/status")
async def status_webhook(request: Request):
    """Handle Twilio call status callbacks"""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "Unknown")
        call_status = form_data.get("CallStatus", "Unknown")
        print(f"📊 Call {call_sid} status: {call_status}")
        
        if db_client:
            try:
                db_client.insert_call_metadata(
                    call_sid=call_sid,
                    call_status=call_status,
                    duration=0,
                    call_type="inbound",
                    tags=[call_status]
                )
            except Exception as e:
                print(f"⚠️ DB status log error: {e}")
        
        return {"status": "received"}
        
    except Exception as e:
        print(f"❌ Error in status_webhook: {e}")
        return {"status": "error", "error": str(e)}

# Global cache
_shopify_cache = {}

# ============================================================
# CALL STATE MANAGER
# ============================================================

class CallState:
    """Manages state for a single call"""
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.conversation_history = []
        self.audio_buffer = []
        self.caller_email = None
        self.caller_number = "Unknown"
        self.called_number = "Unknown"
        self.start_time = time.time()
        self.last_processing_time = time.time()
        self.greeting_sent = False
        self.is_processing = False
        self.agent_speaking = False
        self.current_mark = None
        self.interrupted = False
        self.stream_sid = None
        
    def add_message(self, role: str, text: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "text": text,
            "timestamp": time.time()
        })
        
    def get_context(self, num_messages: int = 6) -> str:
        """Get recent conversation context"""
        recent = self.conversation_history[-num_messages:]
        return "\n".join([f"{msg['role']}: {msg['text']}" for msg in recent])
    
    def get_duration(self) -> int:
        """Get call duration in seconds"""
        return int(time.time() - self.start_time)

# ============================================================
# WEBSOCKET MEDIA STREAM
# ============================================================

@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle real-time audio streaming with Twilio"""
    await websocket.accept()
    print("🔌 WebSocket connection established")
    
    state = None
    stop_received = False
    
    # Audio processing configuration
    PROCESSING_INTERVAL = 1.8  # Process audio every 1.8 seconds
    MIN_AUDIO_SIZE = 2500  # Minimum bytes before processing
    INTERRUPT_THRESHOLD = 400  # RMS threshold for interrupt detection
    SILENCE_THRESHOLD = 150  # RMS threshold for silence
    
    try:
        while not stop_received:
            try:
                # Receive data with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(), 
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                print("⏱️ WebSocket timeout - no data received")
                break
            except WebSocketDisconnect:
                print("🔌 WebSocket disconnected by client")
                break
            except Exception as e:
                print(f"❌ WebSocket receive error: {e}")
                break
            
            event_type = data.get("event")
            
            # =============== START EVENT ===============
            if event_type == "start":
                start_info = data.get("start", {})
                call_sid = start_info.get("callSid")
                stream_sid = start_info.get("streamSid")
                
                # Initialize call state
                state = CallState(call_sid)
                state.stream_sid = stream_sid
                
                # Extract call details from custom parameters if available
                custom_params = start_info.get("customParameters", {})
                state.caller_number = custom_params.get("From", "Unknown")
                state.called_number = custom_params.get("To", "Unknown")
                
                print(f"🎬 Stream started - Call: {call_sid}, Stream: {stream_sid}")
                
                # Send greeting asynchronously
                asyncio.create_task(send_greeting(websocket, state))
                
            # =============== MEDIA EVENT ===============
            elif event_type == "media":
                if not state:
                    continue
                    
                media = data.get("media", {})
                payload = media.get("payload")
                
                if not payload:
                    continue
                
                try:
                    mulaw_bytes = base64.b64decode(payload)
                except Exception as e:
                    print(f"❌ Base64 decode error: {e}")
                    continue
                
                # Check for interruption if agent is speaking
                if state.agent_speaking:
                    try:
                        pcm_chunk = mulaw_to_linear_pcm(mulaw_bytes)
                        if pcm_chunk and len(pcm_chunk) > 0:
                            rms = audioop.rms(pcm_chunk, 2)
                            
                            # Detect user speaking while agent talks
                            if rms > INTERRUPT_THRESHOLD:
                                print(f"⚡ INTERRUPT DETECTED (RMS={rms}) - Stopping agent")
                                state.agent_speaking = False
                                state.interrupted = True
                                
                                # Send clear command to stop current audio
                                try:
                                    await websocket.send_json({
                                        "event": "clear",
                                        "streamSid": state.stream_sid
                                    })
                                    print("🛑 Sent 'clear' command to stop agent audio")
                                except Exception as e:
                                    print(f"⚠️ Failed to send clear: {e}")
                                
                                # Clear buffer and start fresh
                                state.audio_buffer.clear()
                    except Exception as e:
                        print(f"⚠️ Interrupt detection error: {e}")
                
                # Add to buffer
                state.audio_buffer.append(mulaw_bytes)
                
                # Process audio periodically
                now = time.time()
                time_since_last = now - state.last_processing_time
                buffer_size = sum(len(chunk) for chunk in state.audio_buffer)
                
                should_process = (
                    time_since_last >= PROCESSING_INTERVAL and
                    buffer_size >= MIN_AUDIO_SIZE and
                    not state.is_processing and
                    not state.agent_speaking
                )
                
                if should_process:
                    state.is_processing = True
                    state.last_processing_time = now
                    
                    # Get audio and clear buffer
                    audio_chunks = state.audio_buffer.copy()
                    state.audio_buffer.clear()
                    
                    # Process asynchronously
                    asyncio.create_task(
                        process_audio_chunk(
                            audio_chunks,
                            websocket,
                            state
                        )
                    )
            
            # =============== MARK EVENT ===============
            elif event_type == "mark":
                if state:
                    mark_name = data.get("mark", {}).get("name")
                    print(f"✓ Mark received: {mark_name}")
                    
                    if mark_name and "response_" in mark_name:
                        # Agent finished speaking
                        state.agent_speaking = False
                        state.current_mark = None
                        print("🔇 Agent finished speaking")
            
            # =============== STOP EVENT ===============
            elif event_type == "stop":
                if state:
                    print(f"🏁 Stream stopped for call {state.call_sid}")
                stop_received = True
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        traceback.print_exc()
        
    finally:
        # Cleanup and finalization
        if state:
            await finalize_call(state, websocket)
        
        try:
            await websocket.close()
        except:
            pass
        
        print("🔌 WebSocket connection closed")

# ============================================================
# GREETING HANDLER
# ============================================================

async def send_greeting(websocket: WebSocket, state: CallState):
    """Send initial greeting"""
    if state.greeting_sent:
        return
    
    state.greeting_sent = True
    
    try:
        print("🎤 Generating greeting...")
        
        greeting_text = "Hi! Thanks for calling ArtByMaudsch. How can I help you today?"
        
        # Generate audio
        from app.services.ai_service import get_elevenlabs_client
        eleven_client = get_elevenlabs_client()
        
        audio_pcm = await eleven_client.text_to_speech_fast(greeting_text)
        
        if audio_pcm and len(audio_pcm) > 100:
            # Log greeting
            state.add_message("assistant", greeting_text)
            
            # Convert to μ-law and send
            mulaw_audio = linear_pcm_to_mulaw(audio_pcm)
            
            if mulaw_audio and len(mulaw_audio) > 0:
                payload = base64.b64encode(mulaw_audio).decode("utf-8")
                
                # Mark that agent is speaking
                state.agent_speaking = True
                mark_name = f"greeting_{int(time.time())}"
                state.current_mark = mark_name
                
                # Send audio
                await websocket.send_json({
                    "event": "media",
                    "streamSid": state.stream_sid,
                    "media": {"payload": payload}
                })
                
                # Send mark to detect when done
                await websocket.send_json({
                    "event": "mark",
                    "streamSid": state.stream_sid,
                    "mark": {"name": mark_name}
                })
                
                print(f"🔊 Greeting sent: '{greeting_text}'")
                
                # Auto-clear speaking flag after expected duration
                audio_duration = len(audio_pcm) / (8000 * 2)  # 8kHz, 16-bit
                await asyncio.sleep(audio_duration + 0.5)
                if state.current_mark == mark_name:
                    state.agent_speaking = False
                    print("🔇 Greeting completed (timeout)")
            else:
                print("❌ Greeting μ-law conversion failed")
        else:
            print("⚠️ No greeting audio generated")
            
    except Exception as e:
        print(f"⚠️ Greeting error: {e}")
        traceback.print_exc()
        state.greeting_sent = False

# ============================================================
# AUDIO PROCESSING
# ============================================================

async def process_audio_chunk(
    audio_chunks: list,
    websocket: WebSocket,
    state: CallState
):
    """Process accumulated audio chunks"""
    try:
        chunk_start = time.time()
        
        # Combine audio
        combined_mulaw = b''.join(audio_chunks)
        
        print(f"🎤 Processing {len(combined_mulaw)} bytes of audio...")
        
        # Convert to PCM
        pcm_audio = mulaw_to_linear_pcm(combined_mulaw)
        
        if not pcm_audio or len(pcm_audio) < 2000:
            print(f"⏭️ Audio too short after conversion")
            state.is_processing = False
            return
        
        # Check if mostly silence
        try:
            rms = audioop.rms(pcm_audio, 2)
            if rms < 150:
                print(f"⏭️ Mostly silence detected (RMS={rms})")
                state.is_processing = False
                return
        except:
            pass
        
        # Create WAV for transcription
        wav_audio = create_wav_header(len(pcm_audio), sample_rate=8000) + pcm_audio
        
        # Transcribe
        transcript = await transcribe_audio(wav_audio)
        
        if not transcript or len(transcript.strip()) < 3:
            print(f"⏭️ No valid transcript")
            state.is_processing = False
            return
        
        transcript = transcript.strip()
        print(f"👤 User: '{transcript}'")
        
        # Add to history
        state.add_message("user", transcript)
        
        # Get conversation context
        context = state.get_context(num_messages=6)
        
        # Get cached Shopify data
        cached_shopify = _shopify_cache.get(state.call_sid)
        
        # Generate AI response
        print("🤖 Generating AI response...")
        ai_text, ai_audio = await generate_ai_response_live(
            context,
            caller_email=state.caller_email,
            shopify_client=None,
            shopify_data_cache=cached_shopify
        )
        
        if not ai_text:
            print("⚠️ No AI response generated")
            state.is_processing = False
            return
        
        # Add to history
        state.add_message("assistant", ai_text)
        print(f"🤖 AI: '{ai_text}'")
        
        # Send audio response
        if ai_audio and len(ai_audio) > 100:
            # Convert to μ-law
            mulaw_audio = linear_pcm_to_mulaw(ai_audio)
            
            if mulaw_audio and len(mulaw_audio) > 0:
                payload = base64.b64encode(mulaw_audio).decode("utf-8")
                
                # Mark agent as speaking
                state.agent_speaking = True
                state.interrupted = False
                mark_name = f"response_{int(time.time() * 1000)}"
                state.current_mark = mark_name
                
                # Send audio
                await websocket.send_json({
                    "event": "media",
                    "streamSid": state.stream_sid,
                    "media": {"payload": payload}
                })
                
                # Send mark
                await websocket.send_json({
                    "event": "mark",
                    "streamSid": state.stream_sid,
                    "mark": {"name": mark_name}
                })
                
                total_time = time.time() - chunk_start
                print(f"🔊 Response sent ({total_time:.2f}s): {len(mulaw_audio)} bytes")
                
                # Auto-clear speaking flag after expected duration
                audio_duration = len(ai_audio) / (8000 * 2)
                await asyncio.sleep(audio_duration + 0.5)
                if state.current_mark == mark_name and not state.interrupted:
                    state.agent_speaking = False
                    print("🔇 Response playback completed")
            else:
                print("❌ Response μ-law conversion failed")
        else:
            print("⚠️ No AI audio to send")
        
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        traceback.print_exc()
    finally:
        state.is_processing = False

# ============================================================
# CALL FINALIZATION
# ============================================================

async def finalize_call(state: CallState, websocket: WebSocket):
    """Finalize call and save data"""
    try:
        print(f"📝 Finalizing call {state.call_sid}...")
        
        # Process any remaining audio
        if state.audio_buffer:
            try:
                combined_mulaw = b''.join(state.audio_buffer)
                if len(combined_mulaw) > 2000:
                    pcm_audio = mulaw_to_linear_pcm(combined_mulaw)
                    wav_audio = create_wav_header(len(pcm_audio)) + pcm_audio
                    transcript = await transcribe_audio(wav_audio)
                    if transcript and transcript.strip():
                        state.add_message("user", transcript)
                        print(f"👤 User (final): '{transcript}'")
            except Exception as e:
                print(f"⚠️ Final audio processing error: {e}")
        
        # Generate transcript
        full_transcript = "\n".join([
            f"{msg['role'].capitalize()}: {msg['text']}"
            for msg in state.conversation_history
        ])
        
        # Generate summary
        summary = ""
        if full_transcript and len(full_transcript) > 50:
            try:
                summary = await summarize_conversation(full_transcript)
                print(f"📋 Summary: {summary[:100]}...")
            except Exception as e:
                print(f"⚠️ Summary error: {e}")
                summary = "Summary generation failed"
        
        duration = state.get_duration()
        
        # Log to database
        if db_client and db_client.enabled:
            try:
                db_client.insert_call_transcript(
                    call_sid=state.call_sid,
                    from_number=state.caller_number,
                    to_number=state.called_number,
                    transcript=full_transcript,
                    summary=summary
                )
                print("✅ Transcript logged to database")
                
                db_client.insert_call_metadata(
                    call_sid=state.call_sid,
                    call_status="completed",
                    duration=duration,
                    call_type="inbound",
                    tags=["completed", "ai_handled"]
                )
                print("✅ Metadata logged to database")
            except Exception as e:
                print(f"⚠️ Database logging error: {e}")
        
        # Clean up cache
        if state.call_sid in _shopify_cache:
            del _shopify_cache[state.call_sid]
        
        print(f"✅ Call finalized: {state.call_sid}")
        print(f"   Duration: {duration}s")
        print(f"   Messages: {len(state.conversation_history)}")
        
    except Exception as e:
        print(f"❌ Finalization error: {e}")
        traceback.print_exc()

