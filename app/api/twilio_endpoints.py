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
import numpy as np




load_dotenv()
router = APIRouter()


# Initialize clients
try:
    zendesk_client = ZendeskClient()
    print("✅ ZendeskClient initialized")
except Exception as e:
    zendesk_client = None
    print(f"⚠️ ZendeskClient init failed: {e}")


try:
    shopify_client = ShopifyClient()
    print("✅ ShopifyClient initialized")
except Exception as e:
    shopify_client = None
    print(f"⚠️ ShopifyClient init failed: {e}")


try:
    db_client = DatabaseClient()
    print("✅ DatabaseClient initialized")
except Exception as e:
    db_client = None
    print(f"⚠️ DatabaseClient init failed: {e}")




# ============================================================
# AUDIO CONVERSION UTILITIES - OPTIMIZED FOR TWILIO
# ============================================================


def mulaw_to_linear_pcm(mulaw_data: bytes) -> bytes:
    """Convert μ-law to 16-bit linear PCM using Python's audioop."""
    try:
        if not mulaw_data or len(mulaw_data) == 0:
            return b''
        linear_pcm = audioop.ulaw2lin(mulaw_data, 2)
        return linear_pcm
    except Exception as e:
        print(f"❌ μ-law to PCM error: {e}")
        return b''




def linear_pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit linear PCM to μ-law using Python's audioop."""
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




def calculate_audio_energy(pcm_data: bytes) -> float:
    """Calculate RMS energy of audio to detect speech vs silence"""
    try:
        if not pcm_data or len(pcm_data) < 2:
            return 0.0
       
        # Ensure even length
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]
       
        # Convert to numpy array
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
       
        # Calculate RMS energy
        rms = np.sqrt(np.mean(np.square(audio_array.astype(float))))
        return rms
    except Exception as e:
        return 0.0




def is_silence(pcm_data: bytes, threshold: float = 500.0) -> bool:
    """Detect if audio chunk is silence"""
    energy = calculate_audio_energy(pcm_data)
    return energy < threshold




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




# Global cache for Shopify data per call
_shopify_cache = {}




# ============================================================
# WEBSOCKET MEDIA STREAM - WITH INTERRUPTION HANDLING
# ============================================================


@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle real-time audio streaming with interruption detection"""
    await websocket.accept()
    print("🔌 WebSocket connection established")
   
    # Session state
    conversation_history = []
    call_sid = None
    stream_sid = None
    caller_number = None
    called_number = None
    caller_email = None
    audio_buffer = []
    start_time = None
    last_user_speech_time = 0
    greeting_sent = False
    stop_received = False
   
    # Agent state tracking for interruption detection
    agent_is_speaking = False
    agent_started_speaking_at = 0
    current_response_task = None
    speech_buffer = []  # Buffer for continuous speech detection
    silence_counter = 0
    min_speech_chunks = 3  # Minimum chunks of speech to process (reduce false positives)
   
    # Dynamic timing
    processing_interval = 1.8  # Start with 1.8s for responsiveness
    silence_threshold = 800.0  # Energy threshold for silence detection
   
    try:
        while not stop_received:
            try:
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=20.0)
                except asyncio.TimeoutError:
                    print("⏱️ WebSocket timeout - no data received")
                    break
                except WebSocketDisconnect:
                    print("🔌 Client disconnected")
                    break
               
                event_type = data.get("event")
               
                # ==================== START EVENT ====================
                if event_type == "start":
                    start_info = data.get("start", {})
                    call_sid = start_info.get("callSid", f"UNKNOWN_{int(time.time())}")
                    stream_sid = start_info.get("streamSid")
                    caller_number = start_info.get("customParameters", {}).get("From") or start_info.get("from", "Unknown")
                    called_number = start_info.get("customParameters", {}).get("To") or start_info.get("to", "Unknown")
                    custom_params = start_info.get("customParameters", {})
                    caller_email = custom_params.get("caller_email")
                    start_time = time.time()
                    last_user_speech_time = start_time
                   
                    print(f"🎬 Stream started")
                    print(f"   Call SID: {call_sid}")
                    print(f"   Stream SID: {stream_sid}")
                    print(f"   From: {caller_number}")
                    print(f"   To: {called_number}")
                    print(f"   Email: {caller_email or 'Not provided'}")
                   
                    # Fetch Shopify data in background
                    if shopify_client and caller_email:
                        asyncio.create_task(
                            fetch_and_cache_shopify(caller_email, call_sid)
                        )
                   
                    # Send greeting immediately
                    if not greeting_sent and stream_sid:
                        agent_is_speaking = True
                        agent_started_speaking_at = time.time()
                        asyncio.create_task(
                            send_greeting(websocket, stream_sid, conversation_history)
                        )
                        greeting_sent = True
                        # Agent will stop speaking after greeting is sent
                        asyncio.create_task(mark_agent_done_speaking(1.5))
               
                # ==================== MEDIA EVENT ====================
                elif event_type == "media":
                    media = data.get("media", {})
                    audio_payload = media.get("payload")
                   
                    if not audio_payload:
                        continue
                   
                    # Decode incoming μ-law audio
                    mulaw_bytes = base64.b64decode(audio_payload)
                    pcm_chunk = mulaw_to_linear_pcm(mulaw_bytes)
                   
                    if not pcm_chunk:
                        continue
                   
                    # INTERRUPTION DETECTION: Check if user is speaking while agent is speaking
                    if agent_is_speaking:
                        # Check if this chunk contains speech (not silence)
                        if not is_silence(pcm_chunk, silence_threshold):
                            # User is speaking - check if agent has been speaking for at least 0.5s
                            time_since_agent_started = time.time() - agent_started_speaking_at
                            if time_since_agent_started > 0.5:
                                print("🚨 INTERRUPTION DETECTED - User speaking while agent talks")
                                # Cancel current agent response
                                if current_response_task and not current_response_task.done():
                                    current_response_task.cancel()
                                    print("❌ Cancelled agent response")
                               
                                agent_is_speaking = False
                                audio_buffer.clear()  # Clear any buffered audio
                                speech_buffer.clear()
                   
                    # Track speech vs silence
                    is_silent = is_silence(pcm_chunk, silence_threshold)
                   
                    if not is_silent:
                        speech_buffer.append(mulaw_bytes)
                        silence_counter = 0
                        audio_buffer.append(mulaw_bytes)
                    else:
                        silence_counter += 1
                        # Still buffer during short silences (< 0.6s)
                        if silence_counter < 5:
                            audio_buffer.append(mulaw_bytes)
                   
                    # Process audio when we detect end of speech or buffer is full
                    current_time = time.time()
                    time_since_last_speech = current_time - last_user_speech_time
                   
                    should_process = False
                   
                    # Condition 1: Detected speech followed by silence
                    if len(speech_buffer) >= min_speech_chunks and silence_counter >= 4:
                        should_process = True
                        reason = "speech + silence"
                   
                    # Condition 2: Long continuous speech (buffer full)
                    elif len(audio_buffer) > 25:  # ~3 seconds of audio
                        should_process = True
                        reason = "buffer full"
                   
                    # Condition 3: Timeout with some audio
                    elif time_since_last_speech >= processing_interval and len(audio_buffer) >= 8:
                        should_process = True
                        reason = "timeout"
                   
                    if should_process and not agent_is_speaking:
                        print(f"🎤 Processing trigger: {reason}")
                       
                        # Combine buffered audio
                        combined_mulaw = b''.join(audio_buffer)
                        audio_buffer.clear()
                        speech_buffer.clear()
                        silence_counter = 0
                       
                        # Skip very short audio (< 0.5 seconds = 4000 bytes)
                        if len(combined_mulaw) < 4000:
                            print(f"⏭️ Skipping short audio: {len(combined_mulaw)} bytes")
                            continue
                       
                        last_user_speech_time = current_time
                       
                        # Process in background
                        current_response_task = asyncio.create_task(
                            process_audio_chunk_with_interruption(
                                combined_mulaw,
                                websocket,
                                stream_sid,
                                conversation_history,
                                caller_email,
                                call_sid,
                                current_time
                            )
                        )
               
                # ==================== STOP EVENT ====================
                elif event_type == "stop":
                    print(f"🏁 Stream stopped for call: {call_sid}")
                    stop_received = True
           
            except Exception as e:
                print(f"⚠️ WebSocket loop error: {e}")
                traceback.print_exc()
                break
   
    finally:
        print(f"🔌 Closing WebSocket for call: {call_sid}")
       
        # Process any remaining audio
        if audio_buffer and len(audio_buffer) > 0:
            try:
                combined_mulaw = b''.join(audio_buffer)
                if len(combined_mulaw) > 8000:
                    pcm_audio = mulaw_to_linear_pcm(combined_mulaw)
                    if not is_silence(pcm_audio, silence_threshold):
                        wav_audio = create_wav_header(len(pcm_audio), sample_rate=8000) + pcm_audio
                        transcript = await transcribe_audio(wav_audio)
                        if transcript and transcript.strip() and len(transcript.strip()) > 2:
                            # Filter out common Whisper hallucinations
                            if transcript.strip().lower() not in ["thank you", "thanks", "you"]:
                                conversation_history.append({
                                    "role": "user",
                                    "text": transcript,
                                    "timestamp": time.time()
                                })
                                print(f"👤 User (final): '{transcript}'")
            except Exception as e:
                print(f"⚠️ Final audio processing error: {e}")
       
        # Generate conversation summary
        full_transcript = "\n".join([
            f"{msg['role'].capitalize()}: {msg['text']}"
            for msg in conversation_history
        ])
       
        summary = ""
        if full_transcript and len(full_transcript) > 50:
            try:
                summary = await summarize_conversation(full_transcript)
                print(f"📋 Summary: {summary[:100]}...")
            except Exception as e:
                print(f"⚠️ Summary generation error: {e}")
       
        duration = int(time.time() - start_time) if start_time else 0
       
        # Log to database
        if db_client and call_sid:
            try:
                db_client.insert_call_transcript(
                    call_sid=call_sid,
                    from_number=caller_number or "Unknown",
                    to_number=called_number or "Unknown",
                    transcript=full_transcript,
                    summary=summary
                )
                print("✅ Call transcript logged to database")
               
                db_client.insert_call_metadata(
                    call_sid=call_sid,
                    call_status="completed",
                    duration=duration,
                    call_type="inbound",
                    tags=["completed", "ai_handled"]
                )
                print("✅ Call metadata logged to database")
            except Exception as e:
                print(f"⚠️ Database logging error: {e}")


        # Create Zendesk ticket if needed
        if zendesk_client and summary and call_sid:
            try:
                ticket_id = zendesk_client.create_ticket(
                    subject=f"AI Call from {caller_number or 'Unknown'}",
                    description=f"""Call Summary:
{summary}


Full Transcript:
{full_transcript}


Call Details:
- Call SID: {call_sid}
- From: {caller_number}
- To: {called_number}
- Duration: {duration} seconds
- Email: {caller_email or 'Not provided'}
""",
                    requester_email=caller_email or "noreply@artbymaudsch.com",
                    tags=["ai_call", "phone_support", "automated"]
                )
               
                print(f"🎫 Zendesk ticket created: #{ticket_id}")
               
                if db_client:
                    try:
                        db_client.insert_zendesk_ticket(
                            call_sid=call_sid,
                            zendesk_ticket_id=ticket_id
                        )
                    except Exception as e:
                        if "duplicate key" not in str(e).lower():
                            print(f"⚠️ Zendesk ticket logging error: {e}")
           
            except Exception as e:
                print(f"⚠️ Zendesk ticket creation error: {e}")
       
        print(f"✅ Call session complete: {call_sid}")
        print(f"   Duration: {duration}s")
        print(f"   Messages: {len(conversation_history)}")
       
        # Clean up cache
        if call_sid in _shopify_cache:
            del _shopify_cache[call_sid]




async def mark_agent_done_speaking(delay: float):
    """Mark agent as done speaking after a delay"""
    global agent_is_speaking
    await asyncio.sleep(delay)
    agent_is_speaking = False




async def fetch_and_cache_shopify(email: str, call_sid: str):
    """Fetch Shopify data once and cache it"""
    global _shopify_cache
    try:
        if shopify_client:
            print(f"🛒 Background Shopify fetch for {email}...")
            orders = await asyncio.get_event_loop().run_in_executor(
                None,
                shopify_client.get_customer_orders,
                email
            )
            if orders and len(orders) > 0:
                _shopify_cache[call_sid] = orders[0]
                print(f"✅ Shopify: Cached order {orders[0].get('name')} for call {call_sid}")
            else:
                _shopify_cache[call_sid] = None
                print(f"ℹ️  Shopify: No orders found for {email}")
    except Exception as e:
        print(f"⚠️ Background Shopify fetch error: {e}")
        _shopify_cache[call_sid] = None




async def send_greeting(websocket: WebSocket, stream_sid: str, conversation_history: list):
    """Send greeting with minimal delay"""
    global agent_is_speaking, agent_started_speaking_at
   
    try:
        print("🎤 Generating greeting...")
       
        # Simple greeting without customer context
        greeting_prompt = "Say a very brief, warm greeting for an art gallery. Maximum 8 words. Just say hello and ask how you can help."
       
        ai_text, ai_audio_8khz_pcm = await generate_ai_response_live(
            greeting_prompt,
            caller_email=None,
            shopify_client=None,
            shopify_data_cache=None
        )
       
        if ai_text:
            conversation_history.append({
                "role": "assistant",
                "text": ai_text,
                "timestamp": time.time()
            })
       
        if ai_audio_8khz_pcm and len(ai_audio_8khz_pcm) > 100:
            mulaw_audio = linear_pcm_to_mulaw(ai_audio_8khz_pcm)
           
            if mulaw_audio and len(mulaw_audio) > 0:
                payload = base64.b64encode(mulaw_audio).decode("utf-8")
               
                try:
                    await websocket.send_json({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload}
                    })
                    print(f"🔊 Greeting sent: '{ai_text}'")
                   
                    # Estimate greeting duration (bytes / 8000 samples/sec)
                    greeting_duration = len(mulaw_audio) / 8000.0
                    await mark_agent_done_speaking(greeting_duration + 0.3)
                   
                except Exception as send_error:
                    print(f"❌ Failed to send greeting: {send_error}")
                    agent_is_speaking = False
        else:
            print(f"⚠️ No greeting audio generated")
            agent_is_speaking = False
                   
    except Exception as e:
        print(f"⚠️ Greeting generation error: {e}")
        agent_is_speaking = False




async def process_audio_chunk_with_interruption(
    mulaw_audio: bytes,
    websocket: WebSocket,
    stream_sid: str,
    conversation_history: list,
    caller_email: str,
    call_sid: str,
    timestamp: float
):
    """Process audio chunk with interruption awareness"""
    global agent_is_speaking, agent_started_speaking_at
   
    try:
        chunk_start = time.time()
        print(f"🎤 Processing {len(mulaw_audio)} bytes audio...")
       
        # Convert to PCM for analysis and transcription
        pcm_audio = mulaw_to_linear_pcm(mulaw_audio)
       
        if not pcm_audio or len(pcm_audio) < 2000:
            print(f"⚠️ PCM conversion failed or too short")
            return
       
        # Check if it's actually silence
        if is_silence(pcm_audio, 500.0):
            print(f"⏭️ Skipping silence")
            return
       
        # Create WAV for transcription
        wav_audio = create_wav_header(len(pcm_audio), sample_rate=8000) + pcm_audio
       
        # Transcribe with enhanced settings
        transcript = await transcribe_audio(wav_audio, min_length=3)
       
        if not transcript or len(transcript.strip()) < 2:
            print(f"⏭️ Empty or too short transcript")
            return
       
        transcript = transcript.strip()
       
        # Filter common Whisper hallucinations
        hallucinations = [
            "thank you", "thanks", "you", "thank you.", "thanks.",
            "uh", "um", "mhm", "mm-hmm", "uh-huh"
        ]
       
        if transcript.lower() in hallucinations:
            print(f"⏭️ Filtering hallucination: '{transcript}'")
            return
       
        print(f"👤 User said: '{transcript}'")
       
        # Add to conversation history
        conversation_history.append({
            "role": "user",
            "text": transcript,
            "timestamp": timestamp
        })
       
        # Generate AI response
        try:
            # Build context from recent conversation
            context_messages = []
            for msg in conversation_history[-6:]:  # Last 6 messages for better context
                role = msg['role']
                text = msg['text']
                context_messages.append(f"{role}: {text}")
           
            full_context = "\n".join(context_messages)
           
            # Get cached Shopify data
            global _shopify_cache
            cached_shopify = _shopify_cache.get(call_sid)
           
            print("🤖 Generating AI response...")
           
            # Mark agent as speaking BEFORE generating response
            agent_is_speaking = True
            agent_started_speaking_at = time.time()
           
            ai_text, ai_audio_8khz_pcm = await generate_ai_response_live(
                full_context,
                caller_email=caller_email,
                shopify_client=None,
                shopify_data_cache=cached_shopify
            )
           
            if not ai_text:
                print("⚠️ No AI response generated")
                agent_is_speaking = False
                return
           
            # Add AI response to history
            conversation_history.append({
                "role": "assistant",
                "text": ai_text,
                "timestamp": time.time()
            })
           
            print(f"🤖 AI responds: '{ai_text}'")
           
            # Send audio response
            if ai_audio_8khz_pcm and len(ai_audio_8khz_pcm) > 100 and stream_sid:
                mulaw_audio = linear_pcm_to_mulaw(ai_audio_8khz_pcm)
               
                if mulaw_audio and len(mulaw_audio) > 0:
                    payload = base64.b64encode(mulaw_audio).decode("utf-8")
                   
                    try:
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload}
                        })
                       
                        total_time = time.time() - chunk_start
                        print(f"🔊 AI audio sent: {len(mulaw_audio)} bytes")
                        print(f"⏱️ Total response time: {total_time:.2f}s")
                       
                        # Estimate audio duration and mark agent done after it finishes
                        audio_duration = len(mulaw_audio) / 8000.0
                        await mark_agent_done_speaking(audio_duration + 0.3)
                       
                    except Exception as send_error:
                        print(f"❌ Failed to send AI audio: {send_error}")
                        agent_is_speaking = False
                else:
                    print("❌ AI audio μ-law conversion failed")
                    agent_is_speaking = False
            else:
                print(f"⚠️ No AI audio to send")
                agent_is_speaking = False
       
        except asyncio.CancelledError:
            print("🚫 Response generation cancelled (user interrupted)")
            agent_is_speaking = False
            raise
       
        except Exception as e:
            print(f"❌ AI response error: {e}")
            agent_is_speaking = False
   
    except asyncio.CancelledError:
        print("🚫 Audio processing cancelled")
        agent_is_speaking = False
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        traceback.print_exc()
        agent_is_speaking = False

