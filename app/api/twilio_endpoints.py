# app/api/twilio_endpoints.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from app.services.ai_service import (
    generate_ai_response_live,
    transcribe_audio,
    summarize_conversation,
    search_catalog_for_request
)
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
    """
    Convert μ-law to 16-bit linear PCM using Python's audioop.
    Twilio uses 8kHz μ-law audio.
    """
    try:
        if not mulaw_data or len(mulaw_data) == 0:
            return b''
        
        # Use audioop for reliable μ-law decoding
        linear_pcm = audioop.ulaw2lin(mulaw_data, 2)  # 2 = 16-bit
        return linear_pcm
    except Exception as e:
        print(f"❌ μ-law to PCM error: {e}")
        return b''

def linear_pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """
    Convert 16-bit linear PCM to μ-law using Python's audioop.
    CRITICAL: Input must be 8kHz 16-bit PCM for Twilio.
    """
    try:
        if not pcm_data or len(pcm_data) == 0:
            return b''
        
        # Ensure even length for 16-bit samples
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]
        
        if len(pcm_data) == 0:
            return b''
        
        # Use audioop for reliable μ-law encoding
        mulaw_data = audioop.lin2ulaw(pcm_data, 2)  # 2 = 16-bit
        return mulaw_data
    except Exception as e:
        print(f"❌ PCM to μ-law error: {e}")
        traceback.print_exc()
        return b''

def resample_audio_safe(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """
    Safely resample audio with proper frame alignment.
    """
    try:
        if from_rate == to_rate:
            return pcm_data
        
        # Ensure data length is even (16-bit samples)
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]
        
        if len(pcm_data) == 0:
            return b''
        
        # Calculate expected output size
        num_frames_in = len(pcm_data) // 2  # 16-bit = 2 bytes per frame
        num_frames_out = int(num_frames_in * to_rate / from_rate)
        
        # Adjust input to ensure whole number of output frames
        adjusted_frames = int(num_frames_out * from_rate / to_rate)
        adjusted_bytes = adjusted_frames * 2
        
        if adjusted_bytes > len(pcm_data):
            adjusted_bytes = len(pcm_data)
        
        # Trim to adjusted size
        pcm_data = pcm_data[:adjusted_bytes]
        
        # Perform resampling
        resampled, _ = audioop.ratecv(pcm_data, 2, 1, from_rate, to_rate, None)
        return resampled
        
    except Exception as e:
        print(f"❌ Resampling error: {e}")
        traceback.print_exc()
        return pcm_data

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
    header += struct.pack('<H', 1)  # PCM format
    header += struct.pack('<H', channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', byte_rate)
    header += struct.pack('<H', block_align)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)
    
    return header

def calculate_audio_energy(mulaw_data: bytes) -> float:
    """Calculate audio energy to detect speech"""
    try:
        if not mulaw_data or len(mulaw_data) == 0:
            return 0.0
        
        # Convert to PCM for energy calculation
        pcm_data = mulaw_to_linear_pcm(mulaw_data)
        if not pcm_data or len(pcm_data) < 2:
            return 0.0
        
        # Calculate RMS energy
        rms = audioop.rms(pcm_data, 2)
        return float(rms)
    except Exception as e:
        return 0.0

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

        # Get WebSocket URL from environment or fallback
        websocket_url = os.getenv(
            "WEBSOCKET_URL",
            "wss://uncatenated-sherell-diminishingly.ngrok-free.dev/api/twilio/media-stream"
        )
        print(f"🌐 Using WebSocket URL: {websocket_url}")

        # --- Create TwiML manually to include <Parameter> tags ---
        twiml = f"""
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Connect>
                <Stream url="{websocket_url}">
                    <Parameter name="from" value="{caller_number}" />
                    <Parameter name="to" value="{called_number}" />
                    <Parameter name="email" value="info@artsbymaudsch.com" />
                </Stream>
            </Connect>
        </Response>
        """

        print(f"✅ TwiML generated for call SID {call_sid}")
        return Response(content=twiml.strip(), media_type="application/xml")

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

# Global cache for Shopify data per call - ENHANCED
_shopify_cache = {}

# Global catalog snapshot for product knowledge
_catalog_snapshot = {
    "data": None,
    "timestamp": 0
}
_catalog_lock = None
CATALOG_REFRESH_SECONDS = 60 * 60  # 1 hour

# ============================================================
# WEBSOCKET MEDIA STREAM - IMPROVED INTERRUPTION & MEMORY
# ============================================================

@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle real-time audio streaming from Twilio with IMPROVED INTERRUPTION HANDLING and CALLER MEMORY"""
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
    last_user_speech_time = 0  # Track when user last spoke
    greeting_sent = False
    stop_received = False
    
    # NEW: Enhanced customer verification state
    customer_verified = False
    customer_data = None
    verification_attempts = 0
    max_verification_attempts = 2
    
    # NEW: Persistent caller memory across the entire call
    caller_memory = {
        "order_numbers": [],  # Track all mentioned order numbers
        "art_interests": [],  # Track art preferences/interests
        "issues": [],  # Track mentioned problems/issues
        "name": None,  # Caller name if mentioned
        "pending_actions": [],  # Track actions that need confirmation (removed cancel functionality)
        "customer_id": None,  # Shopify customer ID
        "verified_orders": {},  # Cache of verified orders {order_num: order_data}
        "customer_info": None,  # Full customer data from Shopify
        "all_orders": []  # All customer orders from Shopify
    }
    
    # CRITICAL: Single processing lock to prevent concurrent AI responses
    processing_lock = asyncio.Lock()
    
    # NEW: Shopify fetch lock - NEVER interrupted
    shopify_lock = asyncio.Lock()
    shopify_fetch_in_progress = False

    # Catalog knowledge shared across responses
    catalog_context = None
    
    # IMPROVED: Better interruption handling state
    is_agent_speaking = False
    current_agent_task = None  # Track current agent response task
    interrupt_flag = asyncio.Event()
    
    # UPDATED: More robust interruption thresholds
    speech_energy_threshold = 1500.0  # INCREASED from 1000 to reduce false interruptions
    consecutive_speech_frames = 0
    speech_frames_needed = 8  # Require more sustained energy to mark interruptions
    
    # UPDATED: Better silence detection for complete utterances
    silence_frames_to_process = 22  # Wait ~2.2s of silence to avoid cutting callers off
    silence_frame_count = 0
    
    # NEW: Track if we're accumulating a multi-sentence utterance
    utterance_started = False
    utterance_start_time = 0
    max_utterance_duration = 15.0  # Max 15 seconds for a single utterance
    post_speech_grace_seconds = 1.2  # Wait at least 1.2s after user speech before responding
    
    try:
        while not stop_received:
            try:
                # Receive WebSocket message with timeout
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
                    params = start_info.get("customParameters", {})
                    caller_number = params.get("from", "Unknown")
                    called_number = params.get("to", "Unknown")
                    caller_email = params.get("email", "Not provided")

                    start_time = time.time()
                    last_user_speech_time = start_time
                    
                    print(f"🎬 Stream started")
                    print(f"   Call SID: {call_sid}")
                    print(f"   Stream SID: {stream_sid}")
                    print(f"   From: {caller_number}")
                    print(f"   To: {called_number}")
                    print(f"   Email: {caller_email or 'Not provided'}")

                    if shopify_client and catalog_context is None:
                        catalog_context = await get_catalog_snapshot()
                    
                    # CRITICAL: Fetch customer by phone number IMMEDIATELY
                    if shopify_client and caller_number and caller_number != "Unknown":
                        asyncio.create_task(
                            fetch_customer_by_phone(
                                caller_number, 
                                call_sid,
                                caller_memory,
                                shopify_lock
                            )
                        )
                    
                    # Send greeting immediately
                    if not greeting_sent and stream_sid:
                        asyncio.create_task(
                            send_greeting(
                                websocket, 
                                stream_sid, 
                                conversation_history, 
                                interrupt_flag,
                                is_agent_speaking,
                                lambda v: setattr_wrapper(locals(), 'is_agent_speaking', v)
                            )
                        )
                        greeting_sent = True
                
                # ==================== MEDIA EVENT ====================
                elif event_type == "media":
                    media = data.get("media", {})
                    audio_payload = media.get("payload")
                    
                    if not audio_payload:
                        continue
                    
                    # Decode incoming μ-law audio from Twilio
                    mulaw_bytes = base64.b64decode(audio_payload)
                    
                    # Calculate energy for speech detection
                    energy = calculate_audio_energy(mulaw_bytes)
                    
                    # IMPROVED: More robust interruption detection
                    # BUT: Never interrupt if Shopify fetch is in progress
                    if is_agent_speaking and not shopify_fetch_in_progress:
                        if energy > speech_energy_threshold:
                            consecutive_speech_frames += 1
                            
                            # UPDATED: Require sustained speech to interrupt
                            if consecutive_speech_frames >= speech_frames_needed:
                                # INTERRUPTION DETECTED!
                                print(f"🛑 INTERRUPTION DETECTED (energy: {energy:.0f}, frames: {consecutive_speech_frames})")
                                
                                # Cancel current agent response
                                if current_agent_task and not current_agent_task.done():
                                    current_agent_task.cancel()
                                    print("🚫 Cancelled current agent response")
                                
                                interrupt_flag.set()
                                is_agent_speaking = False
                                consecutive_speech_frames = 0
                                
                                # Send clear command to stop audio playback
                                try:
                                    await websocket.send_json({
                                        "event": "clear",
                                        "streamSid": stream_sid
                                    })
                                    print("🔇 Sent clear command to stop agent audio")
                                except Exception as clear_error:
                                    print(f"⚠️ Failed to send clear command: {clear_error}")
                                
                                # Clear audio buffer to start fresh
                                audio_buffer.clear()
                                silence_frame_count = 0
                                utterance_started = False
                        else:
                            # Decay counter gradually
                            consecutive_speech_frames = max(0, consecutive_speech_frames - 1)
                    elif shopify_fetch_in_progress and energy > speech_energy_threshold:
                        # User is trying to interrupt during Shopify fetch
                        # Just log it, don't cancel anything
                        print(f"⏸️ User speaking during Shopify fetch (energy: {energy:.0f})")
                    else:
                        consecutive_speech_frames = 0
                        
                        # IMPROVED: Better utterance accumulation
                        if energy > speech_energy_threshold:
                            # User is speaking
                            if not utterance_started:
                                utterance_started = True
                                utterance_start_time = time.time()
                                print(f"🎙️ Utterance started (energy: {energy:.0f})")
                            
                            audio_buffer.append(mulaw_bytes)
                            silence_frame_count = 0
                            last_user_speech_time = time.time()
                        else:
                            # Silence detected
                            if utterance_started and len(audio_buffer) > 0:
                                audio_buffer.append(mulaw_bytes)
                                silence_frame_count += 1
                                
                                # Check if utterance has gone on too long (safety timeout)
                                utterance_duration = time.time() - utterance_start_time
                                force_process = utterance_duration > max_utterance_duration
                                
                                # Process when we have enough silence after speech OR timeout
                                if silence_frame_count >= silence_frames_to_process or force_process:
                                    # Only process if not already processing
                                    if not processing_lock.locked():
                                        combined_mulaw = b''.join(audio_buffer)

                                        if not force_process:
                                            time_since_last_speech = time.time() - last_user_speech_time
                                            if time_since_last_speech < post_speech_grace_seconds:
                                                # Wait a bit longer before responding
                                                continue

                                        audio_buffer.clear()
                                        silence_frame_count = 0
                                        utterance_started = False
                                        
                                        if force_process:
                                            print(f"⏱️ Force processing after {utterance_duration:.1f}s")
                                        
                                        # Skip if too short (less than 0.5 seconds)
                                        if len(combined_mulaw) >= 4000:
                                            print(f"🎤 Processing complete utterance: {len(combined_mulaw)} bytes ({len(combined_mulaw)/8000:.1f}s)")
                                            
                                            # Process audio with lock to prevent concurrent processing
                                            task = asyncio.create_task(
                                                process_audio_chunk_fast(
                                                    combined_mulaw,
                                                    websocket,
                                                    stream_sid,
                                                    conversation_history,
                                                    caller_email,
                                                    call_sid,
                                                    time.time(),
                                                    interrupt_flag,
                                                    processing_lock,
                                                    caller_memory,  # NEW: Pass caller memory
                                                    shopify_lock,  # NEW: Pass shopify lock
                                                    lambda v: setattr_wrapper(locals(), 'shopify_fetch_in_progress', v),
                                                    catalog_context
                                                )
                                            )
                                            
                                            # Store task reference for potential cancellation
                                            current_agent_task = task
                                            is_agent_speaking = True
                                        else:
                                            print(f"⏭️ Skipping short audio: {len(combined_mulaw)} bytes")
                                            audio_buffer.clear()
                                            silence_frame_count = 0
                                            utterance_started = False
                
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
        
        # Cancel any ongoing agent speech
        if current_agent_task and not current_agent_task.done():
            current_agent_task.cancel()
        interrupt_flag.set()
        is_agent_speaking = False
        
        # Process any remaining audio
        if audio_buffer and len(audio_buffer) > 0:
            try:
                combined_mulaw = b''.join(audio_buffer)
                if len(combined_mulaw) > 8000:
                    pcm_audio = mulaw_to_linear_pcm(combined_mulaw)
                    wav_audio = create_wav_header(len(pcm_audio), sample_rate=8000) + pcm_audio
                    transcript = await transcribe_audio(wav_audio)
                    if transcript and transcript.strip() and len(transcript.strip()) > 2:
                        conversation_history.append({
                            "role": "user",
                            "text": transcript,
                            "timestamp": time.time()
                        })
                        print(f"👤 User (final): '{transcript}'")
            except Exception as e:
                print(f"⚠️ Final audio processing error: {e}")
        
        # NEW: Tag customer if they called and are verified
        if shopify_client and caller_memory.get("customer_id"):
            try:
                customer_id = caller_memory["customer_id"]
                tags = ["ai_call_inquiry"]
                
                # Add specific tags based on what was discussed
                if caller_memory.get("verified_orders"):
                    tags.append("order_inquiry")
                if caller_memory.get("issues"):
                    tags.append("issue_reported")
                if caller_memory.get("art_interests"):
                    tags.append("art_interest")
                
                print(f"🏷️ Tagging customer {customer_id} with: {tags}")
                shopify_client.tag_customer_or_order(customer_id=customer_id, tags=tags)
                print(f"✅ Customer tagged successfully")
            except Exception as e:
                print(f"⚠️ Error tagging customer: {e}")
        
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

        #Create Zendesk ticket (COMMENTED OUT - uncomment if needed)
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
                
                # Log ticket to database
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
                traceback.print_exc()
        
        print(f"✅ Call session complete: {call_sid}")
        print(f"   Duration: {duration}s")
        print(f"   Messages: {len(conversation_history)}")
        
        # NEW: Print caller memory summary
        print(f"🧠 Caller Memory Summary:")
        print(f"   Customer ID: {caller_memory.get('customer_id')}")
        print(f"   Customer Name: {caller_memory.get('name')}")
        print(f"   Order Numbers: {caller_memory['order_numbers']}")
        print(f"   Verified Orders: {list(caller_memory['verified_orders'].keys())}")
        print(f"   Art Interests: {caller_memory['art_interests']}")
        print(f"   Issues: {caller_memory['issues']}")
        
        # Clean up Shopify cache for this call
        if call_sid in _shopify_cache:
            del _shopify_cache[call_sid]

def setattr_wrapper(local_dict, var_name, value):
    """Helper to update local variables"""
    local_dict[var_name] = value

async def fetch_customer_by_phone(phone: str, call_sid: str, caller_memory: dict, shopify_lock: asyncio.Lock):
    """
    NEW: Fetch customer by phone number IMMEDIATELY at call start
    This task is NON-INTERRUPTIBLE
    """
    async with shopify_lock:
        global _shopify_cache
        try:
            if shopify_client:
                print(f"📞 Fetching customer by phone: {phone}...")
                
                # Run in executor to avoid blocking
                customer = await asyncio.get_event_loop().run_in_executor(
                    None,
                    shopify_client.get_customer_by_phone,
                    phone
                )
                
                if customer:
                    customer_id = customer.get("id")
                    customer_email = customer.get("email")
                    customer_first_name = customer.get("first_name")
                    customer_last_name = customer.get("last_name")
                    
                    print(f"✅ Customer found: {customer_email} (ID: {customer_id})")
                    
                    # Store in caller memory
                    caller_memory["customer_id"] = customer_id
                    caller_memory["email"] = customer_email
                    caller_memory["name"] = customer_first_name
                    caller_memory["customer_info"] = customer
                    
                    # Fetch all orders for this customer using customer_id
                    print(f"🛒 Fetching orders for customer ID {customer_id}...")
                    orders = await asyncio.get_event_loop().run_in_executor(
                        None,
                        shopify_client.get_customer_orders,
                        customer_id
                    )
                    
                    if orders and len(orders) > 0:
                        # Cache all orders
                        _shopify_cache[call_sid] = {
                            "customer": customer,
                            "orders": orders
                        }
                        
                        # Store orders in caller memory
                        caller_memory["all_orders"] = orders
                        
                        # Extract order numbers (names) for quick reference
                        for order in orders:
                            # Order "name" is like "#ABM27255X" or "ABM27255X"
                            order_name = order.get("name", "").replace("#", "").strip()
                            if order_name and order_name not in caller_memory["order_numbers"]:
                                caller_memory["order_numbers"].append(order_name)
                        
                        print(f"✅ Cached {len(orders)} orders for call {call_sid}")
                        print(f"   Order names: {caller_memory['order_numbers']}")
                    else:
                        _shopify_cache[call_sid] = {
                            "customer": customer,
                            "orders": []
                        }
                        caller_memory["all_orders"] = []
                        print(f"ℹ️  Customer has no orders")
                else:
                    _shopify_cache[call_sid] = None
                    print(f"⚠️ No customer found for phone: {phone}")
        
        except Exception as e:
            print(f"⚠️ Error fetching customer by phone: {e}")
            traceback.print_exc()
            _shopify_cache[call_sid] = None


async def fetch_order_details(order_name: str, customer_id: Optional[int], caller_memory: dict, shopify_lock: asyncio.Lock) -> Optional[dict]:
    """
    NEW: Fetch and verify specific order details
    This task is NON-INTERRUPTIBLE
    """
    async with shopify_lock:
        try:
            # Check if already verified
            if order_name in caller_memory.get("verified_orders", {}):
                print(f"✅ Order {order_name} already verified (cached)")
                return caller_memory["verified_orders"][order_name]
            
            if shopify_client:
                print(f"🔍 Verifying order {order_name}...")
                
                # Verify order ownership if customer_id provided
                order = await asyncio.get_event_loop().run_in_executor(
                    None,
                    shopify_client.verify_order_number,
                    order_name,
                    customer_id
                )
                
                if order:
                    # Get detailed order information
                    order_details = await asyncio.get_event_loop().run_in_executor(
                        None,
                        shopify_client.get_order_details,
                        order_name
                    )
                    
                    if order_details:
                        # Cache verified order
                        caller_memory.setdefault("verified_orders", {})[order_name] = order_details
                        print(f"✅ Order {order_name} verified and cached")
                        return order_details
                    else:
                        print(f"⚠️ Could not fetch details for order {order_name}")
                        return None
                else:
                    print(f"⚠️ Order {order_name} not found or doesn't belong to customer")
                    return None
            
            return None
        
        except Exception as e:
            print(f"⚠️ Error fetching order details: {e}")
            traceback.print_exc()
            return None


async def get_catalog_snapshot(force: bool = False):
    """
    Fetch and cache Shopify catalog overview for conversational context.
    """
    global _catalog_snapshot, _catalog_lock

    if not shopify_client:
        return None

    now = time.time()
    cached = _catalog_snapshot.get("data")
    if (
        cached
        and not force
        and now - _catalog_snapshot.get("timestamp", 0) < CATALOG_REFRESH_SECONDS
    ):
        return cached

    if _catalog_lock is None:
        _catalog_lock = asyncio.Lock()

    async with _catalog_lock:
        cached = _catalog_snapshot.get("data")
        if (
            cached
            and not force
            and time.time() - _catalog_snapshot.get("timestamp", 0) < CATALOG_REFRESH_SECONDS
        ):
            return cached

        try:
            loop = asyncio.get_event_loop()
            overview = await loop.run_in_executor(
                None,
                shopify_client.get_catalog_overview
            )
            if overview:
                _catalog_snapshot["data"] = overview
                _catalog_snapshot["timestamp"] = time.time()
                return overview
        except Exception as e:
            print(f"⚠️ Catalog snapshot error: {e}")

    return _catalog_snapshot.get("data")

async def send_greeting(websocket: WebSocket, stream_sid: str, conversation_history: list, 
                       interrupt_flag: asyncio.Event, is_speaking: bool, set_speaking):
    """Send greeting with minimal delay and interruption support"""
    try:
        print("🎤 Generating greeting...")
        
        # Generate greeting WITHOUT customer context to avoid confusion
        ai_text, ai_audio_8khz_pcm = await generate_ai_response_live(
            "Generate a brief, friendly greeting for an art gallery customer. Keep it under 10 words.",
            caller_email=None,
            shopify_client=None,
            shopify_data_cache=None,
            caller_memory=None  # NEW: No memory for greeting
        )
        
        # Log greeting
        if ai_text:
            conversation_history.append({
                "role": "assistant",
                "text": ai_text,
                "timestamp": time.time()
            })
        
        # Send audio if available
        if ai_audio_8khz_pcm and len(ai_audio_8khz_pcm) > 100:
            # Convert 8kHz PCM to μ-law for Twilio
            mulaw_audio = linear_pcm_to_mulaw(ai_audio_8khz_pcm)
            
            if mulaw_audio and len(mulaw_audio) > 0:
                interrupt_flag.clear()
                
                payload = base64.b64encode(mulaw_audio).decode("utf-8")
                
                # Check connection before sending
                try:
                    await websocket.send_json({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload}
                    })
                    print(f"🔊 Greeting sent: {len(mulaw_audio)} bytes μ-law")
                    
                    # Wait for audio to finish or be interrupted
                    audio_duration = len(mulaw_audio) / 8000.0
                    try:
                        await asyncio.wait_for(interrupt_flag.wait(), timeout=audio_duration)
                        print("🛑 Greeting interrupted")
                    except asyncio.TimeoutError:
                        pass  # Audio finished naturally
                    
                except Exception as send_error:
                    print(f"❌ Failed to send greeting: {send_error}")
            else:
                print("❌ Greeting μ-law conversion failed")
        else:
            print(f"⚠️ No greeting audio generated")
                    
    except Exception as e:
        print(f"⚠️ Greeting generation error: {e}")
        traceback.print_exc()

async def process_audio_chunk_fast(
    mulaw_audio: bytes,
    websocket: WebSocket,
    stream_sid: str,
    conversation_history: list,
    caller_email: str,
    call_sid: str,
    timestamp: float,
    interrupt_flag: asyncio.Event,
    processing_lock: asyncio.Lock,
    caller_memory: dict,  # NEW: Persistent caller memory
    shopify_lock: asyncio.Lock,  # NEW: Shopify lock
    set_shopify_fetch_in_progress,  # NEW: Callback to set shopify fetch status
    catalog_context: Optional[dict] = None  # NEW: Catalog snapshot for product info
):
    """Process audio chunk with IMPROVED INTERRUPTION HANDLING and CALLER MEMORY"""
    
    # Acquire lock to prevent concurrent processing
    async with processing_lock:
        try:
            chunk_start = time.time()
            print(f"🎤 Processing {len(mulaw_audio)} bytes μ-law audio...")
            
            # Convert μ-law to 16-bit PCM
            pcm_audio = mulaw_to_linear_pcm(mulaw_audio)
            
            if not pcm_audio or len(pcm_audio) < 2000:
                print(f"⚠️ PCM conversion failed or too short")
                return
            
            # Create WAV file for Whisper (8kHz)
            wav_audio = create_wav_header(len(pcm_audio), sample_rate=8000) + pcm_audio
            
            # Transcribe
            transcript = await transcribe_audio(wav_audio)
            
            # CRITICAL: Better validation to avoid "Thank you" problem
            if not transcript or len(transcript.strip()) < 3:
                print(f"⏭️ Empty or invalid transcript")
                return
            
            transcript = transcript.strip()
            
            # CRITICAL: Filter out common transcription errors that cause loops
            low_transcript = transcript.lower()
            if low_transcript in ["thank you", "thank you.", "thanks", "thanks."]:
                print(f"⚠️ Filtered Whisper hallucination: '{transcript}'")
                return
            
            print(f"👤 User said: '{transcript}'")
            
            # Add to conversation history
            conversation_history.append({
                "role": "user",
                "text": transcript,
                "timestamp": timestamp
            })
            
            # Generate AI response with memory and Shopify integration
            try:
                # Get cached Shopify data
                global _shopify_cache
                cached_shopify = _shopify_cache.get(call_sid)
                
                # NEW: Check if we need to fetch order details
                import re
                order_pattern = r'\b([A-Z]{3}\d{5}[A-Z])\b'  # Match patterns like ABM27255X
                potential_order_numbers = re.findall(order_pattern, transcript, re.IGNORECASE)
                
                # Also check for plain numbers that might be order references
                if not potential_order_numbers:
                    number_pattern = r'\b(\d{5,6})\b'
                    potential_numbers = re.findall(number_pattern, transcript)
                    if potential_numbers and caller_memory.get("order_numbers"):
                        # Check if any mentioned number matches part of an existing order
                        for num in potential_numbers:
                            for order_name in caller_memory["order_numbers"]:
                                if num in order_name:
                                    potential_order_numbers.append(order_name)
                                    break
                
                order_context = None
                if potential_order_numbers and shopify_client:
                    # User mentioned an order number - fetch details NON-INTERRUPTIBLY
                    order_num = potential_order_numbers[0].upper()
                    customer_id = caller_memory.get("customer_id")
                    
                    # Check if order already verified
                    if order_num not in caller_memory.get("verified_orders", {}):
                        print(f"🛒 Order {order_num} mentioned - fetching details...")
                        
                        # Set flag to prevent interruption
                        set_shopify_fetch_in_progress(True)
                        
                        # Inform user we're fetching (send quick audio message)
                        try:
                            fetch_message = "Let me check that order for you."
                            from app.services.ai_service import get_elevenlabs_client
                            eleven_client = get_elevenlabs_client()
                            fetch_audio = await eleven_client.text_to_speech_fast(fetch_message)
                            
                            if fetch_audio:
                                mulaw_fetch = linear_pcm_to_mulaw(fetch_audio)
                                if mulaw_fetch:
                                    await websocket.send_json({
                                        "event": "media",
                                        "streamSid": stream_sid,
                                        "media": {"payload": base64.b64encode(mulaw_fetch).decode("utf-8")}
                                    })
                                    print("🔊 Sent 'checking order' message")
                                    
                                    # Wait for message to play
                                    await asyncio.sleep(len(mulaw_fetch) / 8000.0)
                        except Exception as msg_error:
                            print(f"⚠️ Could not send fetch message: {msg_error}")
                        
                        # Fetch order details (NON-INTERRUPTIBLE)
                        order_details = await fetch_order_details(
                            order_num, 
                            customer_id, 
                            caller_memory, 
                            shopify_lock
                        )
                        
                        # Clear flag
                        set_shopify_fetch_in_progress(False)
                        
                        if order_details:
                            order_context = order_details
                            print(f"✅ Order details fetched: {order_details.get('order_number')}")
                        else:
                            # Order not found or doesn't belong to customer
                            if customer_id:
                                print(f"⚠️ Order {order_num} not found for customer")
                            else:
                                print(f"⚠️ Order {order_num} not found")
                    else:
                        # Use cached order
                        order_context = caller_memory["verified_orders"][order_num]
                        print(f"✅ Using cached order: {order_num}")
                
                print("🤖 Generating AI response...")

                product_suggestions = None
                if catalog_context:
                    product_suggestions = search_catalog_for_request(transcript, catalog_context)

                ai_text, ai_audio_8khz_pcm = await generate_ai_response_live(
                    transcript,  # Pass only current transcript
                    caller_email=caller_email,
                    shopify_client=None,  # Don't pass client to avoid API calls
                    shopify_data_cache=cached_shopify,
                    caller_memory=caller_memory,  # NEW: Pass caller memory
                    conversation_history=conversation_history,  # NEW: Pass full conversation history
                    order_context=order_context,  # NEW: Pass specific order context if fetched
                    catalog_context=catalog_context,  # NEW: Product knowledge
                    catalog_suggestions=product_suggestions
                )
                
                if not ai_text or len(ai_text.strip()) < 2:
                    print("⚠️ No AI response generated")
                    return
                
                # Add AI response to history
                conversation_history.append({
                    "role": "assistant",
                    "text": ai_text,
                    "timestamp": time.time()
                })
                
                print(f"🤖 AI responds: '{ai_text}'")
                
                # Send audio response with interruption handling
                if ai_audio_8khz_pcm and len(ai_audio_8khz_pcm) > 100 and stream_sid:
                    # Audio is already 8kHz PCM, just convert to μ-law
                    mulaw_audio = linear_pcm_to_mulaw(ai_audio_8khz_pcm)
                    
                    if mulaw_audio and len(mulaw_audio) > 0:
                        interrupt_flag.clear()
                        
                        payload = base64.b64encode(mulaw_audio).decode("utf-8")
                        
                        try:
                            await websocket.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": payload}
                            })
                            
                            total_time = time.time() - chunk_start
                            print(f"🔊 AI audio sent: {len(mulaw_audio)} bytes μ-law")
                            print(f"⏱️ Total response time: {total_time:.2f}s")
                            
                            # Wait for audio to finish or be interrupted
                            audio_duration = len(mulaw_audio) / 8000.0  # 8kHz μ-law
                            try:
                                await asyncio.wait_for(interrupt_flag.wait(), timeout=audio_duration)
                                print("🛑 Agent speech interrupted by caller")
                            except asyncio.TimeoutError:
                                pass  # Audio finished naturally
                            
                        except asyncio.CancelledError:
                            print("🚫 Audio send cancelled (interruption)")
                            raise
                        except Exception as send_error:
                            print(f"❌ Failed to send AI audio: {send_error}")
                    else:
                        print("❌ AI audio μ-law conversion failed")
                else:
                    print(f"⚠️ No AI audio to send")
            
            except asyncio.CancelledError:
                print("🚫 Response generation cancelled (interruption)")
                raise
            except Exception as e:
                print(f"❌ AI response error: {e}")
                traceback.print_exc()
        
        except asyncio.CancelledError:
            print("🚫 Audio processing cancelled (interruption)")
            # Don't re-raise, just exit cleanly
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            traceback.print_exc()
