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


def resample_audio_safe(pcm_data: bytes, from_rate: int,
                        to_rate: int) -> bytes:
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


def create_wav_header(data_size: int,
                      sample_rate: int = 8000,
                      channels: int = 1,
                      bits_per_sample: int = 16) -> bytes:
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

        print(
            f"📞 Incoming call: {caller_number} → {called_number} (SID: {call_sid})"
        )

        # Get WebSocket URL from environment
        websocket_url = os.getenv(
            "WEBSOCKET_URL",
            "wss://0d9dcab3-940b-48ee-a63c-19a11c56256a-00-36ialn5of1nja.sisko.replit.dev/api/twilio/media-stream"
        )
        print(f"🌐 Using WebSocket URL: {websocket_url}")

        # Create TwiML response
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
        error_response.say(
            "Sorry, something went wrong. Please try again later.")
        return Response(content=str(error_response),
                        media_type="application/xml")


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
                db_client.insert_call_metadata(call_sid=call_sid,
                                               call_status=call_status,
                                               duration=0,
                                               call_type="inbound",
                                               tags=[call_status])
            except Exception as e:
                print(f"⚠️ DB status log error: {e}")

        return {"status": "received"}

    except Exception as e:
        print(f"❌ Error in status_webhook: {e}")
        return {"status": "error", "error": str(e)}


# Global cache for Shopify data per call
_shopify_cache = {}

# ============================================================
# WEBSOCKET MEDIA STREAM - ULTRA-LOW LATENCY
# ============================================================


@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle real-time audio streaming from Twilio with MINIMAL LATENCY"""
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
    last_processing_time = 0
    processing_interval = 2.0  # REDUCED to 2 seconds for faster response
    greeting_sent = False
    stop_received = False
    is_processing = False  # Prevent concurrent processing

    try:
        while not stop_received:
            try:
                # Receive WebSocket message with timeout
                try:
                    data = await asyncio.wait_for(websocket.receive_json(),
                                                  timeout=20.0)
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
                    call_sid = start_info.get("callSid",
                                              f"UNKNOWN_{int(time.time())}")
                    stream_sid = start_info.get("streamSid")
                    caller_number = start_info.get(
                        "customParameters", {}).get("From") or start_info.get(
                            "from", "Unknown")
                    called_number = start_info.get(
                        "customParameters", {}).get("To") or start_info.get(
                            "to", "Unknown")
                    custom_params = start_info.get("customParameters", {})
                    caller_email = custom_params.get("caller_email")
                    start_time = time.time()
                    last_processing_time = start_time

                    print(f"🎬 Stream started")
                    print(f"   Call SID: {call_sid}")
                    print(f"   Stream SID: {stream_sid}")
                    print(f"   From: {caller_number}")
                    print(f"   To: {called_number}")
                    print(f"   Email: {caller_email or 'Not provided'}")

                    # CRITICAL: Fetch Shopify data ONCE and cache it
                    if shopify_client and caller_email:
                        asyncio.create_task(
                            fetch_and_cache_shopify(caller_email, call_sid))

                    # Send greeting immediately WITHOUT Shopify data
                    if not greeting_sent and stream_sid:
                        asyncio.create_task(
                            send_greeting(websocket, stream_sid,
                                          conversation_history))
                        greeting_sent = True

                # ==================== MEDIA EVENT ====================
                elif event_type == "media":
                    media = data.get("media", {})
                    audio_payload = media.get("payload")

                    if not audio_payload:
                        continue

                    # Decode incoming μ-law audio from Twilio
                    mulaw_bytes = base64.b64decode(audio_payload)
                    audio_buffer.append(mulaw_bytes)

                    # Process audio at intervals (but not if already processing)
                    current_time = time.time()
                    if (current_time - last_processing_time
                            >= processing_interval and not is_processing
                            and len(audio_buffer) > 0):

                        is_processing = True
                        last_processing_time = current_time

                        # Combine buffered audio
                        combined_mulaw = b''.join(audio_buffer)
                        audio_buffer.clear()

                        # Skip if too short (less than 0.4 seconds)
                        if len(combined_mulaw) < 3000:
                            print(
                                f"⏭️ Skipping short audio: {len(combined_mulaw)} bytes"
                            )
                            is_processing = False
                            continue

                        # Process in background to not block receiving
                        asyncio.create_task(
                            process_audio_chunk_fast(combined_mulaw, websocket,
                                                     stream_sid,
                                                     conversation_history,
                                                     caller_email, call_sid,
                                                     current_time))

                        # Reset processing flag after a short delay
                        await asyncio.sleep(0.3)
                        is_processing = False

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
                    wav_audio = create_wav_header(len(pcm_audio),
                                                  sample_rate=8000) + pcm_audio
                    transcript = await transcribe_audio(wav_audio)
                    if transcript and transcript.strip():
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
                db_client.insert_call_transcript(call_sid=call_sid,
                                                 from_number=caller_number
                                                 or "Unknown",
                                                 to_number=called_number
                                                 or "Unknown",
                                                 transcript=full_transcript,
                                                 summary=summary)
                print("✅ Call transcript logged to database")

                db_client.insert_call_metadata(
                    call_sid=call_sid,
                    call_status="completed",
                    duration=duration,
                    call_type="inbound",
                    tags=["completed", "ai_handled"])
                print("✅ Call metadata logged to database")
            except Exception as e:
                print(f"⚠️ Database logging error: {e}")

        # Create Zendesk ticket 
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
                    tags=["ai_call", "phone_support", "automated"])

                print(f"🎫 Zendesk ticket created: #{ticket_id}")

                # Log ticket to database
                if db_client:
                    try:
                        db_client.insert_zendesk_ticket(
                            call_sid=call_sid, zendesk_ticket_id=ticket_id)
                    except Exception as e:
                        if "duplicate key" not in str(e).lower():
                            print(f"⚠️ Zendesk ticket logging error: {e}")

            except Exception as e:
                print(f"⚠️ Zendesk ticket creation error: {e}")
                traceback.print_exc()

        # Tag Shopify customer and orders after call
        if shopify_client and (caller_email or caller_number):
            try:
                customer_id = None
                customer_orders = []

                # Try to find customer by email first
                if caller_email:
                    print(f"🛒 Looking up Shopify customer by email: {caller_email}")
                    customer_orders = shopify_client.get_customer_orders(caller_email)
                    if customer_orders and len(customer_orders) > 0:
                        # Extract customer ID from first order
                        customer_id = customer_orders[0].get("customer", {}).get("id")

                # Fallback: Try phone number if email didn't work
                if not customer_id and caller_number:
                    print(f"🛒 Looking up Shopify customer by phone: {caller_number}")
                    customer_data = shopify_client.get_customer_by_phone(caller_number)
                    if customer_data:
                        customer_id = customer_data.get("id")
                        # Get orders for this customer
                        try:
                            orders_url = f"{shopify_client.base_url}/orders.json"
                            orders_params = {"customer_id": customer_id, "status": "any", "limit": 5}
                            orders_response = shopify_client.session.get(orders_url, params=orders_params)
                            orders_response.raise_for_status()
                            customer_orders = orders_response.json().get("orders", [])
                        except Exception as e:
                            print(f"⚠️ Could not fetch orders for customer: {e}")

                # Tag customer if found
                if customer_id:
                    call_date = time.strftime("%Y-%m-%d")
                    customer_tags = [
                        "ai_call",
                        "phone_support",
                        f"called_{call_date}",
                        "ai_handled"
                    ]

                    tag_success = shopify_client.tag_customer_or_order(
                        customer_id=customer_id,
                        tags=customer_tags
                    )

                    if tag_success:
                        print(f"✅ Tagged Shopify customer {customer_id}")

                    # Tag the most recent order if exists
                    if customer_orders and len(customer_orders) > 0:
                        most_recent_order = customer_orders[0]
                        order_id = most_recent_order.get("id")
                        order_name = most_recent_order.get("name", "Unknown")

                        order_tags = [
                            "ai_call_inquiry",
                            f"call_date_{call_date}"
                        ]

                        order_tag_success = shopify_client.tag_customer_or_order(
                            order_id=order_id,
                            tags=order_tags
                        )

                        if order_tag_success:
                            print(f"✅ Tagged Shopify order {order_name} (ID: {order_id})")
                else:
                    print(f"ℹ️ No Shopify customer found for {caller_email or caller_number}")

            except Exception as e:
                print(f"⚠️ Shopify tagging error: {e}")
                traceback.print_exc()

        print(f"✅ Call session complete: {call_sid}")
        print(f"   Duration: {duration}s")
        print(f"   Messages: {len(conversation_history)}")

        # Clean up Shopify cache for this call
        if call_sid in _shopify_cache:
            del _shopify_cache[call_sid]


async def fetch_and_cache_shopify(email: str, call_sid: str):
    """Fetch Shopify data ONCE and cache it for the entire call"""
    global _shopify_cache
    try:
        if shopify_client:
            print(f"🛒 Background Shopify fetch for {email}...")
            orders = await asyncio.get_event_loop().run_in_executor(
                None, shopify_client.get_customer_orders, email)
            if orders and len(orders) > 0:
                _shopify_cache[call_sid] = orders[0]
                print(
                    f"✅ Shopify: Cached order {orders[0].get('name')} for call {call_sid}"
                )
            else:
                _shopify_cache[call_sid] = None
                print(f"ℹ️  Shopify: No orders found for {email}")
    except Exception as e:
        print(f"⚠️ Background Shopify fetch error: {e}")
        _shopify_cache[call_sid] = None


async def send_greeting(websocket: WebSocket, stream_sid: str,
                        conversation_history: list):
    """Send greeting with minimal delay"""
    try:
        print("🎤 Generating greeting...")

        # Generate greeting WITHOUT customer context to avoid confusion
        ai_text, ai_audio_8khz_pcm = await generate_ai_response_live(
            "Generate a brief, friendly greeting for an art gallery customer. Keep it under 10 words.",
            caller_email=None,
            shopify_client=None,
            shopify_data_cache=None)

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
                payload = base64.b64encode(mulaw_audio).decode("utf-8")

                # Check connection before sending
                try:
                    await websocket.send_json({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {
                            "payload": payload
                        }
                    })
                    print(f"🔊 Greeting sent: {len(mulaw_audio)} bytes μ-law")
                except Exception as send_error:
                    print(f"❌ Failed to send greeting: {send_error}")
            else:
                print("❌ Greeting μ-law conversion failed")
        else:
            print(f"⚠️ No greeting audio generated")

    except Exception as e:
        print(f"⚠️ Greeting generation error: {e}")
        traceback.print_exc()


async def process_audio_chunk_fast(mulaw_audio: bytes, websocket: WebSocket,
                                   stream_sid: str, conversation_history: list,
                                   caller_email: str, call_sid: str,
                                   timestamp: float):
    """Process audio chunk asynchronously with MINIMAL LATENCY"""
    try:
        chunk_start = time.time()
        print(f"🎤 Processing {len(mulaw_audio)} bytes μ-law audio...")

        # Convert μ-law to 16-bit PCM
        pcm_audio = mulaw_to_linear_pcm(mulaw_audio)

        if not pcm_audio or len(pcm_audio) < 2000:
            print(f"⚠️ PCM conversion failed or too short")
            return

        # Create WAV file for Whisper (8kHz)
        wav_audio = create_wav_header(len(pcm_audio),
                                      sample_rate=8000) + pcm_audio

        # Transcribe
        transcript = await transcribe_audio(wav_audio)

        if not transcript or len(transcript.strip()) < 3:
            print(f"⏭️ Empty or invalid transcript")
            return

        transcript = transcript.strip()
        print(f"👤 User said: '{transcript}'")

        # Add to conversation history
        conversation_history.append({
            "role": "user",
            "text": transcript,
            "timestamp": timestamp
        })

        # Generate AI response
        try:
            # Build context from recent conversation (only last 4 messages)
            context_messages = []
            for msg in conversation_history[-4:]:
                role = msg['role']
                text = msg['text']
                context_messages.append(f"{role}: {text}")

            full_context = "\n".join(context_messages)

            # Get cached Shopify data
            global _shopify_cache
            cached_shopify = _shopify_cache.get(call_sid)

            print("🤖 Generating AI response...")
            ai_text, ai_audio_8khz_pcm = await generate_ai_response_live(
                full_context,
                caller_email=caller_email,
                shopify_client=None,  # Don't pass client to avoid API calls
                shopify_data_cache=cached_shopify)

            if not ai_text:
                print("⚠️ No AI response generated")
                return

            # Add AI response to history
            conversation_history.append({
                "role": "assistant",
                "text": ai_text,
                "timestamp": time.time()
            })

            print(f"🤖 AI responds: '{ai_text}'")

            # Send audio response
            if ai_audio_8khz_pcm and len(
                    ai_audio_8khz_pcm) > 100 and stream_sid:
                # Audio is already 8kHz PCM, just convert to μ-law
                mulaw_audio = linear_pcm_to_mulaw(ai_audio_8khz_pcm)

                if mulaw_audio and len(mulaw_audio) > 0:
                    payload = base64.b64encode(mulaw_audio).decode("utf-8")

                    try:
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": payload
                            }
                        })

                        total_time = time.time() - chunk_start
                        print(
                            f"🔊 AI audio sent: {len(mulaw_audio)} bytes μ-law")
                        print(f"⏱️ Total response time: {total_time:.2f}s")

                    except Exception as send_error:
                        print(f"❌ Failed to send AI audio: {send_error}")
                else:
                    print("❌ AI audio μ-law conversion failed")
            else:
                print(f"⚠️ No AI audio to send")

        except Exception as e:
            print(f"❌ AI response error: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        traceback.print_exc()
