# app/services/ai_service.py

import io
import asyncio
from typing import Tuple, Optional
from app.core.config import settings
from app.services.shopify_client import ShopifyClient
from app.core.elevenlabs_client import ElevenLabsClient
from openai import OpenAI
import time

# Initialize clients ONCE at module level
client = OpenAI(api_key=settings.OPENAI_API_KEY)
# Keep one ElevenLabs client alive for the entire session
_elevenlabs_client = None

def get_elevenlabs_client():
    """Get or create a persistent ElevenLabs client"""
    global _elevenlabs_client
    if _elevenlabs_client is None:
        _elevenlabs_client = ElevenLabsClient()
    return _elevenlabs_client


async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe caller audio using OpenAI Whisper.
    Expects WAV format audio (8kHz, 16-bit PCM).
    """
    try:
        if len(audio_bytes) < 2000:
            return ""

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "caller.wav"

        loop = asyncio.get_event_loop()
        transcript_resp = await loop.run_in_executor(
            None,
            lambda: client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="text",
                temperature=0.0,
                prompt="Brief customer inquiry about art products."  # Shorter prompt
            )
        )

        text = transcript_resp.strip() if isinstance(transcript_resp, str) else transcript_resp.text.strip()
        return text if text and len(text) > 2 else ""

    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return ""


async def summarize_conversation(transcript: str) -> str:
    """Summarize conversation for reporting."""
    try:
        if not transcript or not transcript.strip():
            return "No conversation to summarize."

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this call briefly in under 100 words."
                    },
                    {"role": "user", "content": transcript}
                ],
                temperature=0.3,
                max_tokens=150
            )
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ Summarization error: {e}")
        return "Summary unavailable."


async def generate_ai_response_live(
    transcript_buffer: str,
    caller_email: Optional[str] = None,
    shopify_client: Optional[ShopifyClient] = None,
    shopify_data_cache: Optional[dict] = None
) -> Tuple[str, Optional[bytes]]:
    """
    Generate AI response with MINIMAL LATENCY.
    Returns (text_response, audio_bytes_8khz_pcm).
    
    CRITICAL OPTIMIZATIONS:
    - Uses cached Shopify data (no API calls during conversation)
    - Parallel text and TTS generation
    - Persistent ElevenLabs client
    - Ultra-short responses (10 words max)
    """
    start_time = time.time()
    
    # Build system prompt with pre-fetched data
    customer_orders_info = ""
    if shopify_data_cache:
        order = shopify_data_cache
        customer_orders_info = f"""Customer has order {order.get('name', 'N/A')} - ${order.get('total_price', 'N/A')} ({order.get('fulfillment_status', 'unknown')} status)."""
    
    # ULTRA-SHORT system prompt for speed
    # Build system prompt
    system_prompt = f"""You are a friendly, professional AI sales agent for ArtByMaudsch.com, an online art gallery specializing in hand-painted artworks.


Customer Information:
- Email: {caller_email or 'Not provided yet'}
{customer_orders_info}


Your Role:
- Answer questions about products, orders, shipping, and payments
- Help track orders and provide updates
- Handle complaints with empathy and professionalism
- Provide product recommendations
- Escalate complex issues when needed


CRITICAL - Response Style for Phone Calls:
- Keep responses EXTREMELY SHORT (1-2 sentences, maximum 20 words)
- Sound natural and conversational, like talking to a friend
- Use contractions (I'm, we'll, you're)
- Avoid formal language or long explanations
- If customer asks something complex, offer to email details
- Never read long lists over the phone
- Be direct and to the point


Guidelines:
- Be warm and helpful
- Use customer's order info when relevant
- Be honest if you don't know something
- Offer to transfer to a human if needed
- BREVITY IS KEY - shorter is better


Remember: This is a PHONE call, not an email. Maximum 20 words per response!"""


    # Generate AI text with aggressive limits
    ai_text = ""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_buffer}
        ]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5,
                max_tokens=25,  # Force short responses
                presence_penalty=0.6,
                frequency_penalty=0.3,
                stop=["\n", ".", "!", "?"]  # Stop at first sentence
            )
        )

        ai_text = response.choices[0].message.content.strip()
        
        # Force truncate if still too long
        words = ai_text.split()
        if len(words) > 12:
            ai_text = ' '.join(words[:12]) + '.'
        
        gpt_time = time.time() - start_time
        print(f"✅ AI text ({gpt_time:.2f}s): '{ai_text}'")

    except Exception as e:
        print(f"⚠️ AI error: {e}")
        ai_text = "Let me help you."

    # Convert to speech IMMEDIATELY (no await)
    ai_audio_8khz_pcm = None
    if ai_text:
        try:
            tts_start = time.time()
            eleven_client = get_elevenlabs_client()  # Reuse persistent client
            
            # Start TTS immediately
            raw_audio_bytes = await eleven_client.text_to_speech_fast(ai_text)
            
            if raw_audio_bytes and len(raw_audio_bytes) > 100:
                ai_audio_8khz_pcm = raw_audio_bytes
                tts_time = time.time() - tts_start
                print(f"✅ TTS ({tts_time:.2f}s): {len(ai_audio_8khz_pcm)} bytes")
            else:
                print(f"⚠️ TTS returned short audio")

        except Exception as e:
            print(f"⚠️ TTS error: {e}")

    total_time = time.time() - start_time
    print(f"⏱️ Total response time: {total_time:.2f}s")
    
    return ai_text, ai_audio_8khz_pcm


async def generate_ai_text_only(
    transcript_buffer: str,
    caller_email: Optional[str] = None,
    shopify_client: Optional[ShopifyClient] = None
) -> str:
    """Generate AI response text without TTS."""
    ai_text, _ = await generate_ai_response_live(transcript_buffer, caller_email, shopify_client)
    return ai_text