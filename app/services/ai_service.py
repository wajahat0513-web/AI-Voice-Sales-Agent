# app/services/ai_service.py

import io
import asyncio
from typing import Tuple, Optional
from app.core.config import settings
from app.services.shopify_client import ShopifyClient
from app.core.elevenlabs_client import ElevenLabsClient
from openai import OpenAI
import time

# Initialize OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Persistent ElevenLabs client
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
                prompt="Customer calling art gallery about products or orders."
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
    - Short, natural responses (max 25 words)
    """
    start_time = time.time()
    
    # Build customer info from cache
    customer_orders_info = ""
    if shopify_data_cache:
        order = shopify_data_cache
        order_name = order.get('name', 'N/A')
        order_total = order.get('total_price', 'N/A')
        fulfillment = order.get('fulfillment_status', 'unknown')
        customer_orders_info = f"""
Customer Order: {order_name} - ${order_total} ({fulfillment} status)
"""
    
    # Optimized system prompt for phone conversations
    system_prompt = f"""You are a friendly AI sales agent for ArtByMaudsch.com, an art gallery.

Customer Info:
- Email: {caller_email or 'Not provided'}
{customer_orders_info}

CRITICAL - Phone Call Rules:
1. Keep responses EXTREMELY SHORT (15-25 words maximum)
2. Sound natural and conversational like talking to a friend
3. Use contractions (I'm, we'll, you're, that's)
4. Be direct - one main point per response
5. If complex, offer to email details instead
6. Never read lists over the phone
7. Use simple, everyday language

Your Role:
- Answer questions about products, orders, shipping
- Track orders and provide updates
- Handle complaints with empathy
- Recommend products when relevant
- Escalate complex issues to humans

Examples of GOOD responses:
"Sure! Your order ships tomorrow. You'll get tracking by email."
"I'd love to help! What brings you in today?"
"That painting's available. Want me to email you details?"

Remember: This is a PHONE CALL. Be brief, warm, and helpful!"""

    # Generate AI text with aggressive token limits
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
                temperature=0.7,
                max_tokens=40,  # Limit token count
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
        )

        ai_text = response.choices[0].message.content.strip()
        
        # Force truncate if too long
        words = ai_text.split()
        if len(words) > 25:
            ai_text = ' '.join(words[:25])
            # Add period if not already ending with punctuation
            if ai_text[-1] not in '.!?':
                ai_text += '.'
        
        gpt_time = time.time() - start_time
        print(f"✅ AI text ({gpt_time:.2f}s, {len(words)} words): '{ai_text}'")

    except Exception as e:
        print(f"⚠️ AI generation error: {e}")
        ai_text = "I'm here to help. Could you repeat that?"

    # Convert to speech IMMEDIATELY
    ai_audio_8khz_pcm = None
    if ai_text:
        try:
            tts_start = time.time()
            eleven_client = get_elevenlabs_client()
            
            # Generate TTS
            raw_audio_bytes = await eleven_client.text_to_speech_fast(ai_text)
            
            if raw_audio_bytes and len(raw_audio_bytes) > 100:
                ai_audio_8khz_pcm = raw_audio_bytes
                tts_time = time.time() - tts_start
                print(f"✅ TTS ({tts_time:.2f}s): {len(ai_audio_8khz_pcm)} bytes")
            else:
                print(f"⚠️ TTS returned insufficient audio")

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
    ai_text, _ = await generate_ai_response_live(
        transcript_buffer, 
        caller_email, 
        shopify_client
    )
    return ai_text

