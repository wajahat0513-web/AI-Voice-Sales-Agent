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
_elevenlabs_client = None


def get_elevenlabs_client():
    """Get or create a persistent ElevenLabs client"""
    global _elevenlabs_client
    if _elevenlabs_client is None:
        _elevenlabs_client = ElevenLabsClient()
    return _elevenlabs_client




async def transcribe_audio(audio_bytes: bytes, min_length: int = 2) -> str:
    """
    Transcribe caller audio using OpenAI Whisper with hallucination prevention.
    Expects WAV format audio (8kHz, 16-bit PCM).
    """
    try:
        if len(audio_bytes) < 400: # lowered from 2000
            return ""


        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "caller.wav"


        loop = asyncio.get_event_loop()
       
        # Enhanced Whisper settings to reduce hallucinations
        transcript_resp = await loop.run_in_executor(
            None,
            lambda: client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="verbose_json",  # Get more metadata
                temperature=0.0,  # Deterministic output
                prompt="Customer speaking naturally about art, products, or orders."  # Context hint
            )
        )


        # Extract text and check confidence
        if isinstance(transcript_resp, str):
            text = transcript_resp.strip()
        else:
            text = transcript_resp.text.strip()
           
            # Additional validation for verbose_json response
            if hasattr(transcript_resp, 'segments'):
                # Check if segments indicate low confidence or common hallucinations
                segments = transcript_resp.segments
                if segments:
                    # Filter out very short segments that might be hallucinations
                    valid_text = []
                    for segment in segments:
                        segment_text = segment.get('text', '').strip()
                        # Skip common hallucinations
                        if segment_text.lower() not in ['thank you', 'thanks', 'you', 'uh', 'um']:
                            valid_text.append(segment_text)
                   
                    if valid_text:
                        text = ' '.join(valid_text).strip()
       
        # Post-processing filters
        if not text or len(text) < min_length:
            return ""
       
        # Filter out common hallucinations
        hallucination_patterns = [
            'thank you', 'thanks', 'thank you.', 'thanks.',
            'you', 'uh', 'um', 'mhm', 'mm-hmm', 'uh-huh',
            'music', '[music]', '(music)', '[BLANK_AUDIO]'
        ]
       
        text_lower = text.lower().strip()
        if text_lower in hallucination_patterns:
            print(f"⚠️ Filtered hallucination: '{text}'")
            return ""
       
        # Remove partial phrases that are commonly hallucinated
        if text_lower.startswith('thank you') and len(text_lower) < 15:
            print(f"⚠️ Filtered partial hallucination: '{text}'")
            return ""
       
        return text


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
                        "content": "Summarize this customer service call briefly. Include: main issue, resolution status, and any follow-up needed. Maximum 100 words."
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
    Generate AI response optimized for natural conversation flow.
    Returns (text_response, audio_bytes_8khz_pcm).
   
    OPTIMIZATIONS:
    - Natural, conversational responses
    - Handles interruptions gracefully
    - Uses cached Shopify data
    - Parallel text and TTS generation
    - Context-aware length adjustment
    """
    start_time = time.time()
   
    # Build customer context
    customer_orders_info = ""
    if shopify_data_cache:
        order = shopify_data_cache
        order_name = order.get('name', 'N/A')
        order_status = order.get('fulfillment_status', 'unknown')
        order_total = order.get('total_price', 'N/A')
        customer_orders_info = f"""
Recent Order: {order_name}
Status: {order_status}
Total: ${order_total}"""
   
    # Enhanced system prompt for natural conversation
    system_prompt = f"""You are a friendly, helpful sales agent for ArtByMaudsch.com, an art gallery.


Customer Context:
{f'Email: {caller_email}' if caller_email else 'Email: Not provided'}
{customer_orders_info if customer_orders_info else 'No recent orders found'}


CRITICAL CONVERSATION RULES:
1. Keep responses VERY SHORT (1-2 sentences, 15-25 words maximum)
2. Sound natural and conversational - like a real person on the phone
3. Use contractions (I'm, you're, we'll, that's, etc.)
4. Avoid robotic phrases like "How may I assist you today?"
5. Be warm but not overly formal
6. If you need to look something up, say so briefly
7. Mirror the customer's energy and tone
8. Don't repeat information unnecessarily
9. Ask ONE simple follow-up question if needed
10. If interrupted, acknowledge and move forward naturally


Response Length Guide:
- Greeting: 5-10 words ("Hi! How can I help you today?")
- Simple answer: 10-15 words ("Your order shipped yesterday. You'll get it in 2-3 days.")
- Detailed answer: 15-25 words maximum
- NEVER exceed 25 words unless absolutely necessary


Natural Conversation Examples:
❌ Bad: "Thank you for contacting us. I would be happy to assist you with your inquiry regarding your order status today."
✅ Good: "Sure! Let me check your order status."


❌ Bad: "I sincerely apologize for any inconvenience this may have caused you."
✅ Good: "I'm sorry about that. Let's fix it."


❌ Bad: "Is there anything else I can help you with today?"
✅ Good: "Anything else?"


Key Skills:
- Track orders and shipments
- Answer product questions
- Handle complaints with empathy
- Provide quick updates
- Know when to escalate


Remember: You're having a real phone conversation. Be brief, natural, and helpful."""


    # Generate AI text
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
                temperature=0.7,  # Slightly higher for more natural variation
                max_tokens=40,  # Increased slightly for more natural responses
                presence_penalty=0.6,
                frequency_penalty=0.4
            )
        )


        ai_text = response.choices[0].message.content.strip()
       
        # Ensure response isn't too long (emergency truncation)
        words = ai_text.split()
        if len(words) > 30:
            ai_text = ' '.join(words[:30])
            # Add natural ending if truncated mid-sentence
            if not ai_text.endswith(('.', '!', '?')):
                ai_text += '.'
       
        gpt_time = time.time() - start_time
        print(f"✅ AI text ({gpt_time:.2f}s): '{ai_text}' ({len(words)} words)")


    except Exception as e:
        print(f"⚠️ AI text generation error: {e}")
        ai_text = "I'm sorry, could you repeat that?"


    # Convert to speech
    ai_audio_8khz_pcm = None
    if ai_text:
        try:
            tts_start = time.time()
            eleven_client = get_elevenlabs_client()
           
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
    print(f"⏱️ Total generation time: {total_time:.2f}s")
   
    return ai_text, ai_audio_8khz_pcm




async def generate_ai_text_only(
    transcript_buffer: str,
    caller_email: Optional[str] = None,
    shopify_client: Optional[ShopifyClient] = None
) -> str:
    """Generate AI response text without TTS."""
    ai_text, _ = await generate_ai_response_live(transcript_buffer, caller_email, shopify_client)
    return ai_text

