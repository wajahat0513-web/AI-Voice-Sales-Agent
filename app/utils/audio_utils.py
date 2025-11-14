import requests
from app.core.config import settings

def elevenlabs_tts(text: str) -> bytes:
    """
    Convert AI-generated text to speech using ElevenLabs.
    Returns audio bytes.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{settings.ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": settings.ELEVENLABS_API_KEY}
    payload = {"text": text}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.content
