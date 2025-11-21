# app/core/elevenlabs_client.py

import httpx
import io
from app.core.config import settings
from typing import Optional
import subprocess
import os

try:
    from pydub import AudioSegment
    try:
        result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip().split('\n')[0]
            AudioSegment.converter = ffmpeg_path
            print(f"✅ Found ffmpeg at: {ffmpeg_path}")
    except:
        pass
    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False
    print("❌ pydub not available")

class ElevenLabsClient:
    def __init__(self):
        self.api_key = settings.ELEVENLABS_API_KEY
        self.voice_id = settings.ELEVENLABS_VOICE_ID
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # CRITICAL: Use persistent connection with connection pooling
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            )
        )

        print(f"✅ ElevenLabsClient initialized (Voice: {self.voice_id[:8]}...)")

    async def text_to_speech_fast(self, text: str) -> Optional[bytes]:
        """
        OPTIMIZED: Convert text to speech with minimal latency
        - Uses eleven_turbo_v2_5 (fastest model)
        - Optimized voice settings for speed
        - Persistent HTTP connection
        - Better error handling
        """
        if not text.strip():
            return None

        try:
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            # Log voice_id to ensure consistency (first 8 chars only for security)
            if not hasattr(self, '_voice_logged'):
                print(f"🔊 Using ElevenLabs voice ID: {self.voice_id[:8]}... (consistent throughout call)")
                self._voice_logged = True
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }

            # CONSISTENT voice settings for stable voice throughout call
            voice_settings = {
                "stability": 0.7,  # Higher = more consistent voice (was 0.5)
                "similarity_boost": 0.75,
                "style": 0.0,  # Disable style for consistency
                "use_speaker_boost": True  # Enable for better voice quality
            }
            
            payload = {
                "text": text,
                "model_id": "eleven_turbo_v2_5",  # Fastest available model
                "voice_settings": voice_settings,
                "optimize_streaming_latency": 4  # Maximum optimization
            }

            # Log settings on first call to verify consistency
            if not hasattr(self, '_settings_logged'):
                print(f"🔊 ElevenLabs settings: voice_id={self.voice_id[:8]}..., model=eleven_turbo_v2_5, stability={voice_settings['stability']}, similarity={voice_settings['similarity_boost']}")
                self._settings_logged = True

            # Make request
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            mp3_bytes = response.content

            if len(mp3_bytes) < 100:
                print("⚠️ TTS returned empty response")
                return None

            # Convert to PCM
            pcm_audio = await self._convert_to_pcm_fast(mp3_bytes)
            return pcm_audio

        except httpx.HTTPStatusError as e:
            print(f"❌ ElevenLabs HTTP error: {e.response.status_code}")
            if e.response.status_code == 401:
                print("❌ Invalid API key")
            elif e.response.status_code == 429:
                print("❌ Rate limit exceeded")
            return None
        except httpx.TimeoutException:
            print(f"❌ ElevenLabs timeout")
            return None
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None

    async def text_to_speech(self, text: str) -> Optional[bytes]:
        """Alias for compatibility"""
        return await self.text_to_speech_fast(text)

    async def _convert_to_pcm_fast(self, mp3_bytes: bytes) -> Optional[bytes]:
        """
        OPTIMIZED: Fast MP3 to PCM conversion with better error handling
        """
        if not PYDUB_AVAILABLE:
            print("❌ pydub unavailable - cannot convert audio")
            return None

        try:
            # Load MP3
            audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            
            # Apply all conversions at once for speed
            audio = (audio
                     .set_frame_rate(8000)  # 8kHz for Twilio
                     .set_channels(1)        # Mono
                     .set_sample_width(2))   # 16-bit
            
            # Quick normalization to prevent clipping
            # Target -3dB to leave headroom
            current_db = audio.max_dBFS
            if current_db > -3.0:
                gain_adjustment = -3.0 - current_db
                audio = audio.apply_gain(gain_adjustment)
            
            pcm_data = audio.raw_data

            if len(pcm_data) < 1000:
                print("⚠️ Converted audio too short")
                return None

            return pcm_data

        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return None

    async def _convert_to_pcm(self, mp3_bytes: bytes) -> Optional[bytes]:
        """Alias for compatibility"""
        return await self._convert_to_pcm_fast(mp3_bytes)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.client.aclose())
        except:
            pass