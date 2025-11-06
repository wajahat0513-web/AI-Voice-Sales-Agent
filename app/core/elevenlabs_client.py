# app/core/elevenlabs_client.py

import httpx
import io
from app.core.config import settings
from typing import Optional
import subprocess

try:
    from pydub import AudioSegment
    try:
        # Try to find ffmpeg
        result = subprocess.run(
            ['where', 'ffmpeg'] if subprocess.os.name == 'nt' else ['which', 'ffmpeg'],
            capture_output=True,
            text=True,
            shell=False
        )
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip().split('\n')[0]
            AudioSegment.converter = ffmpeg_path
            print(f"✅ Found ffmpeg at: {ffmpeg_path}")
        else:
            print("⚠️ ffmpeg not found in PATH")
    except Exception as e:
        print(f"⚠️ Error locating ffmpeg: {e}")
    PYDUB_AVAILABLE = True
except ImportError:
    AudioSegment = None
    PYDUB_AVAILABLE = False
    print("❌ pydub not available - install with: pip install pydub")

class ElevenLabsClient:
    """
    ElevenLabs TTS client optimized for real-time phone calls.
    Uses persistent HTTP connections and the fastest available model.
    """
    
    def __init__(self):
        self.api_key = settings.ELEVENLABS_API_KEY
        self.voice_id = settings.ELEVENLABS_VOICE_ID
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # CRITICAL: Persistent connection with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            ),
            http2=True  # Use HTTP/2 for better performance
        )

        print(f"✅ ElevenLabsClient initialized")
        print(f"   Voice ID: {self.voice_id[:8]}...")
        print(f"   HTTP/2: Enabled")

    async def text_to_speech_fast(self, text: str) -> Optional[bytes]:
        """
        OPTIMIZED: Convert text to speech with minimal latency.
        
        Returns 8kHz 16-bit PCM audio ready for Twilio.
        
        Optimizations:
        - eleven_turbo_v2_5 (fastest model)
        - Minimized voice settings
        - Persistent HTTP connection
        - Maximum streaming latency optimization
        """
        if not text or not text.strip():
            return None

        try:
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }

            # SPEED-OPTIMIZED settings
            payload = {
                "text": text,
                "model_id": "eleven_turbo_v2_5",  # Fastest model
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,  # Disable for speed
                    "use_speaker_boost": False  # Disable for speed
                },
                "optimize_streaming_latency": 4  # Max optimization
            }

            # Make request
            response = await self.client.post(
                url, 
                json=payload, 
                headers=headers
            )
            response.raise_for_status()

            mp3_bytes = response.content

            if len(mp3_bytes) < 100:
                print("⚠️ ElevenLabs returned very short audio")
                return None

            # Convert MP3 to 8kHz PCM
            pcm_audio = await self._convert_to_pcm_fast(mp3_bytes)
            return pcm_audio

        except httpx.HTTPStatusError as e:
            print(f"❌ ElevenLabs HTTP error: {e.response.status_code}")
            try:
                error_body = e.response.json()
                print(f"   Error details: {error_body}")
            except:
                pass
            return None
        except httpx.TimeoutException:
            print(f"❌ ElevenLabs timeout error")
            return None
        except Exception as e:
            print(f"❌ ElevenLabs TTS error: {e}")
            return None

    async def text_to_speech(self, text: str) -> Optional[bytes]:
        """Alias for compatibility"""
        return await self.text_to_speech_fast(text)

    async def _convert_to_pcm_fast(self, mp3_bytes: bytes) -> Optional[bytes]:
        """
        OPTIMIZED: Fast MP3 to 8kHz 16-bit PCM conversion.
        
        Returns audio ready for Twilio (8kHz, mono, 16-bit).
        """
        if not PYDUB_AVAILABLE:
            print("❌ pydub unavailable - cannot convert audio")
            return None

        try:
            # Load MP3
            audio = AudioSegment.from_file(
                io.BytesIO(mp3_bytes), 
                format="mp3"
            )
            
            # Apply all conversions at once for speed
            audio = (audio
                     .set_frame_rate(8000)  # Twilio uses 8kHz
                     .set_channels(1)       # Mono
                     .set_sample_width(2))  # 16-bit
            
            # Normalize volume to prevent clipping
            # Target -3 dBFS for safety margin
            current_db = audio.max_dBFS
            if current_db != float('-inf'):
                target_db = -3.0
                gain_needed = target_db - current_db
                audio = audio.apply_gain(gain_needed)
            
            # Extract raw PCM data
            pcm_data = audio.raw_data

            if len(pcm_data) < 1000:
                print("⚠️ Converted audio too short")
                return None

            return pcm_data

        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _convert_to_pcm(self, mp3_bytes: bytes) -> Optional[bytes]:
        """Alias for compatibility"""
        return await self._convert_to_pcm_fast(mp3_bytes)

    async def close(self):
        """Close the HTTP client gracefully"""
        await self.client.aclose()
        print("✅ ElevenLabsClient closed")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.client.aclose())
        except:
            pass

