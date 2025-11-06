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
            timeout=30.0,  # Reduced timeout
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
        """
        if not text.strip():
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
                "model_id": "eleven_turbo_v2_5",  # Fastest available model
                "voice_settings": {
                    "stability": 0.5,  # Lower = faster processing
                    "similarity_boost": 0.75,
                    "style": 0.0,  # Disable style for speed
                    "use_speaker_boost": False  # Disable for speed
                },
                "optimize_streaming_latency": 4  # Maximum optimization
            }




            # Make request
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()




            mp3_bytes = response.content




            if len(mp3_bytes) < 100:
                return None




            # Convert to PCM
            pcm_audio = await self._convert_to_pcm_fast(mp3_bytes)
            return pcm_audio




        except httpx.HTTPStatusError as e:
            print(f"❌ ElevenLabs error: {e.response.status_code}")
            return None
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return None




    async def text_to_speech(self, text: str) -> Optional[bytes]:
        """Alias for compatibility"""
        return await self.text_to_speech_fast(text)




    async def _convert_to_pcm_fast(self, mp3_bytes: bytes) -> Optional[bytes]:
        """
        OPTIMIZED: Fast MP3 to PCM conversion
        """
        if not PYDUB_AVAILABLE:
            print("❌ pydub unavailable")
            return None




        try:
            # Load and convert in one go
            audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
           
            # Apply all conversions at once
            audio = (audio
                     .set_frame_rate(8000)
                     .set_channels(1)
                     .set_sample_width(2))
           
            # Quick normalization
            audio = audio.apply_gain(-audio.max_dBFS + (-3.0))
           
            pcm_data = audio.raw_data




            if len(pcm_data) < 1000:
                return None




            return pcm_data




        except Exception as e:
            print(f"❌ Conversion error: {e}")
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





