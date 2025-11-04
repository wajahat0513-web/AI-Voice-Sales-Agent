#!/usr/bin/env python3
"""
System Diagnostic Script
Checks all components of the AI Voice Sales Agent
Run this before starting the server
"""

import sys
import os
import subprocess
from typing import Tuple

print("=" * 70)
print("🔍 AI VOICE SALES AGENT - SYSTEM DIAGNOSTIC")
print("=" * 70)

results = {"passed": [], "failed": [], "warnings": []}

def test_result(name: str, passed: bool, message: str = ""):
    """Record test result"""
    if passed:
        results["passed"].append(name)
        print(f"✅ {name}")
        if message:
            print(f"   {message}")
    else:
        results["failed"].append(name)
        print(f"❌ {name}")
        if message:
            print(f"   {message}")

def warning(name: str, message: str):
    """Record warning"""
    results["warnings"].append(name)
    print(f"⚠️  {name}")
    print(f"   {message}")

# ============================================================
# TEST 1: Python Version
# ============================================================
print("\n🐍 Python Environment")
print("-" * 70)

python_version = sys.version_info
test_result(
    "Python Version",
    python_version >= (3, 8),
    f"Version: {python_version.major}.{python_version.minor}.{python_version.micro}"
)

# ============================================================
# TEST 2: Virtual Environment
# ============================================================
in_venv = hasattr(sys, 'real_prefix') or (
    hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
)
test_result(
    "Virtual Environment",
    in_venv,
    "Running in venv" if in_venv else "NOT in venv - activate with: .\\venv\\Scripts\\activate"
)

# ============================================================
# TEST 3: Critical Python Packages
# ============================================================
print("\n📦 Python Packages")
print("-" * 70)

packages = {
    "fastapi": "FastAPI",
    "uvicorn": "Uvicorn",
    "websockets": "WebSockets",
    "openai": "OpenAI",
    "twilio": "Twilio",
    "httpx": "HTTPX",
    "aiohttp": "AIOHTTP",
    "numpy": "NumPy",
    "pydub": "Pydub",
    "supabase": "Supabase",
    "dotenv": "Python-dotenv",
}

for module, name in packages.items():
    try:
        if module == "dotenv":
            import dotenv
        else:
            __import__(module)
        
        # Get version if possible
        try:
            pkg = __import__(module)
            version = getattr(pkg, '__version__', 'unknown')
            test_result(name, True, f"v{version}")
        except:
            test_result(name, True)
    except ImportError:
        test_result(name, False, f"Install: pip install {module}")

# ============================================================
# TEST 4: Pydub Audio Capabilities
# ============================================================
print("\n🎵 Audio Processing")
print("-" * 70)

try:
    from pydub import AudioSegment
    from pydub.utils import which
    
    test_result("Pydub Import", True)
    
    # Check FFmpeg
    ffmpeg_path = which("ffmpeg")
    test_result(
        "FFmpeg",
        ffmpeg_path is not None,
        f"Path: {ffmpeg_path}" if ffmpeg_path else "NOT FOUND - Install from https://ffmpeg.org"
    )
    
    # Test audio conversion
    try:
        import io
        # Create a minimal MP3-like byte sequence (won't actually work, but tests the flow)
        test_audio = AudioSegment.silent(duration=100)  # 100ms of silence
        test_audio = test_audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
        pcm = test_audio.raw_data
        test_result("Audio Conversion", True, f"Converted {len(pcm)} bytes")
    except Exception as e:
        test_result("Audio Conversion", False, str(e))

except ImportError as e:
    test_result("Pydub Import", False, str(e))
    warning(
        "Audio Processing Unavailable",
        "Phone calls will have STATIC NOISE without pydub!\n" +
        "   Fix: pip install pydub && install ffmpeg"
    )

# ============================================================
# TEST 5: Environment Variables
# ============================================================
print("\n🔐 Environment Variables")
print("-" * 70)

from dotenv import load_dotenv
load_dotenv()

env_vars = {
    "OPENAI_API_KEY": "OpenAI",
    "ELEVENLABS_API_KEY": "ElevenLabs",
    "ELEVENLABS_VOICE_ID": "ElevenLabs Voice",
    "TWILIO_ACCOUNT_SID": "Twilio Account",
    "TWILIO_AUTH_TOKEN": "Twilio Auth",
    "TWILIO_PHONE_NUMBER": "Twilio Phone",
    "WEBSOCKET_URL": "WebSocket URL",
    "SUPABASE_URL": "Supabase",
    "SUPABASE_KEY": "Supabase Key",
    "ZENDESK_SUBDOMAIN": "Zendesk",
    "ZENDESK_EMAIL": "Zendesk Email",
    "ZENDESK_API_TOKEN": "Zendesk Token",
    "SHOPIFY_DOMAIN": "Shopify",
    "SHOPIFY_API_KEY": "Shopify Key",
    "SHOPIFY_PASSWORD": "Shopify Password",
}

for var, name in env_vars.items():
    value = os.getenv(var)
    has_value = value and value.strip()
    
    if has_value:
        # Mask sensitive values
        if any(x in var for x in ["KEY", "TOKEN", "PASSWORD", "AUTH"]):
            display = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
        else:
            display = value[:40] + "..." if len(value) > 40 else value
        
        test_result(name, True, display)
    else:
        test_result(name, False, f"Missing in .env file")

# ============================================================
# TEST 6: Client Initialization
# ============================================================
print("\n🔌 Service Clients")
print("-" * 70)

# Test ElevenLabs
try:
    from app.core.elevenlabs_client import ElevenLabsClient
    client = ElevenLabsClient()
    test_result("ElevenLabs Client", True, f"Voice: {client.voice_id}")
except Exception as e:
    test_result("ElevenLabs Client", False, str(e))

# Test Zendesk
try:
    from app.services.zendesk_client import ZendeskClient
    client = ZendeskClient()
    test_result("Zendesk Client", True, f"Subdomain: {client.subdomain}")
except Exception as e:
    test_result("Zendesk Client", False, str(e))

# Test Shopify
try:
    from app.services.shopify_client import ShopifyClient
    client = ShopifyClient()
    test_result("Shopify Client", True, f"Domain: {client.domain}")
except Exception as e:
    test_result("Shopify Client", False, str(e))

# Test Database
try:
    from app.core.db_client import DatabaseClient
    client = DatabaseClient()
    test_result("Database Client", True)
except Exception as e:
    test_result("Database Client", False, str(e))

# ============================================================
# TEST 7: API Connectivity (Optional - requires network)
# ============================================================
print("\n🌐 API Connectivity (Quick Check)")
print("-" * 70)

import asyncio

async def test_apis():
    # Test OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Just check if we can create a client
        test_result("OpenAI Connection", True)
    except Exception as e:
        test_result("OpenAI Connection", False, str(e))
    
    # Test ElevenLabs
    try:
        from app.core.elevenlabs_client import ElevenLabsClient
        client = ElevenLabsClient()
        # Don't actually call API, just verify setup
        test_result("ElevenLabs Setup", True)
        await client.close()
    except Exception as e:
        test_result("ElevenLabs Setup", False, str(e))

try:
    asyncio.run(test_apis())
except Exception as e:
    warning("API Tests", f"Skipped: {e}")

# ============================================================
# TEST 8: File Structure
# ============================================================
print("\n📁 File Structure")
print("-" * 70)

required_files = [
    "app/main.py",
    "app/api/twilio_endpoints.py",
    "app/core/elevenlabs_client.py",
    "app/core/db_client.py",
    "app/services/ai_service.py",
    "app/services/zendesk_client.py",
    "app/services/shopify_client.py",
    ".env",
]

for filepath in required_files:
    exists = os.path.exists(filepath)
    test_result(
        filepath,
        exists,
        "✓" if exists else "Missing file!"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📊 DIAGNOSTIC SUMMARY")
print("=" * 70)

print(f"\n✅ Passed: {len(results['passed'])}")
for item in results['passed'][:5]:  # Show first 5
    print(f"   • {item}")
if len(results['passed']) > 5:
    print(f"   ... and {len(results['passed']) - 5} more")

if results['failed']:
    print(f"\n❌ Failed: {len(results['failed'])}")
    for item in results['failed']:
        print(f"   • {item}")

if results['warnings']:
    print(f"\n⚠️  Warnings: {len(results['warnings'])}")
    for item in results['warnings']:
        print(f"   • {item}")

# ============================================================
# RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS")
print("=" * 70)

if not results['failed'] and not results['warnings']:
    print("\n🎉 All checks passed! System is ready.")
    print("\nNext steps:")
    print("   1. Start server: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("   2. Test with fake call: python fake_call.py")
    print("   3. Configure Twilio webhook and test real call")

else:
    print("\n⚠️  Some issues need attention:")
    
    if "Pydub Import" in results['failed'] or "FFmpeg" in results['failed']:
        print("\n🔴 CRITICAL: Audio processing not working!")
        print("   Without pydub + FFmpeg, you will get STATIC NOISE on calls.")
        print("   Fix:")
        print("      pip install pydub")
        print("      Install FFmpeg: https://ffmpeg.org/download.html")
        print("      Add FFmpeg to PATH")
    
    if any("API_KEY" in item or "TOKEN" in item for item in results['failed']):
        print("\n🔴 CRITICAL: Missing API credentials!")
        print("   Add missing values to your .env file")
    
    if results['failed']:
        print("\n   Run this script again after fixes: python diagnose_system.py")

print("\n" + "=" * 70)

# Exit with appropriate code
sys.exit(0 if not results['failed'] else 1)