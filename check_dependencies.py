# check_dependencies.py
"""
Quick script to verify all required dependencies are installed
Run this before executing test_flow.py
"""

import sys
from typing import List, Tuple

REQUIRED_PACKAGES = {
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'websockets': 'WebSockets',
    'openai': 'OpenAI',
    'twilio': 'Twilio',
    'aiohttp': 'AIOHTTP',
    'supabase': 'Supabase',
    'shopify': 'Shopify (shopifyapi)',
    'python-dotenv': 'Python-dotenv',
    'pydantic': 'Pydantic',
    'requests': 'Requests',
}

def check_imports() -> Tuple[List[str], List[str]]:
    """Check which packages are installed"""
    installed = []
    missing = []
    
    for package, display_name in REQUIRED_PACKAGES.items():
        try:
            # Handle special cases
            if package == 'python-dotenv':
                import dotenv
            elif package == 'shopify':
                import shopify
            else:
                __import__(package)
            installed.append(display_name)
        except ImportError:
            missing.append(f"{display_name} ({package})")
    
    return installed, missing

def check_env_vars() -> Tuple[List[str], List[str]]:
    """Check which environment variables are set"""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    REQUIRED_ENV = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'ELEVENLABS_API_KEY': 'ElevenLabs API Key',
        'ELEVENLABS_VOICE_ID': 'ElevenLabs Voice ID',
        'TWILIO_ACCOUNT_SID': 'Twilio Account SID',
        'TWILIO_AUTH_TOKEN': 'Twilio Auth Token',
        'TWILIO_PHONE_NUMBER': 'Twilio Phone Number',
        'SUPABASE_URL': 'Supabase URL',
        'SUPABASE_KEY': 'Supabase Key',
        'ZENDESK_SUBDOMAIN': 'Zendesk Subdomain',
        'ZENDESK_EMAIL': 'Zendesk Email',
        'ZENDESK_API_TOKEN': 'Zendesk API Token',
        'SHOPIFY_DOMAIN': 'Shopify Domain',
        'SHOPIFY_API_KEY': 'Shopify API Key',
        'SHOPIFY_PASSWORD': 'Shopify Password',
    }
    
    found = []
    missing = []
    
    for env_var, display_name in REQUIRED_ENV.items():
        value = os.getenv(env_var)
        if value and value.strip():
            # Mask sensitive values
            if 'KEY' in env_var or 'TOKEN' in env_var or 'PASSWORD' in env_var:
                masked = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
                found.append(f"{display_name}: {masked}")
            else:
                found.append(f"{display_name}: {value[:30]}...")
        else:
            missing.append(display_name)
    
    return found, missing

def main():
    print("=" * 70)
    print("🔍 AI VOICE SALES AGENT - DEPENDENCY CHECKER")
    print("=" * 70)
    
    # Check Python version
    print("\n🐍 Python Version:")
    print(f"   {sys.version}")
    if sys.version_info < (3, 8):
        print("   ⚠️  WARNING: Python 3.8+ is recommended")
    else:
        print("   ✅ Version OK")
    
    # Check packages
    print("\n📦 Checking Python Packages...")
    print("-" * 70)
    installed, missing = check_imports()
    
    if installed:
        print("\n✅ Installed packages:")
        for pkg in installed:
            print(f"   ✓ {pkg}")
    
    if missing:
        print("\n❌ Missing packages:")
        for pkg in missing:
            print(f"   ✗ {pkg}")
        print("\n💡 Install missing packages with:")
        print("   pip install " + " ".join([p.split('(')[1].rstrip(')') for p in missing]))
    else:
        print("\n✅ All required packages are installed!")
    
    # Check environment variables
    print("\n🔐 Checking Environment Variables...")
    print("-" * 70)
    found, missing_env = check_env_vars()
    
    if found:
        print("\n✅ Found environment variables:")
        for var in found:
            print(f"   ✓ {var}")
    
    if missing_env:
        print("\n❌ Missing environment variables:")
        for var in missing_env:
            print(f"   ✗ {var}")
        print("\n💡 Add these to your .env file")
    else:
        print("\n✅ All required environment variables are set!")
    
    # Final summary
    print("\n" + "=" * 70)
    if not missing and not missing_env:
        print("✅ ALL CHECKS PASSED - Ready to run test_flow.py!")
        print("=" * 70)
        print("\n🚀 Next steps:")
        print("   1. Ensure test_audio.wav exists (or script will create it)")
        print("   2. Run: python test_flow.py")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Please fix the issues above")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())