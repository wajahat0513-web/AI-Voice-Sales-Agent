from dotenv import load_dotenv
import os

load_dotenv()

class Settings:

    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID") 

    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

    ZENDESK_SUBDOMAIN: str = os.getenv("ZENDESK_SUBDOMAIN")
    ZENDESK_EMAIL: str = os.getenv("ZENDESK_EMAIL")
    ZENDESK_API_TOKEN: str = os.getenv("ZENDESK_API_TOKEN")

    SHOPIFY_STORE_DOMAIN: str = os.getenv("SHOPIFY_DOMAIN")
    SHOPIFY_API_KEY: str = os.getenv("SHOPIFY_API_KEY")
    SHOPIFY_PASSWORD: str = os.getenv("SHOPIFY_PASSWORD") 

settings = Settings()
