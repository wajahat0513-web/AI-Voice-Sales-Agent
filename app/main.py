from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.twilio_endpoints import router as twilio_router
from app.core.logger import get_logger

app = FastAPI(
    title="AI Voice Sales Agent",
    description="PoC for AI voice agent integrated with Twilio, Shopify, and Zendesk",
    version="1.0.0"
)

logger = get_logger("main")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(twilio_router, prefix="/api/twilio")

@app.get("/")
async def root():
    """
    Root endpoint to verify API is running
    """
    logger.info("API root called.")
    return {"message": "AI Voice Sales Agent is running!"}

@app.get("/health")
async def health():
    return {"status": "ok"}
