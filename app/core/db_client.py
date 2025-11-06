# app/core/db_client.py


from supabase import create_client
from app.core.config import settings
from datetime import datetime
from typing import List, Dict, Optional


class DatabaseClient:
    def __init__(self):
        self.supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


    # -------------------------
    # Call Transcripts
    # -------------------------
    def insert_call_transcript(
        self, call_sid: str, transcript: str, summary: str, from_number: str = "unknown", to_number: str = "unknown"
    ):
        """Insert a transcript and summary for a call."""
        self.supabase.table("call_transcripts").insert({
            "call_sid": call_sid,
            "from_number": from_number,
            "to_number": to_number,
            "transcript": transcript,
            "summary": summary,
            "created_at": datetime.utcnow().isoformat()
        }).execute()


    # -------------------------
    # Call Metadata
    # -------------------------
    def insert_call_metadata(
        self, call_sid: str, call_status: str, duration: int, call_type: str, tags: Optional[List[str]] = None
    ):
        self.supabase.table("call_metadata").insert({
            "call_sid": call_sid,
            "call_status": call_status,
            "duration": duration,
            "call_type": call_type,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat()
        }).execute()


    # -------------------------
    # Zendesk Ticket Mapping
    # -------------------------
    def insert_zendesk_ticket(self, call_sid: str, zendesk_ticket_id: int):
        self.supabase.table("zendesk_tickets").insert({
            "call_sid": call_sid,
            "zendesk_ticket_id": zendesk_ticket_id,
            "created_at": datetime.utcnow().isoformat()
        }).execute()


    # -------------------------
    # Shopify Call Tags
    # -------------------------
    def insert_shopify_call_tag(self, call_sid: str, order_id: str, tags: Optional[List[str]] = None):
        self.supabase.table("shopify_call_tags").insert({
            "call_sid": call_sid,
            "order_id": order_id,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat()
        }).execute()


    def insert_shopify_tags_bulk(self, call_sid: str, order_tags: List[Dict]):
        records = []
        for ot in order_tags:
            records.append({
                "call_sid": call_sid,
                "order_id": ot["order_id"],
                "tags": ot.get("tags", []),
                "created_at": datetime.utcnow().isoformat()
            })
        if records:
            self.supabase.table("shopify_call_tags").insert(records).execute()



