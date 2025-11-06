# app/core/db_client.py

from typing import Optional
import os
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    PSYCOPG2_AVAILABLE = False
    print("⚠️ psycopg2 not available - database features disabled")


class DatabaseClient:
    """
    PostgreSQL database client for logging calls and transcripts.
    Falls back gracefully if database is not available.
    """
    
    def __init__(self):
        self.connection = None
        self.enabled = False
        
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            print("ℹ️  DATABASE_URL not set - database logging disabled")
            return
        
        if not PSYCOPG2_AVAILABLE:
            print("⚠️ psycopg2 not installed - database logging disabled")
            print("   Install with: pip install psycopg2-binary")
            return
        
        try:
            # Parse DATABASE_URL and create connection
            # Handle both postgresql:// and postgres:// schemes
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            
            # Connect without proxy argument (Render compatibility)
            self.connection = psycopg2.connect(
                database_url,
                cursor_factory=RealDictCursor,
                connect_timeout=10
            )
            self.connection.autocommit = True
            self.enabled = True
            
            # Test connection
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            
            print("✅ DatabaseClient initialized successfully")
            self._create_tables()
            
        except Exception as e:
            print(f"⚠️ DatabaseClient init failed: {e}")
            print("   Continuing without database logging...")
            self.connection = None
            self.enabled = False
    
    def _create_tables(self):
        """Create tables if they don't exist"""
        if not self.enabled:
            return
        
        try:
            with self.connection.cursor() as cursor:
                # Call transcripts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS call_transcripts (
                        id SERIAL PRIMARY KEY,
                        call_sid VARCHAR(255) UNIQUE NOT NULL,
                        from_number VARCHAR(50),
                        to_number VARCHAR(50),
                        transcript TEXT,
                        summary TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Call metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS call_metadata (
                        id SERIAL PRIMARY KEY,
                        call_sid VARCHAR(255) NOT NULL,
                        call_status VARCHAR(50),
                        duration INTEGER,
                        call_type VARCHAR(50),
                        tags TEXT[],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Zendesk tickets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS zendesk_tickets (
                        id SERIAL PRIMARY KEY,
                        call_sid VARCHAR(255) UNIQUE NOT NULL,
                        zendesk_ticket_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
            print("✅ Database tables created/verified")
            
        except Exception as e:
            print(f"⚠️ Table creation error: {e}")
    
    def insert_call_transcript(
        self,
        call_sid: str,
        from_number: str,
        to_number: str,
        transcript: str,
        summary: str
    ):
        """Insert call transcript"""
        if not self.enabled:
            return
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO call_transcripts 
                    (call_sid, from_number, to_number, transcript, summary)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (call_sid) DO UPDATE SET
                        transcript = EXCLUDED.transcript,
                        summary = EXCLUDED.summary
                """, (call_sid, from_number, to_number, transcript, summary))
            
        except Exception as e:
            print(f"⚠️ Insert transcript error: {e}")
    
    def insert_call_metadata(
        self,
        call_sid: str,
        call_status: str,
        duration: int,
        call_type: str,
        tags: list
    ):
        """Insert call metadata"""
        if not self.enabled:
            return
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO call_metadata 
                    (call_sid, call_status, duration, call_type, tags)
                    VALUES (%s, %s, %s, %s, %s)
                """, (call_sid, call_status, duration, call_type, tags))
            
        except Exception as e:
            print(f"⚠️ Insert metadata error: {e}")
    
    def insert_zendesk_ticket(
        self,
        call_sid: str,
        zendesk_ticket_id: str
    ):
        """Insert Zendesk ticket reference"""
        if not self.enabled:
            return
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO zendesk_tickets 
                    (call_sid, zendesk_ticket_id)
                    VALUES (%s, %s)
                    ON CONFLICT (call_sid) DO NOTHING
                """, (call_sid, zendesk_ticket_id))
            
        except Exception as e:
            print(f"⚠️ Insert Zendesk ticket error: {e}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")

