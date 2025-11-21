# app/services/ai_service.py

import io
import asyncio
import re
from typing import Tuple, Optional, List, Dict
from app.core.config import settings
from app.services.shopify_client import ShopifyClient
from app.core.elevenlabs_client import ElevenLabsClient
from openai import OpenAI
import time

# Initialize clients ONCE at module level
client = OpenAI(api_key=settings.OPENAI_API_KEY)
# Keep one ElevenLabs client alive for the entire session
_elevenlabs_client = None

def get_elevenlabs_client():
    """Get or create a persistent ElevenLabs client"""
    global _elevenlabs_client
    if _elevenlabs_client is None:
        _elevenlabs_client = ElevenLabsClient()
    return _elevenlabs_client

async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe caller audio using OpenAI Whisper.
    Expects WAV format audio (8kHz, 16-bit PCM).
    IMPROVED: Better error handling to avoid "Thank you" transcription errors.
    """
    try:
        if len(audio_bytes) < 2000:
            return ""

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "caller.wav"

        loop = asyncio.get_event_loop()
        transcript_resp = await loop.run_in_executor(
            None,
            lambda: client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="text",
                temperature=0.0,
                # IMPROVED: Better prompt to reduce hallucination
                prompt="Customer calling art gallery about orders, products, or inquiries."
            )
        )

        # Handle different response formats (string or object with .text)
        text = transcript_resp.strip() if isinstance(transcript_resp, str) else transcript_resp.text.strip()
        
        # CRITICAL: Additional validation to prevent spurious transcriptions
        if not text or len(text) < 3:
            return ""
        
        # Filter out common Whisper hallucinations
        low_text = text.lower()
        hallucinations = [
            "thank you", "thank you.", "thanks", "thanks.",
            "you", "uh", "um", "hmm", "mhmm", "okay"
        ]
        
        if low_text in hallucinations or (len(low_text.split()) == 1 and low_text in hallucinations):
            print(f"⚠️ Filtered Whisper hallucination: '{text}'")
            return ""
        
        return text

    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return ""

async def summarize_conversation(transcript: str) -> str:
    """Summarize conversation for reporting."""
    try:
        if not transcript or not transcript.strip():
            return "No conversation to summarize."

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this call briefly in under 100 words."
                    },
                    {"role": "user", "content": transcript}
                ],
                temperature=0.3,
                max_tokens=150
            )
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ Summarization error: {e}")
        return "Summary unavailable."

def extract_caller_info(text: str, caller_memory: dict) -> None:
    """
    NEW: Extract and store important information from caller's message
    Updates caller_memory dict in-place
    """
    text_lower = text.lower()
    
    # Extract order numbers (patterns: "order ABM12345X", "ABM12345X", "#ABM12345X")
    order_patterns = [
        r'\b([A-Z]{3}\d{5}[A-Z])\b',  # ABM27255X format
        r'order\s*#?\s*([A-Z]{3}\d{5}[A-Z])',
        r'#([A-Z]{3}\d{5}[A-Z])'
    ]
    
    for pattern in order_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            match_upper = match.upper()
            if match_upper not in caller_memory["order_numbers"]:
                caller_memory["order_numbers"].append(match_upper)
                print(f"🧠 Stored order number: {match_upper}")
    
    # Extract art interests/preferences
    art_keywords = [
        'landscape', 'portrait', 'abstract', 'modern', 'contemporary',
        'painting', 'watercolor', 'oil', 'acrylic', 'canvas',
        'nature', 'ocean', 'mountain', 'cityscape', 'floral',
        'blue', 'red', 'colorful', 'monochrome', 'bright', 'dark'
    ]
    
    for keyword in art_keywords:
        if keyword in text_lower and keyword not in caller_memory["art_interests"]:
            caller_memory["art_interests"].append(keyword)
            print(f"🎨 Stored art interest: {keyword}")
    
    # Extract issues/problems
    issue_keywords = [
        'broken', 'damaged', 'wrong', 'missing', 'late', 'delayed',
        'problem', 'issue', 'complaint', 'return'
    ]
    
    for keyword in issue_keywords:
        if keyword in text_lower and keyword not in caller_memory["issues"]:
            caller_memory["issues"].append(keyword)
            print(f"⚠️ Stored issue: {keyword}")
    
    # Extract name if introduced
    name_patterns = [
        r"(?:my name is|i'm|this is|call me)\s+([A-Z][a-z]+)",
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and not caller_memory["name"]:
            name = match.group(1)
            # Basic filter for common non-names
            if name.lower() not in ['sir', 'maam', 'ma\'am', 'yes', 'no', 'okay']:
                caller_memory["name"] = name.capitalize()
                print(f"👤 Stored caller name: {caller_memory['name']}")
                break
    
    # NEW: Extract email if provided
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match and not caller_memory.get("email"):
        caller_memory["email"] = email_match.group(0)
        print(f"📧 Stored email: {caller_memory['email']}")

def detect_action_request(text: str) -> Optional[str]:
    """
    NEW: Detect if caller is requesting an action that needs confirmation
    Returns action type if detected, None otherwise
    NOTE: cancel_order removed as per requirements
    """
    text_lower = text.lower()
    
    action_patterns = {
        'refund': [
            r'refund',
            r'money back',
            r'get.*back.*money'
        ],
        'return': [
            r'return.*order',
            r'send.*back',
            r'return.*item'
        ],
        'change_address': [
            r'change.*address',
            r'update.*address',
            r'wrong address'
        ]
    }
    
    for action_type, patterns in action_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return action_type
    
    return None

def build_memory_context(caller_memory: dict) -> str:
    """
    NEW: Build context string from caller memory with customer and order information
    """
    context_parts = []
    
    # Customer information
    if caller_memory.get("customer_info"):
        customer = caller_memory["customer_info"]
        name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
        if name:
            context_parts.append(f"Customer Name: {name}")
        if customer.get("email"):
            context_parts.append(f"Email: {customer.get('email')}")
        context_parts.append(f"Customer ID: {caller_memory.get('customer_id')}")
        
        # Add orders count
        orders_count = len(caller_memory.get("all_orders", []))
        if orders_count > 0:
            context_parts.append(f"Total Orders: {orders_count}")
    elif caller_memory.get("name"):
        context_parts.append(f"Caller's name: {caller_memory['name']}")
    
    # Order information
    if caller_memory.get("all_orders"):
        orders = caller_memory["all_orders"]
        order_summary = []
        for order in orders[:5]:  # Limit to 5 most recent
            order_name = order.get("name", "").replace("#", "")
            status = order.get("fulfillment_status", "unfulfilled")
            order_summary.append(f"{order_name} ({status})")
        
        if order_summary:
            context_parts.append(f"Customer Orders: {', '.join(order_summary)}")
    
    if caller_memory.get("verified_orders"):
        verified_list = ', '.join(caller_memory['verified_orders'].keys())
        context_parts.append(f"Currently Discussing Orders: {verified_list}")
    
    if caller_memory.get("art_interests"):
        context_parts.append(f"Art interests: {', '.join(caller_memory['art_interests'][:5])}")  # Limit to 5
    
    if caller_memory.get("issues"):
        context_parts.append(f"Mentioned issues: {', '.join(caller_memory['issues'][:3])}")  # Limit to 3
    
    if caller_memory.get("pending_actions"):
        context_parts.append(f"Pending actions: {', '.join(caller_memory['pending_actions'])}")
    
    if context_parts:
        return "CALLER MEMORY:\n" + "\n".join(context_parts) + "\n"
    
    return ""

def build_order_context(order_context: Optional[Dict]) -> str:
    """
    NEW: Build detailed order context string
    """
    if not order_context:
        return ""
    
    context = f"\nCURRENT ORDER DETAILS:\n"
    context += f"Order Number: {order_context.get('order_number', 'N/A')}\n"
    context += f"Total: ${order_context.get('total', 'N/A')} {order_context.get('currency', 'USD')}\n"
    context += f"Payment Status: {order_context.get('status', 'unknown')}\n"
    context += f"Fulfillment Status: {order_context.get('fulfillment_status', 'unknown')}\n"
    
    items = order_context.get('items', [])
    if items:
        context += f"Items:\n"
        for item in items[:3]:  # Limit to 3 items
            context += f"  - {item.get('title')} (Qty: {item.get('quantity')}, ${item.get('price')})\n"
    
    tracking = order_context.get('tracking_numbers', [])
    if tracking:
        context += f"Tracking: {', '.join(tracking)}\n"
    
    shipping = order_context.get('shipping_address')
    if shipping:
        context += f"Shipping to: {shipping.get('city', '')}, {shipping.get('province', '')}\n"
    
    return context


def build_catalog_highlights(catalog_context: Optional[Dict]) -> str:
    """
    Build a concise string describing featured collections and artworks.
    """
    if not catalog_context:
        return ""

    sections = []

    custom = catalog_context.get("custom_collections") or []
    if custom:
        lines = []
        for col in custom[:4]:
            summary = col.get("body") or "Handmade selection"
            title = col.get("title", "Collection")
            lines.append(f"- {title}: {summary}")
        sections.append("FEATURED COLLECTIONS:\n" + "\n".join(lines))

    products = catalog_context.get("products") or []
    if products:
        lines = []
        for product in products[:5]:
            title = product.get("title", "Artwork")
            art_type = product.get("type") or "Art"
            price = product.get("price_range")
            tags = product.get("tags")
            desc = product.get("description")

            parts = [f"{title} ({art_type})"]
            extras = []
            if price:
                extras.append(price)
            if tags:
                tag_preview = tags if len(tags) <= 50 else tags[:50] + "..."
                extras.append(tag_preview)
            if extras:
                parts.append(" • ".join(extras))
            if desc:
                snippet = desc if len(desc) <= 100 else desc[:100] + "..."
                parts.append(snippet)

            lines.append("- " + " :: ".join(parts))
        sections.append("HIGHLIGHTED ARTWORKS:\n" + "\n".join(lines))

    if sections:
        return "\n".join(sections) + "\n"

    return ""


def resolve_keywords(base_term: str) -> List[str]:
    term = base_term.lower()
    synonyms = {
        "modern": ["modern", "contemporary", "abstract", "minimalist"],
        "abstract": ["abstract", "modern", "contemporary"],
        "landscape": ["landscape", "coastal", "seascape", "nature", "mountain", "forest"],
        "seascape": ["seascape", "coastal", "ocean", "marine"],
        "floral": ["flowers", "floral", "botanical", "garden"],
        "textured": ["textured", "3d", "impasto", "raised", "sculpted"],
        "minimal": ["minimal", "minimalist", "modern", "clean"],
        "animal": ["animal", "wildlife", "safari", "lion", "horse", "bird"],
        "coastal": ["coastal", "seaside", "seascape", "beach"],
        "blue": ["blue", "ocean", "sky", "azure"],
        "gold": ["gold", "metallic", "luxury", "gilded"],
        "warm": ["warm", "earthy", "sunset"],
        "cool": ["cool", "icy", "calm"],
        "triptych": ["triptych", "set", "three-piece", "3-set"],
        "office": ["office", "workspace", "corporate"],
        "living": ["living room", "family room", "sofa"],
        "bedroom": ["bedroom", "relaxing", "calming"]
    }
    for key, values in synonyms.items():
        if term in values:
            return values
    return [term]


def search_catalog_for_request(
    message: str,
    catalog_context: Optional[Dict],
    max_results: int = 3
) -> Optional[Dict]:
    """
    Lightweight keyword search across catalog snapshot to find relevant products.
    """
    if not catalog_context or not message:
        return None

    base_words = [
        w.lower()
        for w in re.findall(r"[A-Za-z]{3,}", message)
        if len(w) >= 3
    ]
    if not base_words:
        return None

    expanded_terms = set()
    for w in base_words:
        for term in resolve_keywords(w):
            expanded_terms.add(term)

    products = catalog_context.get("products") or []
    matches = []
    for product in products:
        haystack = " ".join([
            product.get("title", ""),
            product.get("type", "") or "",
            product.get("tags", "") or "",
            product.get("description", "") or ""
        ]).lower()
        score = sum(1 for word in expanded_terms if word in haystack)
        if score > 0:
            matches.append((score, product))

    matches.sort(key=lambda x: x[0], reverse=True)
    top_matches = [p for _, p in matches[:max_results]]

    # Fallback collections if no matches
    fallbackCols = []
    if not top_matches:
        fallbackCols = (catalog_context.get("custom_collections") or [])[:max_results]

    if not top_matches and not fallbackCols:
        return None

    return {
        "query_terms": list(expanded_terms),
        "products": top_matches,
        "fallback_collections": fallbackCols
    }


def build_catalog_match_section(suggestions: Optional[Dict]) -> str:
    if not suggestions:
        return ""

    lines = []
    products = suggestions.get("products") or []

    if products:
        lines.append("MATCHED ARTWORKS (mention at least two before asking questions):")
        for product in products:
            title = product.get("title", "Artwork")
            price = product.get("price_range") or ""
            desc = product.get("description") or ""
            snippet = desc if len(desc) <= 100 else desc[:100] + "..."
            tags = product.get("tags") or ""
            lines.append(f"- {title} ({price}) • {tags}\n  {snippet}")

    fallbacks = suggestions.get("fallback_collections") or []
    if fallbacks:
        lines.append("SUGGESTED COLLECTIONS:")
        for col in fallbacks:
            title = col.get("title")
            body = col.get("body") or "Handmade art collection"
            lines.append(f"- {title}: {body}")

    if lines:
        return "\n".join(lines) + "\n"

    return ""

async def generate_ai_response_live(
    current_message: str,
    caller_email: Optional[str] = None,
    shopify_client: Optional[ShopifyClient] = None, # Kept for compatibility, though not used internally
    shopify_data_cache: Optional[dict] = None,
    caller_memory: Optional[dict] = None,
    conversation_history: Optional[List[Dict]] = None,
    order_context: Optional[Dict] = None,  # NEW: Specific order context
    catalog_context: Optional[Dict] = None,  # NEW: Catalog highlights for product knowledge
    catalog_suggestions: Optional[Dict] = None  # NEW: Products matched to current message
) -> Tuple[str, Optional[bytes]]:
    """
    Generate AI response with IMPROVED MEMORY and CONFIRMATION HANDLING.
    Returns (text_response, audio_bytes_8khz_pcm).
    """
    caller_memory = caller_memory or {}
    
    start_time = time.time()
    
    # NEW: Initialize caller_memory if not provided (safety)
    if caller_memory is None:
        caller_memory = {
            "order_numbers": [],
            "art_interests": [],
            "issues": [],
            "name": None,
            "pending_actions": [],
            "customer_id": None,
            "verified_orders": {},
            "customer_info": None,
            "all_orders": []
        }
    
    # NEW: Extract information from current message and update memory
    extract_caller_info(current_message, caller_memory)
    
    # NEW: Check customer verification status - FIXED: Customer is verified if we have customer_id
    customer_verified = caller_memory.get("customer_id") is not None
    
    # NEW: Get mentioned order for prompt logic
    mentioned_order = None
    order_pattern = r'\b([A-Z]{3}\d{5}[A-Z])\b'
    potential_orders = re.findall(order_pattern, current_message, re.IGNORECASE)
    if potential_orders:
        mentioned_order = potential_orders[0].upper()
    
    # Build customer context from pre-fetched data
    customer_orders_info = ""
    if shopify_data_cache:
        customer = shopify_data_cache.get("customer", {})
        orders = shopify_data_cache.get("orders", [])
        
        if customer:
            customer_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
            customer_orders_info = f"Customer: {customer.get('email', 'N/A')}"
            if customer_name:
                customer_orders_info += f" ({customer_name})"
            customer_orders_info += "\n"
            
            if orders:
                customer_orders_info += f"Customer has {len(orders)} total orders.\n"
                # List recent order names (not IDs)
                recent_orders = [o.get("name", "").replace("#", "") for o in orders[:5]]
                customer_orders_info += f"Customer's orders: {', '.join(recent_orders)}\n"
    
    # NEW: Build memory context
    memory_context = build_memory_context(caller_memory) if caller_memory else ""
    
    # NEW: Build order context if available
    order_info = build_order_context(order_context) if order_context else ""

    # NEW: Build catalog context
    catalog_info = build_catalog_highlights(catalog_context)
    catalog_matches = build_catalog_match_section(catalog_suggestions)
    
    # NEW: Build conversation context (use last 8 messages)
    recent_conversation = ""
    if conversation_history and len(conversation_history) > 1:
        # Exclude the current user message and get the last 8 messages
        recent_messages = conversation_history[:-1] 
        recent_messages = recent_messages[-8:] if len(recent_messages) > 8 else recent_messages
        
        recent_conversation = "RECENT CONVERSATION:\n" + "\n".join([
            f"{msg['role'].capitalize()}: {msg['text']}"
            for msg in recent_messages
        ]) + "\n"
    
    # NEW: Detect if caller is requesting an action for confirmation flow (no cancel_order)
    action_requested = detect_action_request(current_message)
    if action_requested:
        if action_requested not in caller_memory.get("pending_actions", []):
            caller_memory.setdefault("pending_actions", []).append(action_requested)
            print(f"🎯 Action detected and pending: {action_requested}")
    
    # NEW: Check if we need to ask for confirmation
    needs_confirmation = False
    confirmation_action = None
    if caller_memory.get("pending_actions"):
        # Check if last message contains confirmation words
        confirmation_words = ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'confirm', 'go ahead', 'do it']
        denial_words = ['no', 'nope', 'don\'t', 'nevermind', 'wait']
        
        text_lower = current_message.lower()
        has_confirmation = any(word in text_lower for word in confirmation_words)
        has_denial = any(word in text_lower for word in denial_words)
        
        if has_denial:
            # User denied, clear pending actions
            caller_memory["pending_actions"] = []
            print("❌ User denied action")
        elif has_confirmation:
            # User confirmed, we can proceed
            confirmation_action = caller_memory["pending_actions"][0]
            caller_memory["pending_actions"] = []
            print(f"✅ User confirmed action: {confirmation_action}")
        else:
            # Still waiting for confirmation
            needs_confirmation = True
            confirmation_action = caller_memory["pending_actions"][0]
    
    # FIXED: New verification guidance based on customer verification status
    verification_guidance = ""
    if not customer_verified:
        # Customer phone not found in system
        verification_guidance = """
IMPORTANT - CUSTOMER NOT VERIFIED:
- The customer's phone number was not found in our system
- They are calling from an unrecognized number
- If they ask about orders, politely explain: "I couldn't find your phone number in our system. This might be a different number than what's on file. Could you provide your order number so I can help you?"
- NEVER ask for email - just ask for the order number directly
- Once they provide an order number, you can look it up and assist them
"""
    else:
        # Customer verified - phone found
        verification_guidance = """
IMPORTANT - CUSTOMER VERIFIED:
- The customer's phone number IS in our system
- They are a known customer with existing orders
- You have their account information already
- Do NOT ask for email or phone - you already have it
- If they ask about orders, ask them: "Which order would you like to know about?" or "Which order number can I help you with?"
- Their order numbers are listed in the CALLER MEMORY section
"""
    
    # NEW: Build order verification guidance
    order_verification_guidance = ""
    # Check if an order was mentioned AND (it's not in verified cache OR there's no order context provided)
    if mentioned_order and not order_context and caller_memory.get("verified_orders") is not None and mentioned_order not in caller_memory.get("verified_orders", {}):
        if customer_verified:
            # Check if order exists in customer's orders
            customer_order_names = [o.get("name", "").replace("#", "").upper() for o in caller_memory.get("all_orders", [])]
            if mentioned_order not in customer_order_names:
                order_verification_guidance = f"""
IMPORTANT - ORDER VERIFICATION:
- Customer mentioned order {mentioned_order}
- This order was NOT found for this verified customer
- Politely say: "I couldn't find order {mentioned_order} under your account. Could you double-check the order number? Your orders are: {', '.join(customer_order_names[:5])}"
"""
        elif not customer_verified:
            order_verification_guidance = f"""
IMPORTANT - ORDER VERIFICATION:
- Customer mentioned order {mentioned_order}
- We're checking if this order exists and will provide details once verified
"""
    
    # IMPROVED: More comprehensive system prompt with memory and confirmation handling
    # FIXED: Remove all email-asking logic
    system_prompt = f"""You are "Maud," the friendly and professional AI sales agent for ArtByMaudsch.com — an online art gallery offering hand-painted artworks.

Customer Info:
- Phone: {caller_memory.get('customer_info', {}).get('phone', 'Caller phone')}
- Email: {caller_email or caller_memory.get('email') or caller_memory.get('customer_info', {}).get('email', 'On file')}
{customer_orders_info}

Catalog Highlights:
{catalog_info or 'Use your best judgment based on caller requests.'}

Live Catalog Matches:
{catalog_matches or 'If none are listed, ask a clarifying question before recommending.'}
If MATCHED ARTWORKS are provided, speak about at least two specific pieces before asking for more preferences.

{verification_guidance}

{order_verification_guidance}

Conversation Memory:
{memory_context}

{order_info}

{recent_conversation}

Your Purpose:
- Help customers with products, orders, payments, and shipping
- Offer helpful, genuine advice about choosing art
- Handle complaints calmly and with empathy
- Confirm clearly before making changes (refunds, returns, address changes)
- Use and remember customer details they've already given
- **CRITICAL: NEVER ASK FOR EMAIL - You either already have it (customer verified) or just need the order number (customer not verified)**
- If customer is verified by phone: Just ask for order number
- If customer is not verified by phone: Ask for order number to look up their order
- **IMPORTANT: You CANNOT cancel orders. If asked, politely explain that order cancellations must be processed by our team and provide them with contact information.**

Voice Style:
- Speak naturally, as if on a friendly call
- Keep it short: 10–20 words per message
- Use contractions ("I'm", "you're", "that's")
- Be warm, confident, and helpful — not robotic
- One clear thought per reply
- If unsure, ask one short question

Tone: Calm, friendly, and slightly enthusiastic — like a helpful art consultant.

Art-Only Policy:
If someone asks about anything unrelated to art, their order, or ArtByMaudsch:
→ Politely redirect with something like:
"I'm happy to chat about art or your order — would you like to explore some pieces?"
Never continue irrelevant topics.

Customer Verification Rules:
- If customer IS verified (phone found): You already have their info - just ask for order number
- If customer NOT verified (phone not found): Ask for order number to look it up
- **NEVER EVER ask for email address - we don't use email for verification**
- Once you have order number, provide information about that order

Order Reference Rules:
- When discussing orders, use the order NAME (e.g., "ABM27255X") not the order ID
- Customers will refer to orders by these names
- If customer mentions an order name, look it up in their order list
- If order not found for verified customer, suggest double-checking the order number and mention their actual orders

Order Status Information:
- You can tell customers about their order status, shipping info, and tracking
- You can provide information about items in their orders
- You CANNOT cancel orders - direct them to contact support for cancellations

Confirmation Rules (for refunds, returns, address changes):
Before taking action, always confirm:
"Just to confirm, you want me to [action] for order [name]?"
After 'yes': 
"I'll create a request for that. Our team will process the [action] for order [name] and contact you shortly."
After 'no':
"No problem! What else can I help with?"

Response Guidelines:
- Ask only one thing at a time
- Reference names and order names naturally (e.g., "your order ABM27255X")
- Never repeat questions already answered
- **CRITICAL: Never ask for email - just ask for order number**
- Avoid long lists or descriptions over the phone
- End with a light, friendly tone
- If mentioning you're "checking" or "looking up" something, keep it very brief (under 5 words)

Examples:
Customer (not verified): "Where's my order?"
You: "I'd love to help! What's your order number?"

Customer (verified): "I want to know about my orders."
You: "Sure! Which order would you like to know about?"

Customer (verified, wrong order): "Where's order ABM99999X?"
You: "I don't see order ABM99999X for you. Your orders are ABM27255X and ABM23658X."

Customer (verified): "What's the status of ABM27255X?"
You: "Order ABM27255X is paid and currently unfulfilled. It's being prepared for shipment."

Customer: "Cancel my order ABM27255X."
You: "I can't cancel orders directly, but I can help you submit a cancellation request. Should I do that?"

Customer: "What's your favorite movie?"
You: "I'm all about art, not movies — maybe I can show you some new paintings?"
"""

    # NEW: Handle confirmation flow
    if needs_confirmation and confirmation_action:
        order_num = mentioned_order or caller_memory.get("order_numbers", ["your order"])[0] if caller_memory.get("order_numbers") else "your order"
        action_text = {
            'refund': f'process a refund request for order {order_num}',
            'return': f'process a return request for order {order_num}',
            'change_address': f'update the address for order {order_num}'
        }.get(confirmation_action, 'proceed with that')
        
        # Force confirmation question
        ai_text = f"Just to confirm, you want me to {action_text}?"
        print(f"❓ Asking for confirmation: {ai_text}")
    
    elif confirmation_action:
        # User just confirmed, acknowledge action
        order_num = mentioned_order or caller_memory.get("order_numbers", ["your order"])[0] if caller_memory.get("order_numbers") else "your order"
        action_text = {
            'refund': f'refund request for order {order_num}',
            'return': f'return request for order {order_num}',
            'change_address': f'address update for order {order_num}'
        }.get(confirmation_action, 'that request')
        
        ai_text = f"I'll create a {action_text}. Our team will process it and contact you shortly."
        print(f"✅ Confirming action: {ai_text}")
    
    else:
        # Normal conversation flow
        # Generate AI text with improved context
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Respond naturally (max 20 words): {current_message}"}
            ]

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=45,
                    presence_penalty=0.6,
                    frequency_penalty=0.3,
                )
            )

            ai_text = response.choices[0].message.content.strip()
            
            # UPDATED: Force truncate if still too long (max 20 words for normal, 15 for questions)
            words = ai_text.split()
            soft_limit = 15 if '?' in ai_text else 20
            hard_limit = soft_limit + 5  # allow a few extra words to finish the thought
            
            if len(words) > hard_limit:
                ai_text = ' '.join(words[:hard_limit])
            elif len(words) > soft_limit:
                ai_text = ' '.join(words[:len(words)])
            
            # Ensure sentence ends cleanly
            if ai_text and not ai_text[-1] in '.!?':
                ai_text += '.'
            
            gpt_time = time.time() - start_time
            print(f"✅ AI text ({gpt_time:.2f}s): '{ai_text}'")

        except Exception as e:
            print(f"⚠️ AI error: {e}")
            ai_text = "I apologize, I missed that. How can I help you today?"

    # Convert to speech IMMEDIATELY
    ai_audio_8khz_pcm = None
    if ai_text:
        try:
            tts_start = time.time()
            eleven_client = get_elevenlabs_client()  # Reuse persistent client
            
            # Start TTS immediately
            raw_audio_bytes = await eleven_client.text_to_speech_fast(ai_text)
            
            if raw_audio_bytes and len(raw_audio_bytes) > 100:
                ai_audio_8khz_pcm = raw_audio_bytes
                tts_time = time.time() - tts_start
                print(f"✅ TTS ({tts_time:.2f}s): {len(ai_audio_8khz_pcm)} bytes")
            else:
                print(f"⚠️ TTS returned short audio")

        except Exception as e:
            print(f"⚠️ TTS error: {e}")

    total_time = time.time() - start_time
    print(f"⏱️ Total response time: {total_time:.2f}s")
    
    return ai_text, ai_audio_8khz_pcm

# UPDATED: Updated signature to be compatible with generate_ai_response_live
async def generate_ai_text_only(
    current_message: str,
    caller_email: Optional[str] = None,
    shopify_client: Optional[ShopifyClient] = None,
    shopify_data_cache: Optional[dict] = None,
    caller_memory: Optional[dict] = None,
    conversation_history: Optional[List[Dict]] = None,
    order_context: Optional[Dict] = None,
    catalog_context: Optional[Dict] = None,
    catalog_suggestions: Optional[Dict] = None
) -> str:
    """Generate AI response text without TTS."""
    ai_text, _ = await generate_ai_response_live(
        current_message,
        caller_email,
        shopify_client,
        shopify_data_cache,
        caller_memory,
        conversation_history,
        order_context,
        catalog_context,
        catalog_suggestions
    )
    return ai_text