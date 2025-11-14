# app/services/zendesk_client.py

import os
import requests
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class ZendeskClient:
    """
    Zendesk API client for ticket creation and management
    Uses Zendesk API v2 with token authentication
    """
    
    def __init__(self):
        self.subdomain = os.getenv("ZENDESK_SUBDOMAIN")
        self.email = os.getenv("ZENDESK_EMAIL")
        self.api_token = os.getenv("ZENDESK_API_TOKEN")
        
        if not all([self.subdomain, self.email, self.api_token]):
            raise ValueError("❌ Missing Zendesk credentials in .env")
        
        self.base_url = f"https://{self.subdomain}.zendesk.com/api/v2"
        
        # Authentication: email/token:api_token
        self.session = requests.Session()
        self.session.auth = (f"{self.email}/token", self.api_token)
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        print(f"✅ ZendeskClient initialized for {self.subdomain}.zendesk.com")
    
    def create_ticket(
        self,
        subject: str,
        description: str,
        requester_email: str,
        tags: Optional[List[str]] = None,
        priority: str = "normal"
    ) -> Optional[int]:
        """
        Create a new Zendesk ticket
        
        Args:
            subject: Ticket subject line
            description: Full ticket description/body
            requester_email: Customer email address
            tags: List of tags to add
            priority: Ticket priority (low, normal, high, urgent)
            
        Returns:
            Ticket ID if successful, None otherwise
        """
        try:
            url = f"{self.base_url}/tickets.json"
            
            payload = {
                "ticket": {
                    "subject": subject,
                    "comment": {
                        "body": description
                    },
                    "requester": {
                        "email": requester_email,
                        "name": requester_email.split("@")[0]  # Use email prefix as name
                    },
                    "priority": priority,
                    "tags": tags or [],
                    "type": "task"  # or "question", "incident", "problem"
                }
            }
            
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            ticket_data = response.json()
            ticket_id = ticket_data["ticket"]["id"]
            
            print(f"✅ Zendesk ticket created: #{ticket_id}")
            return ticket_id
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ Zendesk API error: {e}")
            print(f"   Status: {e.response.status_code}")
            print(f"   Response: {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ Error creating Zendesk ticket: {e}")
            return None
    
    def update_ticket(
        self,
        ticket_id: int,
        comment: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Update an existing Zendesk ticket
        
        Args:
            ticket_id: Zendesk ticket ID
            comment: New comment to add
            status: New status (new, open, pending, hold, solved, closed)
            tags: Tags to add
            
        Returns:
            Success boolean
        """
        try:
            url = f"{self.base_url}/tickets/{ticket_id}.json"
            
            payload = {"ticket": {}}
            
            if comment:
                payload["ticket"]["comment"] = {"body": comment}
            
            if status:
                payload["ticket"]["status"] = status
            
            if tags:
                payload["ticket"]["tags"] = tags
            
            response = self.session.put(url, json=payload)
            response.raise_for_status()
            
            print(f"✅ Ticket #{ticket_id} updated")
            return True
            
        except Exception as e:
            print(f"❌ Error updating ticket: {e}")
            return False
    
    def get_ticket(self, ticket_id: int) -> Optional[dict]:
        """
        Get ticket details
        
        Args:
            ticket_id: Zendesk ticket ID
            
        Returns:
            Ticket dictionary or None
        """
        try:
            url = f"{self.base_url}/tickets/{ticket_id}.json"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json().get("ticket")
            
        except Exception as e:
            print(f"❌ Error fetching ticket: {e}")
            return None
    
    def search_tickets_by_requester(self, email: str) -> List[dict]:
        """
        Search tickets by requester email
        
        Args:
            email: Requester email address
            
        Returns:
            List of ticket dictionaries
        """
        try:
            url = f"{self.base_url}/search.json"
            params = {
                "query": f"type:ticket requester:{email}"
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            return response.json().get("results", [])
            
        except Exception as e:
            print(f"❌ Error searching tickets: {e}")
            return []