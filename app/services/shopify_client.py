# app/services/shopify_client.py

import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class ShopifyClient:
    """
    Shopify API client using REST Admin API
    Handles customer orders, stock checking, and tagging
    """

    def __init__(self):
        self.domain = os.getenv("SHOPIFY_DOMAIN")
        self.api_key = os.getenv("SHOPIFY_API_KEY")
        self.password = os.getenv("SHOPIFY_PASSWORD")

        if not all([self.domain, self.api_key, self.password]):
            raise ValueError("❌ Missing Shopify credentials in .env")

        # Use 2024-10 API version (stable)
        self.api_version = "2024-10"
        self.base_url = f"https://{self.api_key}:{self.password}@{self.domain}/admin/api/{self.api_version}"

        self.session = requests.Session()
        self.session.auth = (self.api_key, self.password)

        print(f"✅ ShopifyClient initialized for {self.domain}")

    def get_customer_orders(self, email: str) -> List[Dict]:
        """
        Get all orders for a customer by email
        FIXED: Use correct endpoint /customers.json with email filter
        """
        try:
            # Correct endpoint with email filter
            customers_url = f"{self.base_url}/customers.json"
            params = {"email": email}

            response = self.session.get(customers_url, params=params)
            response.raise_for_status()

            customers = response.json().get("customers", [])

            if not customers:
                print(f"ℹ️  No customer found for {email}")
                return []

            customer_id = customers[0]["id"]

            # Get orders for this customer
            orders_url = f"{self.base_url}/orders.json"
            orders_params = {
                "customer_id": customer_id,
                "status": "any",
                "limit": 5
            }

            orders_response = self.session.get(orders_url, params=orders_params)
            orders_response.raise_for_status()

            orders = orders_response.json().get("orders", [])
            print(f"✅ Found {len(orders)} orders for {email}")
            return orders

        except requests.exceptions.HTTPError as e:
            print(f"⚠️ Shopify API error: {e}")
            print(f"   Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            return []
        except Exception as e:
            print(f"⚠️ Shopify error: {e}")
            return []

    def check_stock(self, product_id: int) -> int:
        """Check stock quantity for a product"""
        try:
            url = f"{self.base_url}/products/{product_id}.json"
            response = self.session.get(url)
            response.raise_for_status()

            product = response.json().get("product", {})
            variants = product.get("variants", [])

            total_stock = sum(v.get("inventory_quantity", 0) for v in variants)
            return total_stock

        except Exception as e:
            print(f"⚠️ Error checking stock: {e}")
            return 0

    def get_shipping_estimate(self, order_name: str) -> str:
        """Get shipping estimate for an order"""
        try:
            url = f"{self.base_url}/orders.json"
            params = {"name": order_name, "status": "any"}

            response = self.session.get(url, params=params)
            response.raise_for_status()

            orders = response.json().get("orders", [])
            if not orders:
                return "Order not found"

            order = orders[0]
            fulfillment_status = order.get("fulfillment_status", "unfulfilled")

            if fulfillment_status == "fulfilled":
                return "Order has been shipped"
            elif fulfillment_status == "partial":
                return "Order partially shipped"
            else:
                return "3-5 business days"

        except Exception as e:
            print(f"⚠️ Error getting shipping estimate: {e}")
            return "Unable to estimate shipping time"

    def tag_customer_or_order(
        self,
        customer_id: Optional[int] = None,
        order_id: Optional[int] = None,
        tags: List[str] = None
    ) -> bool:
        """Add tags to a customer or order"""
        if not tags:
            return False

        try:
            if order_id:
                url = f"{self.base_url}/orders/{order_id}.json"

                response = self.session.get(url)
                response.raise_for_status()
                order = response.json().get("order", {})

                existing_tags = order.get("tags", "").split(",")
                existing_tags = [t.strip() for t in existing_tags if t.strip()]
                new_tags = list(set(existing_tags + tags))

                payload = {"order": {"id": order_id, "tags": ", ".join(new_tags)}}
                response = self.session.put(url, json=payload)
                response.raise_for_status()
                print(f"✅ Tagged order {order_id} with: {tags}")
                return True

            elif customer_id:
                url = f"{self.base_url}/customers/{customer_id}.json"

                response = self.session.get(url)
                response.raise_for_status()
                customer = response.json().get("customer", {})

                existing_tags = customer.get("tags", "").split(",")
                existing_tags = [t.strip() for t in existing_tags if t.strip()]
                new_tags = list(set(existing_tags + tags))

                payload = {"customer": {"id": customer_id, "tags": ", ".join(new_tags)}}
                response = self.session.put(url, json=payload)
                response.raise_for_status()
                print(f"✅ Tagged customer {customer_id} with: {tags}")
                return True

            return False

        except Exception as e:
            print(f"⚠️ Error tagging: {e}")
            return False

    def get_customer_by_phone(self, phone: str) -> Optional[Dict]:
        """Find customer by phone number"""
        try:
            url = f"{self.base_url}/customers.json"
            params = {"phone": phone}

            response = self.session.get(url, params=params)
            response.raise_for_status()

            customers = response.json().get("customers", [])
            return customers[0] if customers else None

        except Exception as e:
            print(f"⚠️ Error finding customer by phone: {e}")
            return None
