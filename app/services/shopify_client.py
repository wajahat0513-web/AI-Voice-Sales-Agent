# app/services/shopify_client.py

import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time
import json

load_dotenv()


class ShopifyClient:
    """
    Shopify API client using REST Admin API with Access Token
    Handles customer orders, stock checking, and tagging
    ENHANCED: Better logging, GraphQL fallback, improved phone matching
    FIXED: Handles phones in addresses and updated GraphQL schema
    """

    def __init__(self):
        self.domain = os.getenv("SHOPIFY_DOMAIN")
        self.access_token = os.getenv("SHOPIFY_ADMIN_ACCESS_TOKEN")

        if not all([self.domain, self.access_token]):
            raise ValueError("❌ Missing Shopify credentials in .env (SHOPIFY_DOMAIN and SHOPIFY_ADMIN_ACCESS_TOKEN required)")

        self.api_version = "2024-10"
        self.base_url = f"https://{self.domain}/admin/api/{self.api_version}"
        self.graphql_url = f"{self.base_url}/graphql.json"

        self.session = requests.Session()
        self.session.headers.update({
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json"
        })

        print(f"✅ ShopifyClient initialized for {self.domain}")
        print(f"   Using Admin Access Token authentication")

    # ---------------------------------------------------------------
    # 🔹 HELPER METHODS
    # ---------------------------------------------------------------

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to digits only for comparison."""
        if not phone:
            return ""
        return ''.join(filter(str.isdigit, phone))

    def _phones_match(self, phone1: str, phone2: str) -> bool:
        """Compare phone numbers ignoring formatting and +1 prefixes."""
        if not phone1 or not phone2:
            return False
        digits1 = self._normalize_phone(phone1)
        digits2 = self._normalize_phone(phone2)
        if len(digits1) == 11 and digits1.startswith('1'):
            digits1 = digits1[1:]
        if len(digits2) == 11 and digits2.startswith('1'):
            digits2 = digits2[1:]
        return digits1 == digits2

    def _search_customers_graphql(self, query: str) -> List[Dict]:
        """Search customers using Shopify GraphQL API (updated schema)."""
        try:
            graphql_query = """
            query searchCustomers($query: String!) {
                customers(first: 5, query: $query) {
                    edges {
                        node {
                            id
                            email
                            phone
                            firstName
                            lastName
                            numberOfOrders
                            createdAt
                            addresses {
                                phone
                                address1
                                city
                                country
                            }
                        }
                    }
                }
            }
            """
            payload = {"query": graphql_query, "variables": {"query": query}}
            response = self.session.post(self.graphql_url, json=payload)

            if response.status_code != 200:
                print(f"    GraphQL request failed: {response.status_code}")
                return []

            data = response.json()
            if "errors" in data:
                print(f"    GraphQL errors: {data['errors']}")
                return []

            edges = data.get("data", {}).get("customers", {}).get("edges", [])
            customers = []
            for edge in edges:
                node = edge.get("node", {})
                gid = node.get("id", "")
                numeric_id = gid.split("/")[-1] if "/" in gid else None
                customers.append({
                    "id": int(numeric_id) if numeric_id and numeric_id.isdigit() else None,
                    "email": node.get("email"),
                    "phone": node.get("phone"),
                    "first_name": node.get("firstName"),
                    "last_name": node.get("lastName"),
                    "orders_count": node.get("numberOfOrders", 0),
                    "created_at": node.get("createdAt"),
                    "addresses": node.get("addresses", [])
                })
            return customers
        except Exception as e:
            print(f"    GraphQL search error: {e}")
            return []

    # ---------------------------------------------------------------
    # 🔹 MAIN METHOD: get_customer_by_phone
    # ---------------------------------------------------------------

    def get_customer_by_phone(self, phone: str) -> Optional[Dict]:
        """
        Find customer by phone number (checks both phone field and address phones).
        Uses REST Search first, then GraphQL fallback.
        """
        start_time = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"🔍 SHOPIFY: Searching for customer by phone")
            print(f"    Input: {phone}")

            digits_only = ''.join(filter(str.isdigit, phone))
            variants = [phone, digits_only]

            # Add +1 / formatted variants
            if len(digits_only) == 10:
                variants += [f"+1{digits_only}", f"1{digits_only}"]
            elif len(digits_only) == 11 and digits_only.startswith("1"):
                variants.append(digits_only[1:])

            # Deduplicate
            seen = set()
            phone_variants = [v for v in variants if not (v in seen or seen.add(v))]

            print(f"    Trying {len(phone_variants)} phone variants")

            # --- REST SEARCH ---
            for idx, variant in enumerate(phone_variants, 1):
                print(f"\n📡 REST Search {idx}/{len(phone_variants)}: '{variant}'")
                url = f"{self.base_url}/customers/search.json"
                response = self.session.get(url, params={"query": f"phone:{variant}"}, timeout=10)

                if response.status_code != 200:
                    print(f"    ⚠️ REST request failed: {response.status_code}")
                    continue

                customers = response.json().get("customers", [])
                print(f"    Found {len(customers)} potential match(es)")

                for customer in customers:
                    all_phones = [customer.get("phone")] + [
                        a.get("phone") for a in customer.get("addresses", []) if a.get("phone")
                    ]
                    all_phones = [p for p in all_phones if p]

                    for p in all_phones:
                        if self._phones_match(phone, p):
                            print(f"\n✅ SUCCESS: Customer found via REST Search!")
                            print(f"    Name: {customer.get('first_name')} {customer.get('last_name')}")
                            print(f"    ID: {customer.get('id')}")
                            print(f"    Email: {customer.get('email')}")
                            print(f"    Phones matched: {p}")
                            print(f"    Total orders: {customer.get('orders_count', 0)}")
                            print(f"{'='*60}\n")
                            return customer

                    print("    ⚠️ No phone match in this customer record.")
                    print(f"       Phones in record: {all_phones}")

            # --- GRAPHQL FALLBACK ---
            print("\n🔁 Falling back to GraphQL search...")
            graphql_queries = [f"phone:{v}" for v in phone_variants]
            for query in graphql_queries:
                print(f"    GraphQL query: {query}")
                customers = self._search_customers_graphql(query)
                if not customers:
                    continue

                for customer in customers:
                    all_phones = [customer.get("phone")] + [
                        a.get("phone") for a in customer.get("addresses", []) if a.get("phone")
                    ]
                    for p in all_phones:
                        if self._phones_match(phone, p):
                            print(f"\n✅ SUCCESS: Customer found via GraphQL!")
                            print(f"    Name: {customer.get('first_name')} {customer.get('last_name')}")
                            print(f"    ID: {customer.get('id')}")
                            print(f"    Email: {customer.get('email')}")
                            print(f"    Phones matched: {p}")
                            print(f"{'='*60}\n")
                            return customer

            print("\n❌ No matching customer found via REST or GraphQL.")
            print(f"{'='*60}\n")
            return None

        except Exception as e:
            print(f"\n❌ EXCEPTION in get_customer_by_phone: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            print(f"⏱️ Completed in {(time.time() - start_time):.2f}s")



    def get_customer_by_email(self, email: str) -> Optional[Dict]:
        """
        Find customer by email address
        ENHANCED: Dedicated method with better logging
        """
        start_time = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"📧 SHOPIFY: Searching for customer by email")
            print(f"    Email: {email}")
            
            url = f"{self.base_url}/customers.json"
            params = {"email": email}

            response = self.session.get(url, params=params, timeout=10)
            print(f"    Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"    Response body: {response.text[:300]}")
            
            response.raise_for_status()

            customers = response.json().get("customers", [])
            print(f"    Customers found: {len(customers)}")
            
            if customers:
                customer = customers[0]
                print(f"\n    ✅ SUCCESS: Customer found!")
                print(f"    Customer ID: {customer.get('id')}")
                print(f"    Name: {customer.get('first_name')} {customer.get('last_name')}")
                print(f"    Phone: {customer.get('phone')}")
                print(f"    Total orders: {customer.get('orders_count', 0)}")
                print(f"{'='*60}\n")
                return customer
            
            print(f"    ❌ FAILURE: No customer found for email: {email}")
            print(f"{'='*60}\n")
            return None

        except Exception as e:
            print(f"\n    ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return None
        finally:
            end_time = time.time()
            print(f"⏱️ SHOPIFY: get_customer_by_email completed in {(end_time - start_time):.2f}s")


    def get_customer_orders(self, customer_id: int) -> Optional[list]:
        """
        Retrieve all orders for a given Shopify customer ID.
        Returns a list of orders (empty list if none found).
        """
        start_time = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"📦 SHOPIFY: Fetching orders for customer ID: {customer_id}")

            url = f"{self.base_url}/orders.json"
            params = {"customer_id": customer_id, "status": "any", "limit": 50}

            response = self.session.get(url, params=params, timeout=15)
            print(f"    Response status: {response.status_code}")

            if response.status_code != 200:
                print(f"    Response text: {response.text[:300]}")
                return []

            orders = response.json().get("orders", [])
            print(f"    Orders found: {len(orders)}")

            if not orders:
                print(f"    ❌ No orders found for customer {customer_id}")
                return []

            formatted_orders = []
            for order in orders:
                formatted_orders.append({
                    "id": order.get("id"),
                    "name": order.get("name"),
                    "total_price": order.get("total_price"),
                    "currency": order.get("currency"),
                    "financial_status": order.get("financial_status"),
                    "fulfillment_status": order.get("fulfillment_status"),
                    "created_at": order.get("created_at"),
                    "line_items": [
                        {
                            "title": item.get("title"),
                            "quantity": item.get("quantity"),
                            "price": item.get("price")
                        } for item in order.get("line_items", [])
                    ]
                })

            print(f"\n✅ SUCCESS: Retrieved {len(formatted_orders)} orders for customer {customer_id}")
            print(f"{'='*60}\n")
            return formatted_orders

        except Exception as e:
            print(f"⚠️ Error fetching customer orders: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            print(f"⏱️ SHOPIFY: get_customer_orders completed in {(time.time() - start_time):.2f}s")


    def verify_order_number(self, order_name: str, customer_id: Optional[int] = None) -> Optional[Dict]:
        """
        Verify if an order number exists and optionally belongs to a customer
        ENHANCED: Better logging
        """
        start_time = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"🔍 SHOPIFY: Verifying order number")
            print(f"    Order name: {order_name}")
            print(f"    Customer ID: {customer_id or 'Not provided'}")
            
            # Normalize order name
            if not order_name.startswith("#"):
                order_name = f"#{order_name}"
            
            print(f"    Normalized: {order_name}")
            
            url = f"{self.base_url}/orders.json"
            params = {"name": order_name, "status": "any"}

            response = self.session.get(url, params=params, timeout=10)
            print(f"    Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"    Response body: {response.text[:300]}")
            
            response.raise_for_status()

            orders = response.json().get("orders", [])
            print(f"    Orders found: {len(orders)}")
            
            if not orders:
                print(f"    ❌ FAILURE: Order {order_name} not found")
                print(f"{'='*60}\n")
                return None
            
            order = orders[0]
            order_customer = order.get("customer", {})
            order_customer_id = order_customer.get("id") if order_customer else None
            
            print(f"\n    Order found:")
            print(f"      Number: {order.get('name')}")
            print(f"      Customer ID: {order_customer_id}")
            print(f"      Total: ${order.get('total_price')}")
            print(f"      Status: {order.get('financial_status')}")
            
            # If customer_id provided, verify ownership
            if customer_id:
                if order_customer_id != customer_id:
                    print(f"\n    ❌ FAILURE: Order belongs to different customer")
                    print(f"       Expected: {customer_id}")
                    print(f"       Actual: {order_customer_id}")
                    print(f"{'='*60}\n")
                    return None
                else:
                    print(f"    ✅ Ownership verified")
            
            print(f"\n    ✅ SUCCESS: Order verified")
            print(f"{'='*60}\n")
            return order

        except Exception as e:
            print(f"\n    ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return None
        finally:
            end_time = time.time()
            print(f"⏱️ SHOPIFY: verify_order_number completed in {(end_time - start_time):.2f}s")

    def get_order_details(self, order_name: str) -> Optional[Dict]:
        """
        Get detailed information about an order
        ENHANCED: Better logging and formatting
        """
        start_time = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"📦 SHOPIFY: Fetching order details")
            print(f"    Order name: {order_name}")
            
            if not order_name.startswith("#"):
                order_name = f"#{order_name}"
            
            url = f"{self.base_url}/orders.json"
            params = {"name": order_name, "status": "any"}

            response = self.session.get(url, params=params, timeout=10)
            print(f"    Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"    Response body: {response.text[:300]}")
            
            response.raise_for_status()

            orders = response.json().get("orders", [])
            if not orders:
                print(f"    ❌ FAILURE: Order not found")
                print(f"{'='*60}\n")
                return None
            
            order = orders[0]
            
            # Format order details
            details = {
                "order_number": order.get("name"),
                "order_id": order.get("id"),
                "total": order.get("total_price"),
                "currency": order.get("currency"),
                "status": order.get("financial_status"),
                "fulfillment_status": order.get("fulfillment_status", "unfulfilled"),
                "created_at": order.get("created_at"),
                "items": [],
                "shipping_address": order.get("shipping_address"),
                "tracking_numbers": []
            }
            
            # Extract line items
            for item in order.get("line_items", []):
                details["items"].append({
                    "title": item.get("title"),
                    "quantity": item.get("quantity"),
                    "price": item.get("price")
                })
            
            # Extract tracking numbers from fulfillments
            for fulfillment in order.get("fulfillments", []):
                tracking_number = fulfillment.get("tracking_number")
                if tracking_number:
                    details["tracking_numbers"].append(tracking_number)
            
            print(f"\n    ✅ SUCCESS: Order details retrieved")
            print(f"      Order: {details['order_number']}")
            print(f"      Total: ${details['total']} {details['currency']}")
            print(f"      Status: {details['status']}")
            print(f"      Fulfillment: {details['fulfillment_status']}")
            print(f"      Items: {len(details['items'])}")
            print(f"      Tracking numbers: {len(details['tracking_numbers'])}")
            print(f"{'='*60}\n")
            
            return details

        except Exception as e:
            print(f"\n    ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return None
        finally:
            end_time = time.time()
            print(f"⏱️ SHOPIFY: get_order_details completed in {(end_time - start_time):.2f}s")


    def check_stock(self, product_id: int) -> int:
        """Check stock quantity for a product"""
        try:
            url = f"{self.base_url}/products/{product_id}.json"
            response = self.session.get(url, timeout=10)
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

            response = self.session.get(url, params=params, timeout=10)
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

                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                order = response.json().get("order", {})

                existing_tags = order.get("tags", "").split(",")
                existing_tags = [t.strip() for t in existing_tags if t.strip()]
                new_tags = list(set(existing_tags + tags))

                payload = {"order": {"id": order_id, "tags": ", ".join(new_tags)}}
                response = self.session.put(url, json=payload, timeout=10)
                response.raise_for_status()
                print(f"✅ Tagged order {order_id} with: {tags}")
                return True

            elif customer_id:
                url = f"{self.base_url}/customers/{customer_id}.json"

                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                customer = response.json().get("customer", {})

                existing_tags = customer.get("tags", "").split(",")
                existing_tags = [t.strip() for t in existing_tags if t.strip()]
                new_tags = list(set(existing_tags + tags))

                payload = {"customer": {"id": customer_id, "tags": ", ".join(new_tags)}}
                response = self.session.put(url, json=payload, timeout=10)
                response.raise_for_status()
                print(f"✅ Tagged customer {customer_id} with: {tags}")
                return True

            return False

        except Exception as e:
            print(f"⚠️ Error tagging: {e}")
            return False