# test_shopify.py
import requests
from dotenv import load_dotenv
from services.shopify_client import ShopifyClient

load_dotenv()


def find_customer_by_phone(phone_number: str):
    print("\n" + "=" * 70)
    print(f"🔍 TESTING SHOPIFY CUSTOMER LOOKUP FOR: {phone_number}")
    print("=" * 70)

    client = ShopifyClient()

    # Get customer
    customer = client.get_customer_by_phone(phone_number)
    if not customer:
        print(f"\n❌ No customer found for phone number: {phone_number}")
        print("=" * 70)
        return

    # ✅ Print customer info
    print(f"\n✅ CUSTOMER FOUND FOR PHONE {phone_number}")
    print("-" * 70)
    print(f"🆔 ID: {customer.get('id')}")
    print(f"👤 Name: {customer.get('first_name')} {customer.get('last_name')}")
    print(f"📧 Email: {customer.get('email')}")
    print(f"📞 Phone (top-level): {customer.get('phone') or 'None'}")

    # Address phones
    addresses = customer.get("addresses", [])
    if addresses:
        print("🏠 Address phone(s):")
        for addr in addresses:
            addr_phone = addr.get("phone")
            address_summary = f"{addr.get('address1', '')}, {addr.get('city', '')}, {addr.get('country', '')}"
            print(f"   - {addr_phone} ({address_summary})")
    else:
        print("🏠 No addresses found for this customer.")

    # ============================================================
    # 🧾 Fetch orders for this customer (read_orders scope required)
    # ============================================================
    print("\n" + "=" * 70)
    print(f"📦 FETCHING ORDERS for {customer.get('first_name')} {customer.get('last_name')}")
    print("=" * 70)

    orders = client.get_customer_orders(customer_id=customer.get("id"))
    if not orders:
        print("❌ No orders found for this customer.")
        return

    print(f"✅ Found {len(orders)} order(s):\n")
    for order in orders:
        print(f"🧾 Order ID: {order['id']}")
        print(f"   🏷️ Name: {order['name']}")
        print(f"   💲 Total: {order['total_price']} {order['currency']}")
        print(f"   🗓️ Created: {order['created_at']}")
        print(f"   💳 Financial: {order['financial_status']}")
        print(f"   📦 Fulfillment: {order['fulfillment_status'] or 'Unfulfilled'}")
        print(f"   🛍️ Items: {len(order['line_items'])}")
        print("-" * 50)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # 🔢 Replace with any number you'd like to test
    test_phone = "+17786523395"  # Mustafa Zaka’s number
    find_customer_by_phone(test_phone)
