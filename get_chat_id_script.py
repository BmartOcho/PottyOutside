import requests
import json

# Your bot token
BOT_TOKEN = "8338190440:AAHPj9HfF3bDVCFvCl65dzM5S9M1p-8wf5c"

def get_updates():
    """Get recent messages sent to the bot to find chat IDs"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    
    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('ok'):
                updates = data.get('result', [])
                if updates:
                    print("📨 Recent messages:")
                    print("-" * 50)
                    
                    chat_ids = set()
                    for update in updates[-10:]:  # Show last 10 updates
                        if 'message' in update:
                            msg = update['message']
                            chat = msg.get('chat', {})
                            chat_id = chat.get('id')
                            chat_type = chat.get('type')
                            chat_title = chat.get('title', 'N/A')
                            
                            from_user = msg.get('from', {})
                            username = from_user.get('username', 'N/A')
                            first_name = from_user.get('first_name', 'N/A')
                            
                            text = msg.get('text', 'N/A')
                            date = msg.get('date', 'N/A')
                            
                            print(f"Chat ID: {chat_id}")
                            print(f"Chat Type: {chat_type}")
                            print(f"Chat Title: {chat_title}")
                            print(f"From: {first_name} (@{username})")
                            print(f"Message: {text}")
                            print(f"Date: {date}")
                            print("-" * 30)
                            
                            chat_ids.add(chat_id)
                    
                    print("\n🎯 Available Chat IDs:")
                    for chat_id in sorted(chat_ids):
                        print(f"  {chat_id}")
                    
                    print("\n💡 Instructions:")
                    print("1. Use one of the Chat IDs above in your script")
                    print("2. If no messages shown, send a message to your bot first")
                    print("3. For groups: add the bot to the group and send a message mentioning the bot")
                    
                else:
                    print("❌ No recent messages found.")
                    print("\n💡 To get your chat ID:")
                    print("1. Send a message to your bot (@DogPotty_bot)")
                    print("2. Or add the bot to a group and send a message")
                    print("3. Then run this script again")
            else:
                print(f"❌ API error: {data}")
        else:
            print(f"❌ HTTP error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting updates: {e}")

def test_chat_id(chat_id):
    """Test if a specific chat ID works"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"✅ Test successful! This chat ID ({chat_id}) works with your PuppyCam bot."
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('ok'):
                print(f"✅ Chat ID {chat_id} works!")
                return True
            else:
                print(f"❌ Chat ID {chat_id} failed: {data.get('description')}")
                return False
        else:
            print(f"❌ HTTP error for {chat_id}: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing {chat_id}: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Telegram Chat ID Finder")
    print("=" * 50)
    
    # First, get recent updates to see available chats
    get_updates()
    
    print("\n" + "=" * 50)
    print("🧪 Testing your current chat ID...")
    
    # Test the current chat ID
    current_chat_id = "-1004985758286"
    test_chat_id(current_chat_id)
    
    print("\n" + "=" * 50)
    print("🔧 Manual Testing")
    print("You can also test specific chat IDs by running:")
    print("python -c \"from telegram_test import test_chat_id; test_chat_id('YOUR_CHAT_ID')\"")