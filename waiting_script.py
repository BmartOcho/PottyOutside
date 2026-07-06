import requests
import time
import json

BOT_TOKEN = "8338190440:AAHPj9HfF3bDVCFvCl65dzM5S9M1p-8wf5c"

def wait_for_message():
    """Wait for a new message and show the chat ID"""
    print("🤖 Waiting for a message to your bot...")
    print("📱 Please send a message to @DogPotty_bot now!")
    print("   You can send anything like: 'Hello' or 'Test'")
    print("   Or add the bot to a group and mention it: '@DogPotty_bot hello'")
    print()
    print("⏳ Waiting... (Press Ctrl+C to stop)")
    
    last_update_id = 0
    
    # Get current updates to establish baseline
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('ok') and data.get('result'):
                last_update_id = max([update['update_id'] for update in data['result']])
                print(f"📋 Starting from update ID: {last_update_id}")
    except Exception as e:
        print(f"Warning: Could not get baseline updates: {e}")
    
    try:
        while True:
            # Get new updates only
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            if last_update_id > 0:
                url += f"?offset={last_update_id + 1}"
            
            response = requests.get(url, timeout=10)
            if response.ok:
                data = response.json()
                if data.get('ok'):
                    updates = data.get('result', [])
                    
                    for update in updates:
                        if 'message' in update:
                            msg = update['message']
                            chat = msg.get('chat', {})
                            chat_id = chat.get('id')
                            chat_type = chat.get('type')
                            chat_title = chat.get('title', 'Private Chat')
                            
                            from_user = msg.get('from', {})
                            username = from_user.get('username', 'N/A')
                            first_name = from_user.get('first_name', 'N/A')
                            text = msg.get('text', 'N/A')
                            
                            print("🎉 NEW MESSAGE RECEIVED!")
                            print("=" * 50)
                            print(f"✅ Chat ID: {chat_id}")
                            print(f"📱 Chat Type: {chat_type}")
                            print(f"🏷️  Chat Title: {chat_title}")
                            print(f"👤 From: {first_name} (@{username})")
                            print(f"💬 Message: {text}")
                            print("=" * 50)
                            
                            # Test this chat ID immediately
                            print(f"🧪 Testing chat ID {chat_id}...")
                            test_success = test_chat_id(chat_id)
                            
                            if test_success:
                                print(f"🎯 PERFECT! Use this in your PuppyCam script:")
                                print(f'TELEGRAM_CHAT_ID = "{chat_id}"')
                                print()
                                print("💡 Copy this line into your PuppyCam config!")
                                return chat_id
                            
                        last_update_id = max(last_update_id, update['update_id'])
                    
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopped waiting for messages.")
        return None
    except Exception as e:
        print(f"❌ Error while waiting: {e}")
        return None

def test_chat_id(chat_id):
    """Test if a chat ID works by sending a test message"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"🎉 SUCCESS! Your PuppyCam bot can send messages to this chat!\n\nChat ID: {chat_id}\n\nYou can now use this bot for dog potty notifications! 🐶"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('ok'):
                print(f"✅ Test message sent successfully!")
                return True
            else:
                print(f"❌ Test message failed: {data.get('description')}")
                return False
        else:
            print(f"❌ HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error sending test message: {e}")
        return False

if __name__ == "__main__":
    print("🐶 PuppyCam Telegram Setup")
    print("=" * 50)
    
    chat_id = wait_for_message()
    
    if chat_id:
        print(f"\n🏁 Setup complete! Your chat ID is: {chat_id}")
    else:
        print("\n❌ Setup incomplete. Please try again.")
        print("💡 Make sure to send a message to @DogPotty_bot first!")