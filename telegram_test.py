import requests

# Test bot
url = f"https://api.telegram.org/bot8338190440:AAHPj9HfF3bDVCFvCl65dzM5S9M1p-8wf5c/getMe"
response = requests.get(url)
print("Bot test:", response.json())

# Test message
url = f"https://api.telegram.org/bot8338190440:AAHPj9HfF3bDVCFvCl65dzM5S9M1p-8wf5c/sendMessage"
data = {"chat_id": "-1004985758286", "text": "Test message"}
response = requests.post(url, json=data)
print("Message test:", response.json())