import sys
import json
import base64
import time

token = sys.argv[1]
try:
    # Decode header and payload
    parts = token.split('.')
    header = json.loads(base64.urlsafe_b64decode(parts[0] + "==").decode('utf-8'))
    payload = json.loads(base64.urlsafe_b64decode(parts[1] + "==").decode('utf-8'))
    
    print("Header:", header)
    print("Payload:", payload)
    
    exp = payload.get('exp')
    if exp:
        now = time.time()
        print(f"Expires: {exp} (Current: {now})")
        print(f"Remaining: {exp - now} seconds")
        if exp < now:
            print("STATUS: EXPIRED")
        else:
            print("STATUS: VALID (Time-wise)")
    else:
        print("STATUS: NO EXPIRY")
        
except Exception as e:
    print(f"Error decoding: {e}")
