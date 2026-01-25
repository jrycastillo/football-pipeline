import requests

SBG_BASE = "https://api-staging.scoutbridge.net/football-gallery/api"
SBG_LIST_URL = f"{SBG_BASE}/v2/files/list/video/for-match-analysis"
SBG_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3Njk1OTExNDQsInN1YiI6ImFudG9uaW9qaGFuY2Vkcmljays1QGdtYWlsLmNvbSIsInVzZXJfaWQiOiJlZTIyYWFlNiJ9.r_bPNFgPTEM0kJyiPig29A2TJVBtJTRmQSLPQwR5i2A"
TARGET_USER = "01c073e5"

def get_url():
    headers = {
        "Authorization": f"Bearer {SBG_TOKEN}",
        "Content-Type": "application/json"
    }
    resp = requests.get(SBG_LIST_URL, headers=headers, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        items = data.get("items", [])
        for item in items:
            path = item.get("fileLocation", "")
            if TARGET_USER in path:
                print(item.get("spacesURL"))
                return

if __name__ == "__main__":
    get_url()
