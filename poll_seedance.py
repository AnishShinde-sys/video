import requests
import json
import time

api_key = "MP10190B9U3F0FBNI5AYOR0PYBYZHDT6EX4M"
prediction_id = "a5d9caaf-a868-4872-b3f0-503b08f21122"
url = f"https://api.eachlabs.ai/v1/prediction/{prediction_id}"

headers = {"X-API-Key": api_key}

print(f"Polling prediction: {prediction_id}")
for i in range(5):
    response = requests.get(url, headers=headers)
    print(f"\n[Attempt {i+1}] Status Code: {response.status_code}")
    if response.ok:
        result = response.json()
        print(json.dumps(result, indent=2))
        if result.get("status") == "success":
            break
    time.sleep(2)
