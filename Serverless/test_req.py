import requests
import base64
import json


URL = 'https://h4jciligth.execute-api.us-east-2.amazonaws.com/default/pp21'

with open('kitten.jpg', 'rb') as img:
    payload = img.read()

im_b64 = base64.b64encode(payload).decode("utf8")
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
payload = json.dumps({"image": im_b64, "other_key": "value"})

res = requests.post(url=URL, data=payload, headers=headers)

print(res)
print(res.text)
