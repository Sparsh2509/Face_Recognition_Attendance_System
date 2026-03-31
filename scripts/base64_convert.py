import requests
import base64

url = "https://res.cloudinary.com/dzcwomu3h/image/upload/v1749470300/Achal_Test_g2gzox.jpg"
resp = requests.get(url)
resp.raise_for_status()

b64 = base64.b64encode(resp.content).decode("utf-8")
payload = "data:image/jpeg;base64," + b64

# Write to disk
with open("achal_base64.txt", "w") as f:
    f.write(payload)

print("Base64 written to achal_base64.txt")