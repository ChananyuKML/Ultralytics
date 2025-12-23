# Define the JSON data payload for the request body
$jsonPayload = @"
{
  "dataroot": "17",
  "epochs": 10,
  "task": "obb",
  "model": "yolo12",
  "size": "n",
  "device": "cuda:0",
  "batch": 1,
  "workers": 1
}
"@

# Define the target URL (assuming your Docker container is running on localhost)
$url = "http://localhost:8000/training"

# Send the POST request with the JSON body
Invoke-RestMethod -Uri $url -Method Post -Body $jsonPayload -ContentType "application/json"