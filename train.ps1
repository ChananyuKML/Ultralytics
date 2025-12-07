# Define the JSON data payload for the request body
$jsonPayload = @"
{
  "path": "C:/Users/YourUser/project/yolo",
  "dataroot": "datasets/coco8",
  "train_path": "datasets/coco8/images/train",
  "val_path": "datasets/coco8/images/val",
  "test_path": "datasets/coco8/images/test",
  "train_label_path": "datasets/coco8/labels/train",
  "val_label_path": "datasets/coco8/labels/val",
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