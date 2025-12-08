from ultralytics import YOLO
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transform as transforms
from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
    
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        return image_tensor
    
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}", file=sys.stderr)
        return None
    
def get_image_features(image_path, model):
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return None
    model.eval()
    with torch.no_grad(): 
        features = model(image_tensor)
        
    return features
   
def calculate_similarity(features1, features2):
    similarity_score = F.cosine_similarity(features1, features2, dim=1)
    return similarity_score.item()

def run(img1, img2):
    feature_extractor = FeatureExtractor()
    print("Extracting features...")
    features1 = get_image_features(img1, feature_extractor)
    features2 = get_image_features(img2, feature_extractor)

    if features1 is not None and features2 is not None:
        print("\n--- Similarity Scores ---")
        sim_1_2 = calculate_similarity(features1, features2)
        
    return f"  (Image 1 vs Image 2): {sim_1_2:.4f}"