import bentoml
import torch
from PIL import Image
from torchvision import transforms
from src.models.Classifiers import RoomClassifier 
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.container import Sequential
from torch.nn.modules.linear import Linear

torch.serialization.add_safe_globals([RoomClassifier, ResNet, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Sequential, BasicBlock, Linear])
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 60}
)
class RoomService:
    bento_model = bentoml.models.BentoModel("room_classifier:latest")

    def __init__(self):
        self.model = bentoml.pytorch.load_model(self.bento_model)
        self.model.eval()

        self.class_names = self.bento_model.info.metadata.get("class_names", [])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @bentoml.api
    def classify(self, img: Image.Image) -> dict:
        tensor = self.transform(img).unsqueeze(0) 
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            
            topk_probs, topk_indices = torch.topk(probs, k=3, dim=1)
            
        top_predictions = []
        for i in range(3):
            idx = int(topk_indices[0][i].item())
            confidence = float(topk_probs[0][i].item())
            
            top_predictions.append({
                "label": self.class_names[idx] if self.class_names else f"ID_{idx}",
                "confidence": round(confidence, 4),
                "index": idx
            })
            
        return {
            "top_prediction": top_predictions[0]["label"],
            "all_results": top_predictions
        }