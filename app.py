# app.py
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import sys

# Load pre-trained ResNeXt model
model = models.resnext50_32x4d(pretrained=True)
model.eval()

# Define image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(img_path):
    img = Image.open(img_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)
        _, predicted = torch.max(out, 1)
        return predicted.item()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <image_path>")
    else:
        prediction = predict(sys.argv[1])
        print(f'Predicted class index: {prediction}')
