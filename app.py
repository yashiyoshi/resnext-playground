import torch
import torchvision.models as models

# resnext with 50 layers and C=32
model = models.resnext50_32x4d(pretrained=True)
model.eval()

# dummy data
dummy_image = torch.rand(1, 3, 224, 224) # 1 image/batch, 3 channels, 224x224

with torch.no_grad():
    output = model(dummy_image)

# Print the output
print(output)
