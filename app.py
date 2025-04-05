import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 38  # change to your actual number of classes

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, num_classes)
model.load_state_dict(torch.load("mobilenet_pretrained.pth", map_location=device))
model.eval().to(device)

# Class labels (replace with your actual class names)
class_names = train_dataset.classes  # or manually list: ['Apple___Black_rot', 'Apple___healthy', ...]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item() * 100
        label = class_names[predicted.item()]
    return f"{label} ({confidence:.2f}%)"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌿 Plant Disease Classifier",
    description="Upload a leaf image to detect its disease (or check if it's healthy).",
)

if __name__ == "__main__":
    interface.launch()
