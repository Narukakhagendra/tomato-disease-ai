import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model.load_state_dict(torch.load("tomato_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Class labels
classes = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Mosaic_virus",
    "Tomato_Healthy"
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Disease advisory dictionary
advisory = {
    "Tomato_Bacterial_spot": "Cause: Bacterial infection.\nPrevention: Use disease-free seeds.\nTreatment: Copper-based sprays.",
    "Tomato_Early_blight": "Cause: Fungal infection.\nPrevention: Crop rotation.\nTreatment: Fungicide spray.",
    "Tomato_Late_blight": "Cause: Water mold pathogen.\nPrevention: Avoid wet leaves.\nTreatment: Fungicide treatment.",
    "Tomato_Leaf_Mold": "Cause: High humidity fungus.\nPrevention: Improve air circulation.\nTreatment: Fungicide spray.",
    "Tomato_Septoria_leaf_spot": "Cause: Fungus Septoria lycopersici.\nPrevention: Remove infected leaves.\nTreatment: Copper fungicide.",
    "Tomato_Spider_mites": "Cause: Tiny mites feeding on leaves.\nPrevention: Maintain humidity.\nTreatment: Neem oil spray.",
    "Tomato_Target_Spot": "Cause: Fungal disease.\nPrevention: Proper plant spacing.\nTreatment: Fungicide application.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Cause: Whitefly transmitted virus.\nPrevention: Control whiteflies.\nTreatment: Remove infected plants.",
    "Tomato_Mosaic_virus": "Cause: Viral infection.\nPrevention: Avoid contaminated tools.\nTreatment: Remove infected plants.",
    "Tomato_Healthy": "Plant is healthy. No treatment required."
}

# Prediction function
def predict_disease(image):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    disease = classes[pred.item()]
    advice = advisory[disease]

    return f"Predicted Disease: {disease}\n\n{advice}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🍅 Tomato Plant Disease Detection AI",
    description="Upload a tomato leaf image to detect disease and get treatment advice."
)

# Launch App
interface.launch()