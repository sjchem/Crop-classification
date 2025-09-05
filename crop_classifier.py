import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import joblib
import io
import streamlit as st  # Added missing import

model_data = joblib.load('crop_classification_model.pkl')

# Create the same model architecture that was used during training
# The error shows the model was trained with ResNet18 (512 features), not ResNet50 (2048 features)
model = models.resnet18(pretrained=False)  # Keeping ResNet18 which has 512 features in final layer

# Check if the model was created with a specific weights version
if 'model_architecture' in model_data:
    # If architecture info is saved, use it to recreate the exact model
    model = model_data['model_architecture']
else:
    # If no architecture info, modify the model to match the saved state dict
    # Fix: Use 512 instead of 2048 as that's the correct size for ResNet18
    num_classes = len(model_data['class_to_idx'])
    model.fc = torch.nn.Linear(512, num_classes)  # Changed from 2048 to 512 for ResNet18

# Load the state dictionary with strict=False to ignore non-matching keys
model.load_state_dict(model_data['model_state_dict'], strict=False)
model.eval()

idx_to_class = {v: k for k, v in model_data['class_to_idx'].items()}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
st.title("Crop Classification")
st.markdown("Upload an image of a crop to predict it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) 

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_name = idx_to_class[predicted.item()]
    
    st.success(f"Predicted Crop: **{class_name}**")