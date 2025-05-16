import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision import models
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set Web App title
st.set_page_config(page_title="Rice Seeds Classification", layout='centered', initial_sidebar_state="expanded")

# Load the model
def load_model():
    model_path = './Fyp_ResnetmodelAdamW1.pth'  # Update this path if necessary
    state_dict = torch.load(model_path, map_location=device)
    
    # Initialize model with the correct architecture
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify the model for binary classification
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 384),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(384, 1)  # Single output for binary classification
    )
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

# Class names
rice_name = ['Cultivated', 'Weedy']

# Load model
model = load_model()

# Define a function to compute metrics
def compute_metrics(loader, model):
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return accuracy, precision, recall, f1

# Classify function
def classify(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        conf_score = torch.sigmoid(outputs).item()  # Confidence score for Weedy class
        predicted = int(conf_score > 0.5)  # Class (0 or 1)
    
    # Calculate confidence scores for both classes
    if predicted == 0:  # Predicted class is Cultivated
        cultivated_score = 1 - conf_score  # Confidence for Cultivated
        weedy_score = conf_score  # Confidence for Weedy
    else:  # Predicted class is Weedy
        cultivated_score = 1 - conf_score  # Confidence for Cultivated
        weedy_score = conf_score  # Confidence for Weedy
    
    return rice_name[predicted], conf_score, cultivated_score, weedy_score

# Define pages
def page_home():
    st.title("About This App")
    st.write("""
    This application classifies rice seeds into **Cultivated** or **Weedy** based on uploaded images.
    The model used here is ResNet50, a pre-trained deep learning model fine-tuned for binary classification.
    """)

def page_classification():
    st.title("Rice Seeds Classification")
    file = st.file_uploader("Upload an image of a rice crop...", type=['jpeg', 'jpg', 'png'])
    
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        
        # Classify the image and get results
        rice_type, conf_score, cultivated_score, weedy_score = classify(image, model)
        
        # Display prediction and confidence score
        st.write(f"## Prediction: {rice_type}")
       # If the predicted class is Cultivated, show only the cultivated score
        if rice_type == "Cultivated":
            st.write(f"### Cultivated Confidence Score: {cultivated_score:.2f}")
            
        
        # If the predicted class is Weedy, show only the weedy score
        else:
            st.write(f"### Weedy Confidence Score: {weedy_score:.2f}")
            
        # Visualize confidence scores for both classes (Cultivated, Weedy)
        st.subheader("Confidence Score Visualization")
        
        # Use seaborn style for the chart
        sns.set(style="whitegrid")
        
        # Create the bar chart with more stylish visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Cultivated", "Weedy"], [cultivated_score, weedy_score], color=['#4C72B0', '#F28E2B'], width=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Confidence", fontsize=12)
        ax.set_title(f"Model Confidence for {rice_type}", fontsize=14)
        
        # Annotate the bar chart with the score values
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), textcoords='offset points')
        
        # Display the plot
        st.pyplot(fig)
        
        # Description of the prediction
        st.subheader("Explanation of Prediction")
        
        if rice_type == "Weedy":
            st.write(""" 
            The model predicted **Weedy** because the seed image likely has irregular patterns or a rough texture, which is typical for weeds.
            Weeds often exhibit certain features that distinguish them from cultivated rice, such as:
            - **Irregular shapes**: Weeds often have less defined, jagged edges or asymmetry compared to cultivated rice.
            - **Different color patterns**: Weedy seeds may have varying shades or color contrasts, often more erratic than cultivated rice.
            - **Texture differences**: Weeds may have a rougher or uneven texture compared to the smoother appearance of cultivated rice seeds.
            This combination of characteristics leads the model to classify it as a **Weedy** seed.
            """)
        else:
            st.write(""" 
            The model predicted **Cultivated** because the seed image shows characteristics commonly found in cultivated rice:
            - **Smooth, regular shape**: Cultivated rice seeds typically have a more uniform, smooth shape.
            - **Consistent color**: The seeds are generally more uniform in color with less variation compared to weedy seeds.
            - **Refined texture**: Cultivated rice seeds are usually more polished and have a more consistent texture than their weedy counterparts.
            This led the model to confidently classify the seed as **Cultivated**.
            """)


# Main function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction"])
    
    if page == "Home":
        page_home()
    elif page == "Prediction":
        page_classification()

if __name__ == "__main__":
    main()