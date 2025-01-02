import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ai_model = genai.GenerativeModel("gemini-1.5-flash")

@st.cache_resource
def load_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),          # Resize images to 224x224 for EfficientNet
        transforms.RandomHorizontalFlip(),       # Random horizontal flip for augmentation
        transforms.RandomRotation(15),           # Random rotation for augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color jitter
        transforms.ToTensor(),                   # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize for pre-trained models
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genai.configure(api_key=GOOGLE_API_KEY)

    model = timm.create_model('efficientnet_b3', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 10)  # 10 classes
    model = model.to(device)

    model.load_state_dict(torch.load("best_nigerian_soups_classifier.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, transform, device

model, transform, device = load_model()

# Function to classify a single image
def predict_class(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Map the prediction to class names
    class_names = ['Afang', 'Banga', 'Edikaikong',
                   'Efo riro', 'Egusi','Ewedu','Ofe-Nsala','Ogbono',
                   'Oha','Okro']  # Adjust to your class names
    return class_names[predicted.item()]

# prompt generator function
def system_prompt(user_prompt, soup, memory=None):
    prompt = f"""
    You are a professional nigerian culinary expert. You know so much about
    nigerian soups. Your job is to answer every question a user asks about any soup
    in this list ['Afang', 'Banga', 'Edikaikong','Efo riro', 'Egusi','Ewedu',
    'Ofe-Nsala','Ogbono','Oha','Okro']. 
    You must never answer any question outside of this scope. In the event that a user
    ask an unrelated question, just politely say you cannot help with that.
    
    The user has chosen: {soup} soup
    user_prompt: {user_prompt}
    previous_questions: {memory}
    
    final note: if soup is None, tell them to upload a soup first before chatting with
    you. You should always keep your answers clear and straight to the point. Always try
    to itemize your points also for easy understanding.
    """
    return prompt

st.title("Nigerian Soup Bot")

   # Sidebar for Image Upload
st.sidebar.title("Upload an Image")
uploaded_file = st.sidebar.file_uploader("choose an image...", type=["jpg", "jpeg", "png"])

    # Initialize variables
image = None
predicted_class = None
memory = []
    
if uploaded_file is not None:
    # Process and display the uploaded image
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image")
    file_extension = Path(uploaded_file.name).suffix
    standardized_name = f"test{file_extension}"
    with open(standardized_name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    predicted_class = predict_class(standardized_name)
   
     # Predict class
if st.sidebar.button(label='Analyze this Soup', type = 'secondary'):
    st.sidebar.success(f'Looks like this is {predicted_class} soup')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Hey Foodie, What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = ai_model.generate_content(system_prompt(prompt, soup=predicted_class, memory=memory))
    response = response.text
    memory.append(response)
    #response = predicted_class
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    

