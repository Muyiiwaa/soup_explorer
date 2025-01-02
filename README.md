# Naija-Soup Buddy

## Overview

The **Naija Soup Buddy** is an interactive application designed to identify Nigerian soups from uploaded images and provide insightful information about them. The app combines state-of-the-art image classification using a pre-trained `EfficientNet` model with an AI-powered conversational system to deliver an engaging and educational user experience. Whether you're curious about the ingredients, cooking techniques, or cultural relevance of a Nigerian soup, this bot can handle that!

---

## Features

### 1. **Soup Image Classification**
- Upload an image of a Nigerian soup, and the app will identify it as one of the following:
  - **Afang**
  - **Banga**
  - **Edikaikong**
  - **Efo Riro**
  - **Egusi**
  - **Ewedu**
  - **Ofe Nsala**
  - **Ogbono**
  - **Oha**
  - **Okro**

### 2. **Culinary Knowledge Base**
- The bot can answer questions about Nigerian soups, such as:
  - Ingredients used
  - Nutritional value
  - Preparation methods
  - Regional variations

### 3. **Interactive Chat System**
- Chat with the bot to learn more about the identified soup.
- The system uses Google's `Gemini-1.5` model for intelligent and contextual responses.

### 4. **User-Friendly Interface**
- Streamlit-based application for easy interaction.
- Sidebar for image upload and soup analysis.
- Responsive and clean design for an enhanced user experience.

---

## How It Works

### 1. **Image Upload**
- Upload a `.jpg`, `.jpeg`, or `.png` image of a Nigerian soup via the sidebar.

### 2. **Image Classification**
- The uploaded image is processed using a pre-trained `EfficientNet` model fine-tuned on 10 Nigerian soup categories.
- The app displays the predicted soup class.

### 3. **Conversational Interaction**
- Start a chat with the bot to ask questions about the identified soup.
- The bot uses contextual memory to provide coherent and tailored answers.

---

## Technology Stack

### **Backend**
- **PyTorch**: For deep learning and image classification.
- **Timm**: Pre-trained `EfficientNet` model for transfer learning.

### **Frontend**
- **Streamlit**: For the user interface and chat functionality.

### **Generative AI**
- **Google Generative AI (Gemini-1.5)**: For intelligent, conversational responses.

### **Image Processing**
- **Pillow**: For image loading and preprocessing.
- **Torchvision**: For image transformations.

### **Environment Configuration**
- `.env` file for securely storing API keys.

---

## Setup and Installation

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/muyiiwaa/soup-classifier.git
cd soup-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run Command:
```bash
streamlit run main.py
```
