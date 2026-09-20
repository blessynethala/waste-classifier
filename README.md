# ♻️ Smart Waste Classifier

An AI-powered web application that classifies uploaded waste images into different waste categories using a Convolutional Neural Network (CNN). The application helps users identify the correct waste category, promoting efficient waste segregation and environmental sustainability.

## 🔗 Live Demo

Try it here: **https://waste-classifier-bvh55tqnbu3syfngoip6zs.streamlit.app/**

> The app is hosted on Streamlit Community Cloud and goes to sleep after a period of inactivity. If you see a "This app has gone to sleep" page, click **"Yes, get this app back up!"** and wait a few seconds.

## 🚀 Features

- Upload a waste image (PNG, JPG, JPEG or WEBP) through the web interface
- Classifies images into six categories:
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash
- Displays the predicted category with a confidence score
- Shows confidence scores for all six classes
- Simple and user-friendly interface

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- Streamlit
- Streamlit Community Cloud (deployment)

> The project was originally built with Flask, HTML and CSS and deployed on Render. It has since been migrated to Streamlit and Streamlit Community Cloud.

## 📂 Project Structure

```
waste-classifier/
│── streamlit_app.py       # Streamlit web app
│── waste_classifier.h5    # Trained CNN model
│── background.jpg         # App background image
│── requirements.txt       # Python dependencies
│── .streamlit/
│   └── config.toml        # Theme settings
│── app.py                 # Original Flask version
│── templates/             # Original Flask HTML templates
│── README.md
```

## ⚙️ Installation

1. Clone the repository

```
git clone https://github.com/blessynethala/waste-classifier.git
```

2. Navigate to the project directory

```
cd waste-classifier
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Run the application

```
streamlit run streamlit_app.py
```

5. Open your browser and visit

```
http://localhost:8501
```

## 📸 How It Works

1. Upload an image of waste.
2. The image is resized to 224x224 and normalised.
3. The trained CNN model predicts probabilities for the six waste classes.
4. The predicted category and confidence scores are displayed on the page.
