# ♻️ Smart Waste Classifier

An AI-powered web application that classifies uploaded waste images into different waste categories using a Convolutional Neural Network (CNN). The application helps users identify the correct waste category, promoting efficient waste segregation and environmental sustainability.

## 🚀 Features

- Upload a waste image through the web interface
- Classifies images into:
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash
- Displays the predicted waste category instantly
- Simple and user-friendly interface
- Deployed online using Render

## 🛠️ Tech Stack

- Python
- Keras
- CNN (Convolutional Neural Network)
- Flask
- HTML
- CSS
- Render

## 📂 Project Structure

```
waste-classifier/
│── app.py
│── model/
│── static/
│── templates/
│── uploads/
│── requirements.txt
│── README.md
```

## ⚙️ Installation

1. Clone the repository

```bash
git clone https://github.com/blessynethala/waste-classifier.git
```

2. Navigate to the project directory

```bash
cd waste-classifier
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

5. Open your browser and visit

```
http://127.0.0.1:5000
```

## 📸 How It Works

1. Upload an image of waste.
2. The trained CNN model processes the image.
3. The application predicts the waste category.
4. The predicted result is displayed on the webpage.




