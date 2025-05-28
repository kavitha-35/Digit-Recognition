# 🧠 Digit Recognition Web App

**Digit Recognition Web App** is a user-friendly web application built using Flask that allows users to upload images of handwritten digits (0–9) and receive real-time predictions using a Convolutional Neural Network (CNN) model trained on the MNIST dataset. The project leverages Python, TensorFlow/Keras for model training and inference, and HTML/CSS for a responsive interface. Designed for simplicity and educational purposes, this app demonstrates the integration of deep learning models into a web environment for interactive digit classification.


---

## 🚀 Features

- Upload an image of a digit (0-9)
- Predicts the digit using a trained CNN
- Simple UI using HTML/CSS
- Live preview of uploaded digit and prediction

---

## 🛠️ Tech Stack

- Python
- Flask
- TensorFlow / Keras
- MNIST Dataset
- HTML/CSS

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/kavitha-35/Digit-Recognition.git
cd Digit-Recognition
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
---
## 📁 Project Structure

├── app.py
├── templates/
│   ├── index.html
│   └── prediction.html
├── static/
│   └── style.css
|   └── style1.css
├── model/
│   └── Number_Identification_model.keras
├── README.md
└── .gitignore
