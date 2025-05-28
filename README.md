# 🧠 Digit Recognition Web App

A Flask-based web app for recognizing handwritten digits using a CNN trained on the MNIST dataset.

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
