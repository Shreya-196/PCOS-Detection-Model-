# PCOD_Detection
📌 Overview

This project aims to detect Polycystic Ovary Syndrome (PCOS) using a hybrid Machine Learning + Deep Learning approach. It integrates clinical data and ultrasound images to provide an accurate diagnosis.

🚀 Features

Hybrid Model: Combines a CNN for ultrasound image classification and an ANN for clinical data analysis.

Dual Input Support: Works with or without ultrasound images.

High Accuracy: Achieves 93% accuracy based on training data.

Web-Based Interface: Users can upload data/images via a web app.

Deployed Online: Hosted on Vercel/Heroku for easy access.

🔧 Installation

1️⃣ Clone the Repository

git clone https://github.com/Shreya-196/PCOS-Detection-Model.git
cd PCOS-Detection-Model

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Application

python app.py

Open http://localhost:5000 in your browser.

📊 Model Details

CNN (Convolutional Neural Network): Used for ultrasound image classification.

ANN (Artificial Neural Network): Analyzes clinical data like age, BMI, hormone levels, and symptoms.

Final Decision: Merges predictions from both models for a comprehensive diagnosis.

🚀 Deployment

The project is deployed using Vercel/Heroku.

GUI:
![image](https://github.com/user-attachments/assets/e0bd45c1-a330-4140-9af7-66d4bbe7ff5d)

