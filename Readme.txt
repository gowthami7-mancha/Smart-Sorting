# 🧠 Smart Sorting: Healthy vs Rotten Classifier

This project uses **deep learning and transfer learning** to classify images of fruits and vegetables as either **Fresh (Healthy)** or **Rotten**.  
It leverages a pre-trained **MobileNetV2** model for accurate, lightweight classification — suitable for web or mobile deployment.

---

## 📌 Features

- Image classification using **MobileNetV2**
- Transfer learning on a custom dataset
- Web deployment ready (Flask-compatible)
- Easily extendable to multi-class fruit/vegetable sorting

---

## 📁 Project Structure
SmartSortingProject/
├── Smart_Sorting_Healthy_vs_Rotten.ipynb # Main notebook
├── healthy_vs_rotten.h5 # Trained model
├── SmartSortingDataset/ # Image dataset
│ ├── train/
│ │ ├── fresh/
│ │ └── rotten/
│ └── test/
│ ├── fresh/
│ └── rotten/
├── app.py # (Optional) Flask web app
├── templates/
│ └── index.html # Web UI template
├── static/uploads/ # Image uploads
└── README.md # Project documentatio

