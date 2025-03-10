# SuryaNamaskar-Pose-Detection
This project uses a fine-tuned MobileNetV2 model to detect and classify 8 different Surya Namaskar poses from a video. The dataset consists of images categorized by pose names, and the model predicts the correct pose in real time

## 📌 Project Overview
This project focuses on **Surya Namaskar (Sun Salutation) Pose Detection** using **Deep Learning (MobileNetV2)**. The system can classify different yoga poses in Surya Namaskar from a video input and predict the correct pose in real-time. The model is trained on a custom dataset containing images of eight different Surya Namaskar poses.

## 📂 Project Structure
The repository is structured as follows:

```
SuryaNamaskar-Pose-Detection/
│── dataset/                # Contains dataset link and information
│── data_preprocessing/     # Contains scripts for data labeling & organizing
│── model_training/         # Model training scripts (base & fine-tuned model)
│── model_testing/          # Script for testing pose detection on videos
│── models/                 # Trained models (.h5 files)
|--Testing_Results/          # Testing Results
│── README.md               # Project documentation
```

## 📥 Dataset
The dataset consists of images categorized into eight different Surya Namaskar poses. Each pose contains around **280–320 images**. The dataset was obtained from **Roboflow** and was further processed for training.

🔗 **[Dataset Source (Roboflow)](https://universe.roboflow.com/lalitha-uruu5/surya-namaskar)**

### 🔹 Data Organization
The dataset is organized into folders where each folder represents a specific yoga pose:
```
dataset/
│── pranamasana/
│── hasta_utthanasana/
│── padahastasana/
│── ashwa_sanchalanasana/
│── kumbhakasana/
│── ashtanga_namaskara/
│── bhujangasana/
│── adho_mukh_svanasana/
```

## 🛠 Data Preprocessing
The **data_preprocessing** folder contains scripts to:
1. **Label Images**: Assign correct labels to images based on the CSV file.
2. **Organize Images**: Move images into folders based on their corresponding yoga poses.

### 📜 Scripts
- `data_labeling.py`: Creates labeled CSV file from raw dataset.
- `data_organization.py`: Organizes images into respective pose folders.

## 🏋️‍♂️ Model Training
The **model_training** folder contains scripts for:
1. **Baseline Model**: Trains a MobileNetV2 model on the dataset.
2. **Fine-tuned Model**: Improves accuracy using additional training techniques.

### 📜 Scripts
- `train_model.py`: Trains a basic MobileNetV2 model.
- `fine_tune_model.py`: Fine-tunes MobileNetV2 for better accuracy.

### 📌 Model Used: MobileNetV2
- **Pretrained Model**: MobileNetV2 (ImageNet weights)
- **Input Size**: 224x224
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 20 (adjustable)

## 🎥 Model Testing
The **model_testing** folder contains a script to test the trained model on video input.

### 📜 Script
- `pose_detection.py`: Processes a video and detects Surya Namaskar poses in real-time.

### 🔹 Features
✔️ Loads the trained MobileNetV2 model
✔️ Uses OpenCV to process video frames
✔️ Predicts poses and overlays labels on the video
✔️ Groups multiple frames for improved accuracy

## 🛠 Setup Instructions
Follow these steps to run the project on your local machine:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Prathamesh603/SuryaNamaskar-Pose-Detection.git
cd SuryaNamaskar-Pose-Detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download the Dataset
Download the dataset from **Roboflow** and place it in the `dataset/` folder.

### 4️⃣ Run Data Preprocessing
```bash
python data_preprocessing/data_labeling.py
python data_preprocessing/data_organization.py
```

### 5️⃣ Train the Model
```bash
python model_training/train_model.py   # For base model
python model_training/fine_tune_model.py  # For fine-tuned model
```

### 6️⃣ Test the Model on Video
```bash
python model
