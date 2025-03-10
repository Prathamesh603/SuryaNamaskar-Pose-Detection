# SuryaNamaskar-Pose-Detection
This project uses a fine-tuned MobileNetV2 model to detect and classify 8 different Surya Namaskar poses from a video. The dataset consists of images categorized by pose names, and the model predicts the correct pose in real time

## 📌 Project Overview
This project focuses on **Surya Namaskar (Sun Salutation) Pose Detection** using **Deep Learning (MobileNetV2)**. The system can classify different yoga poses in Surya Namaskar from a video input and predict the correct pose in real-time. The model is trained on a custom dataset containing images of eight different Surya Namaskar poses.

## 📂 Project Structure
The repository is structured as follows:

```
SuryaNamaskar-Pose-Detection/
│── dataset/                      # Data source link (organized images in respective folders)
│── data_preprocessing/            # Scripts for preprocessing dataset
│   │── label_generator.py         # Generates CSV with labels from image folders
│   └── dataset_organizer.py       # Moves images into respective class folders
│── model_training/                # Model training scripts
│   │── train_model.py             # Standard training of MobileNetV2
│   └── fine_tune_model.py         # Fine-tuning of MobileNetV2
│── model_testing/                 # Scripts for testing the trained model
│   └── test_pose_from_video.py    # Detects poses from video input
│── models/                        # Pretrained and fine-tuned models
│   │── surya_namaskar_pose_model.h5
│   └── fine_tuned_mobilenetv2.h5
│── testing_results/               # Results and output images from testing
│── README.md                      # Project documentation
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


