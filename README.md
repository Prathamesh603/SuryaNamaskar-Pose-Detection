# SuryaNamaskar-Pose-Detection
This project uses a fine-tuned MobileNetV2 model to detect and classify 8 different Surya Namaskar poses from a video. The dataset consists of images categorized by pose names, and the model predicts the correct pose in real time

## 📌 Overview

Surya Namaskar (Sun Salutation) is a sequence of yoga poses performed in a flow. This project aims to develop a deep learning-based model that accurately classifies and detects various Surya Namaskar poses using image and video input. The model is built using **MobileNetV2** and fine-tuned to improve accuracy. However, the model currently experiences fluctuations in certain cases, which will require further improvements.

## 📁 Project Structure

```plaintext
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
│── testing_results/               # Results, screenshots, and accuracy graphs from testing
│── README.md                      # Project documentation
```

## 📥 Dataset

This project uses a **Surya Namaskar Yoga Pose Dataset** downloaded from **Roboflow**. The dataset contains images categorized into **8 different yoga poses**.

🔗 **Dataset Link:** [Roboflow Dataset](https://roboflow.com/)

The dataset has been preprocessed and organized into separate folders per pose.

## 🛠 Requirements

Make sure you have the following dependencies installed before running the scripts:

```bash
pip install tensorflow keras opencv-python mediapipe numpy pandas matplotlib scikit-learn
```

## 🚀 Usage

### 1️⃣ Data Preprocessing

Run these scripts to prepare the dataset before training:

```bash
python data_preprocessing/label_generator.py  # Generates labeled CSV
python data_preprocessing/dataset_organizer.py  # Organizes dataset into folders
```

### 2️⃣ Model Training

Train the model using:

```bash
python model_training/train_model.py  # Train MobileNetV2
python model_training/fine_tune_model.py  # Fine-tune MobileNetV2
```

### 3️⃣ Model Testing

To test the trained model on a **video input**:

```bash
python model_testing/test_pose_from_video.py
```

This will process the video and predict the yoga poses frame by frame.

## 🔬 Testing Results

The **testing\_results/** folder contains:

- Sample output images with pose labels
- Screenshots of pose detection from video
- Accuracy graphs comparing basic and fine-tuned models

## 📌 Models

Trained models are stored in the **models/** folder:

- `surya_namaskar_pose_model.h5` (Basic MobileNetV2 model)
- `fine_tuned_mobilenetv2.h5` (Fine-tuned version for better accuracy)

## 🔄 Improvements for Better Accuracy

The current model exhibits some fluctuations in predictions. Here are potential improvements:

- **Data Augmentation**: Introduce rotation, flipping, and brightness adjustments to improve model robustness.
- **More Training Data**: Expanding the dataset with diverse examples can improve generalization.
- **Better Feature Extraction**: Using advanced architectures like EfficientNet or Vision Transformers (ViTs).

## 📌 Contribution

Feel free to contribute to this project! If you find any improvements, submit a pull request. 🚀

## 📜 License

This project is open-source and available under the **MIT License**.



### 🔹 Features
✔️ Loads the trained MobileNetV2 model
✔️ Uses OpenCV to process video frames
✔️ Predicts poses and overlays labels on the video
✔️ Groups multiple frames for improved accuracy


