import os
import shutil
import pandas as pd

# -------------------------------
# 1. Define Paths (Update These Paths!)
# -------------------------------
image_folder = r"dataset\label"  # Source image folder
csv_path = r"labeled_data.csv"  # CSV file with image names and labels
dataset_folder = r"dataset"  # Destination folder for organized images

# -------------------------------
# 2. Load CSV and Prepare Data
# -------------------------------
print("🔍 Loading CSV file...")
df = pd.read_csv(csv_path)

# Ensure column names match expected format
expected_columns = {"filename", "label"}
if not expected_columns.issubset(df.columns):
    print("Error: CSV does not contain the required columns ('filename' and 'label').")
    print("Found columns:", df.columns)
    exit()

# Remove extra spaces and check for empty values
df["filename"] = df["filename"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

# Debug: Print first few rows to verify data
print("\nCSV Data Sample:\n", df.head())

# -------------------------------
# 3. Create Dataset Folder (if not exists)
# -------------------------------
print("\nChecking dataset folder...")
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder, exist_ok=True)
    print(f"Created main dataset folder: {dataset_folder}")
else:
    print(f"Main dataset folder already exists: {dataset_folder}")

# -------------------------------
# 4. Move Images into Folders Based on Pose Label
# -------------------------------
print("\nOrganizing images into folders...")

for index, row in df.iterrows():
    file_name = row["filename"]
    label = row["label"]

    # Ensure label is valid and not empty
    if not label:
        print(f"Skipping {file_name} due to missing label")
        continue

    # Build source and destination paths
    src = os.path.join(image_folder, file_name)
    dest_label_folder = os.path.join(dataset_folder, label)  # Create separate folder per pose
    dst = os.path.join(dest_label_folder, file_name)

    # Debug: Print folder paths before creating them
    print(f"Checking folder: {dest_label_folder}")

    # Create label folder if it doesn't exist
    if not os.path.exists(dest_label_folder):
        os.makedirs(dest_label_folder, exist_ok=True)
        print(f"Created folder for pose '{label}': {dest_label_folder}")

    # Check if the source image exists before moving
    if os.path.exists(src):
        try:
            shutil.move(src, dst)
            print(f"Moved: {src} --> {dst}")
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")
    else:
        print(f"Missing image: {src}")

print("\nImages organized into folders successfully!")
