import os
import pandas as pd

# Paths
image_folder = "D:\\ET23BTCO123\\suraya namskar pose detection\\dataset\\label"  # Update with your actual image folder path
csv_path = "C:\\Users\\prath\\Downloads\\surya namaskar.v4i.multiclass\\train\\_classes.csv"  # Update with the correct CSV path

# Load CSV
df = pd.read_csv(csv_path)
image_filenames = set(df["filename"])  # Get filenames from CSV
existing_images = set(os.listdir(image_folder))  # Get actual images

# Check missing files
missing_files = image_filenames - existing_images
print(f"Missing images: {len(missing_files)}")
# Convert one-hot encoding to a single label
df["label"] = df.iloc[:, 1:].idxmax(axis=1)  # Find column with value 1
df = df[["filename", "label"]]  # Keep only filename & label
df.to_csv("labeled_data.csv", index=False)  # Save new CSV