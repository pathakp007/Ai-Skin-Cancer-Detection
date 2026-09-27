import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================
# PATHS
# ==========================

BASE_DIR = "dataset"

METADATA_FILE = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

IMAGE_DIR1 = os.path.join(BASE_DIR, "HAM10000_images_part_1")
IMAGE_DIR2 = os.path.join(BASE_DIR, "HAM10000_images_part_2")

OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset")

# ==========================
# READ CSV
# ==========================

df = pd.read_csv(METADATA_FILE)

print("Total Images:", len(df))

# ==========================
# DISEASE LABELS
# ==========================

label_names = {
    "akiec": "Actinic_Keratoses",
    "bcc": "Basal_Cell_Carcinoma",
    "bkl": "Benign_Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic_Nevi",
    "vasc": "Vascular_Lesion"
}

# Replace short labels with full names
df["dx"] = df["dx"].map(label_names)

# ==========================
# CREATE FOLDERS
# ==========================

splits = ["train", "valid", "test"]

for split in splits:
    for disease in label_names.values():
        folder = os.path.join(OUTPUT_DIR, split, disease)
        os.makedirs(folder, exist_ok=True)

print("Folders Created Successfully")

# ==========================
# TRAIN / VALID / TEST SPLIT
# ==========================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["dx"]
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["dx"]
)

print("Train:", len(train_df))
print("Validation:", len(valid_df))
print("Test:", len(test_df))

# ==========================
# COPY FUNCTION
# ==========================

def copy_images(dataframe, split):

    for _, row in dataframe.iterrows():

        image_id = row["image_id"] + ".jpg"
        disease = row["dx"]

        src1 = os.path.join(IMAGE_DIR1, image_id)
        src2 = os.path.join(IMAGE_DIR2, image_id)

        if os.path.exists(src1):
            source = src1
        elif os.path.exists(src2):
            source = src2
        else:
            print(f"Image not found: {image_id}")
            continue

        destination = os.path.join(
            OUTPUT_DIR,
            split,
            disease,
            image_id
        )

        shutil.copy2(source, destination)

# ==========================
# COPY DATA
# ==========================

print("\nCopying Training Images...")
copy_images(train_df, "train")

print("Copying Validation Images...")
copy_images(valid_df, "valid")

print("Copying Testing Images...")
copy_images(test_df, "test")

print("\nDataset Prepared Successfully!")

print(f"\nSaved inside:\n{OUTPUT_DIR}")