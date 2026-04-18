import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Paths assuming the script is run from the project root
    data_dir = "data"
    metadata_path = os.path.join(
        data_dir, "PlantCLEF2024_single_plant_training_metadata.csv"
    )
    out_val_path = os.path.join(data_dir, "validation_ground_truth.csv")
    out_train_path = os.path.join(data_dir, "train_metadata_split.csv")

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        print("Please ensure the training metadata is downloaded before running.")
        return

    print(f"Loading training metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path, sep=";", dtype={"partner": str})

    # Identify the correct column for the image name/ID
    possible_id_cols = ["image_name", "image_path", "id", "image_id"]
    id_col = next((col for col in possible_id_cols if col in df.columns), df.columns[0])

    print(f"Using '{id_col}' as the identifier column.")
    print("Splitting data into 90% train and 10% validation...")

    try:
        # Stratify ensures each species is proportionally represented in train and val
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df["species_id"]
        )
    except ValueError:
        # Fallback to random split if some rare species have only 1 sample
        print(
            "Stratified split failed due to rare classes. Using standard random split..."
        )
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    print(f"Train size: {len(train_df)} images | Val size: {len(val_df)} images")

    # Save the training split for model training
    train_df.to_csv(out_train_path, sep=";", index=False)
    print(f"Saved new training split metadata to {out_train_path}")

    # Group out validation into the exact format `baseline.ipynb` requires to compute metrics
    print("Formatting validation ground truth...")
    val_gt = pd.DataFrame(
        {
            "quadrat_id": val_df[id_col].apply(
                lambda x: str(os.path.basename(str(x))).split(".")[0]
            ),
            "species_ids": val_df["species_id"].apply(lambda x: f"[{int(x)}]"),
        }
    )

    val_gt.to_csv(out_val_path, index=False)
    print(f"Validation ground truth successfully saved to {out_val_path}")


if __name__ == "__main__":
    main()
