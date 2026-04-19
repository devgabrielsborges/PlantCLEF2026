import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()  # carrega variáveis de ambiente do .env


def main():
    data_dir = os.environ.get("DATA_DIR", "../data")
    metadata_path = os.path.join(
        data_dir, "PlantCLEF2024_single_plant_training_metadata.csv"
    )
    out_val_gt_path = os.path.join(data_dir, "validation_ground_truth.csv")
    out_val_composition_path = os.path.join(data_dir, "val_quadrat_composition.csv")
    out_val_metadata_path = os.path.join(data_dir, "val_metadata_split.csv")
    out_train_path = os.path.join(data_dir, "train_metadata_split.csv")

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return

    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path, sep=";", dtype={"partner": str})

    possible_id_cols = ["image_name", "image_path", "id", "image_id"]
    id_col = next((col for col in possible_id_cols if col in df.columns), df.columns[0])
    print(f"Using '{id_col}' as identifier column.")

    # Stratified split: handle species with only 1 sample
    species_counts = df["species_id"].value_counts()
    single_sample_species = species_counts[species_counts < 2].index

    if not single_sample_species.empty:
        print(
            f"Detected {len(single_sample_species)} species with only one sample. "
            "Assigning them to train set before stratified split."
        )
        df_single = df[df["species_id"].isin(single_sample_species)]
        df_multi = df[~df["species_id"].isin(single_sample_species)]

        # Stratified split for species with more than 1 sample
        # We need to check if some species have very few samples (e.g., 2 or 3)
        # where test_size=0.1 would result in 0 samples for validation if stratified.
        # sklearn's train_test_split handles this but requires at least 1 member for each class to stratify if possible?
        # Actually, it requires at least 2 members to have at least one in each split if we are unlucky?
        # No, it just needs at least one member to perform stratification if test_size is very small.
        # But for 0.1, it needs at least 2 to potentially put one in val.
        train_multi, val_multi = train_test_split(
            df_multi, test_size=0.1, random_state=42, stratify=df_multi["species_id"]
        )

        train_df = pd.concat([train_multi, df_single])
        val_df = val_multi
    else:
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=df["species_id"]
        )

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    train_df.to_csv(out_train_path, sep=";", index=False)
    val_df.to_csv(out_val_metadata_path, sep=";", index=False)
    print(f"Saved train split to {out_train_path}")
    print(f"Saved val split to {out_val_metadata_path}")

    # --- gera quadrats sintéticos ---
    print("Generating synthetic quadrats...")

    rng = np.random.default_rng(42)
    N_QUADRATS = 500
    MIN_SPECIES = 2
    MAX_SPECIES = 10

    # agrupa imagens de validação por espécie para facilitar amostragem
    species_to_images = val_df.groupby("species_id")[id_col].apply(list).to_dict()
    all_species = list(species_to_images.keys())

    quadrat_gt_rows = []  # vai virar validation_ground_truth.csv
    quadrat_comp_rows = []  # vai virar val_quadrat_composition.csv
    IMAGE_DIR = os.path.join(data_dir, "images_max_side_800")

    for i in range(N_QUADRATS):
        quadrat_id = f"synthetic_quadrat_{i:04d}"
        n_species = int(rng.integers(MIN_SPECIES, MAX_SPECIES + 1))

        # amostra espécies aleatórias
        sampled_species = rng.choice(
            all_species, size=n_species, replace=False
        ).tolist()

        # para cada espécie, sorteia UMA imagem representativa
        for sp in sampled_species:
            images_for_sp = species_to_images[sp]
            chosen_image = rng.choice(images_for_sp)
            quadrat_comp_rows.append(
                {
                    "quadrat_id": quadrat_id,
                    "image_path": os.path.join(
                        IMAGE_DIR, str(sp), chosen_image
                    ),  # caminho/nome da imagem individual
                    "species_id": sp,
                }
            )

        # ground truth do quadrat = lista de todas as espécies amostradas
        quadrat_gt_rows.append(
            {
                "quadrat_id": quadrat_id,
                "species_ids": str(sampled_species),
            }
        )

    df_gt = pd.DataFrame(quadrat_gt_rows)
    df_comp = pd.DataFrame(quadrat_comp_rows)

    df_gt.to_csv(out_val_gt_path, index=False)
    df_comp.to_csv(out_val_composition_path, index=False)

    print(f"Saved {N_QUADRATS} synthetic quadrats to {out_val_gt_path}")
    print(f"Saved composition map to {out_val_composition_path}")


if __name__ == "__main__":
    main()
