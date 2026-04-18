import pandas as pd


def load_metadata(data_dir: str):
    df_species_ids = pd.read_csv(f"{data_dir}/species_ids.csv")

    df_metadata = pd.read_csv(
        f"{data_dir}/PlantCLEF2024_single_plant_training_metadata.csv",
        sep=";",
        dtype={"partner": str},
    )
    class_map = df_species_ids["species_id"].to_dict()

    return df_species_ids, df_metadata, class_map
