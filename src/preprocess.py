# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import os

def load_and_clean(file_path="dataset/raw/BDD_ICP.xlsx", balance=True, save=True):
    """
    Charger, nettoyer et équilibrer le dataset géochimique.
    
    Args:
        file_path (str): chemin vers le fichier Excel
        balance (bool): si True, applique oversampling pour équilibrer les classes
        save (bool): si True, sauvegarde le dataset traité
        
    Returns:
        tuple: (DataFrame cleaned, StandardScaler fitted)
    """
    # Charger le fichier Excel
    df = pd.read_excel(file_path)
    
    # Remplacer les virgules par des points
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    
    # Fonction pour gérer "<value"
    def convert_threshold(value):
        if isinstance(value, str) and value.startswith("<"):
            try:
                return float(value[1:]) / 2
            except:
                return 0
        try:
            return float(value)
        except:
            return 0
    
    # Conversion des colonnes numériques
    for col in df.columns:
        if col != "Type":
            df[col] = df[col].apply(convert_threshold)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Encodage de la colonne "Type"
    type_mapping = {"Sterile": 0, "Potentiel": 1, "Minerai": 2}
    df["Type"] = df["Type"].map(type_mapping)
    
    # Séparer features et target AVANT normalisation
    X = df.drop("Type", axis=1)
    y = df["Type"]
    
    # Normalisation des features numériques
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns,
        index=X.index
    )
    
    # Reconstruire le dataframe
    df_scaled = X_scaled.copy()
    df_scaled["Type"] = y
    
    # --- Équilibrage du dataset ---
    if balance:
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_scaled, y)
        
        df_scaled = pd.DataFrame(X_res, columns=X_scaled.columns)
        df_scaled["Type"] = y_res
    
    # --- Sauvegarde ---
    if save:
        output_dir = "dataset/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "BDD_ICP_balanced.csv")
        df_scaled.to_csv(output_file, index=False)
        print(f"✅ Dataset sauvegardé dans {output_file}")
    
    # --- Affichage ---
    print("\nDistribution des classes :")
    print(df_scaled["Type"].value_counts())
    print("\nAperçu du dataset :")
    print(df_scaled.head())
    
    return df_scaled, scaler

if __name__ == "__main__":
    df_balanced, scaler = load_and_clean("dataset/raw/BDD_ICP.xlsx")