# infer.py
import joblib
import pandas as pd
import numpy as np

def predict_sample(sample_data, model_path="models/geochem_pipeline.joblib"):
    """
    Faire une prédiction sur un nouvel échantillon.
    
    Args:
        sample_data: liste de valeurs RAW (avant normalisation)
        model_path: chemin vers le modèle sauvegardé
    
    Returns:
        str: classe prédite ('Sterile', 'Potentiel', 'Minerai')
    """
    # Charger le pipeline sauvegardé 
    
    pipeline = joblib.load(model_path)
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    columns = pipeline["columns"]
    label_encoder = pipeline["label_encoder"]
    
    # Vérifications
    if scaler is None:
        raise ValueError("Pas de scaler dans le modèle. Réentraîner avec use_raw_data=True")
    
    if len(sample_data) != len(columns):
        raise ValueError(f"Nombre de features attendu: {len(columns)}, fourni: {len(sample_data)}")
    
    df_sample = pd.DataFrame([sample_data], columns=columns)
    
    df_sample_scaled = pd.DataFrame(
        scaler.transform(df_sample), 
        columns=columns
    )
    
    print("Données après normalisation:")
    print(f"Min: {df_sample_scaled.iloc[0].min():.3f}")
    print(f"Max: {df_sample_scaled.iloc[0].max():.3f}")
    print(f"Mean: {df_sample_scaled.iloc[0].mean():.3f}")
    
    # Faire prédiction
    pred = model.predict(df_sample_scaled)[0]
    probabilities = model.predict_proba(df_sample_scaled)[0]
    
    # Inverser l'encodage numérique
    inv_map = {v: k for k, v in label_encoder.items()}
    predicted_class = inv_map[pred]
    
    # Afficher les probabilités pour debug
    print("\n Probabilités par classe:")
    for class_idx, prob in enumerate(probabilities):
        class_name = inv_map[class_idx]
        print(f"  {class_name}: {prob:.3f}")
    
    return predicted_class

def preprocess_raw_sample(sample_data):
    """
    Préprocesser un échantillon raw comme dans preprocess.py
    
    Args:
        sample_data: liste de valeurs potentiellement avec virgules et symboles "<"
    
    Returns:
        liste de float nettoyées
    """
    def convert_threshold(value):
        if isinstance(value, str):
            
            value = value.replace(",", ".")
            
            if value.startswith("<"):
                try:
                    return float(value[1:]) / 2
                except:
                    return 0.0
        try:
            return float(value)
        except:
            return 0.0
    
    return [convert_threshold(val) for val in sample_data]

if __name__ == "__main__":

    min_sample = [
        47.74, 12.99, 8.56, 3.13, 4.15, 6.02, 0.28, 0.60, 0.26, 0.77,
        29488, 16777, 146, 728, 0.2, 20, 631, 114, 76, 2272,
        10, 16, 47, 8, 4, 71, 14311, 5.14, 1420, 40,
        20, 38, 23, 20, 7789, 7.32
    ]
    

    pot_sample = [
        52.99, 17.4, 9.51, 1.38, 4.39, 4.59, 0.16, 0.86, 0.19, 1.36,
        4.5, 2600, 151, 639, 0.1, 20, 4, 14, 176, 53,
        10, 9, 84, 10, 12, 57, 904, 0.35, 17, 20,
        10, 31, 11.5, 13, 336, 6.18
    ]
    
   
    str_sample = [
        60.78, 11.98, 4.57, 3.61, 3.61, 3.17, 0.17, 0.9, 0.32, 2.37,
        3.4, 111, 25, 1507, 0.1, 20, 2, 1, 54, 12,
        10, 3, 15, 6, 13, 17, 191, 0.32, 5, 20,
        10, 87, 11.5, 38, 362, 7.2
    ]
    
    # Liste contenant les valeurs
    data = [
    57.86, 16.46, 7.96, 0.92, 3.39, 3.29, 0.11, 0.74, 0.23, 2.77,
    5.5, 1400, 117, 1287, 0.9, 20, 4, 91, 119, 123, 10, 3, 49, 12,
    43, 68, 65, 1.76, 9, 40, 20, 110, 23, 17, 164, 4.46
    ]


    print("TEST - Échantillon Minerai:")
    pred_min = predict_sample(data)
    print(f"Résultat: {pred_min}\n")
    
    print("TEST - Échantillon Potentiel:")
    pred_pot = predict_sample(pot_sample)
    print(f"Résultat: {pred_pot}\n")
    
    print("TEST - Échantillon Stérile:")
    pred_str = predict_sample(str_sample)
    print(f"Résultat: {pred_str}\n")