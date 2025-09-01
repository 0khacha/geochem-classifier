# analyze_svm_thresholds.py
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_model_and_analyze(model_path="models/svm_geochem_pipeline.joblib"):
    """
    Charger le modèle et analyser les seuils de décision
    """
    pipeline = joblib.load(model_path)
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    columns = pipeline["columns"]
    label_encoder = pipeline["label_encoder"]
    
    return model, scaler, columns, label_encoder

def analyze_decision_boundaries(model, sample_data, scaler, columns, label_encoder, 
                              feature_indices=[0, 1]):
    """
    Analyser les frontières de décision pour 2 features spécifiques
    """
    # Préparer l'échantillon
    df_sample = pd.DataFrame([sample_data], columns=columns)
    df_sample_scaled = pd.DataFrame(scaler.transform(df_sample), columns=columns)
    
    # Obtenir la prédiction de base
    base_pred = model.predict(df_sample_scaled)[0]
    base_probas = model.predict_proba(df_sample_scaled)[0]
    
    inv_map = {v: k for k, v in label_encoder.items()}
    
    print(f"Échantillon de base: {inv_map[base_pred]}")
    print("Probabilités:")
    for i, prob in enumerate(base_probas):
        print(f"  {inv_map[i]}: {prob:.3f}")
    
    return df_sample_scaled, inv_map

def find_minerai_thresholds(model, sample_data, scaler, columns, label_encoder, 
                           target_feature_idx=0, variation_range=(-3, 3), steps=100):
    """
    Trouve à partir de quelle valeur une feature fait basculer vers 'Minerai'
    """
    df_sample = pd.DataFrame([sample_data], columns=columns)
    df_sample_scaled = pd.DataFrame(scaler.transform(df_sample), columns=columns)
    
    inv_map = {v: k for k, v in label_encoder.items()}
    minerai_class = None
    for k, v in label_encoder.items():
        if k.lower() == 'minerai':
            minerai_class = v
            break
    
    if minerai_class is None:
        print("Classe 'Minerai' non trouvée!")
        return None
    
    feature_name = columns[target_feature_idx]
    original_value = df_sample_scaled.iloc[0, target_feature_idx]
    
    # Tester différentes valeurs
    test_values = np.linspace(variation_range[0], variation_range[1], steps)
    results = []
    
    for test_val in test_values:
        # Copier l'échantillon et modifier la feature
        test_sample = df_sample_scaled.copy()
        test_sample.iloc[0, target_feature_idx] = test_val
        
        pred = model.predict(test_sample)[0]
        probas = model.predict_proba(test_sample)[0]
        
        results.append({
            'feature_value': test_val,
            'predicted_class': inv_map[pred],
            'is_minerai': pred == minerai_class,
            'minerai_proba': probas[minerai_class],
            'all_probas': probas
        })
    
    df_results = pd.DataFrame(results)
    
    # Trouver les seuils
    minerai_samples = df_results[df_results['is_minerai']]
    if not minerai_samples.empty:
        min_threshold = minerai_samples['feature_value'].min()
        max_threshold = minerai_samples['feature_value'].max()
        
        print(f"\n=== ANALYSE FEATURE: {feature_name} ===")
        print(f"Valeur originale (normalisée): {original_value:.3f}")
        print(f"Range testé: {variation_range}")
        print(f"Seuil minimum pour 'Minerai': {min_threshold:.3f}")
        print(f"Seuil maximum pour 'Minerai': {max_threshold:.3f}")
        
        # Probabilité maximale de minerai
        max_minerai_proba = df_results['minerai_proba'].max()
        best_value = df_results.loc[df_results['minerai_proba'].idxmax(), 'feature_value']
        print(f"Meilleure valeur: {best_value:.3f} (proba minerai: {max_minerai_proba:.3f})")
        
        return df_results, min_threshold, max_threshold
    else:
        print(f"\nAucune valeur de {feature_name} ne donne 'Minerai' dans le range testé")
        return df_results, None, None

def analyze_all_features(model, sample_data, scaler, columns, label_encoder):
    """
    Analyser toutes les features importantes
    """
    print("=== ANALYSE COMPLÈTE DES SEUILS ===\n")
    
    # Features géochimiques importantes (indices basés sur votre code)
    important_features = {
        'Cu': 11,      # Cuivre  
        'Pb': 12,      # Plomb
        'Zn': 13,      # Zinc
        'Ag': 14,      # Argent
        'As': 15,      # Arsenic
        'Sb': 16,      # Antimoine
        'Bi': 17,      # Bismuth
        'SiO2': 0,     # Silice
        'Al2O3': 1,    # Alumine
    }
    
    thresholds_summary = {}
    
    for feature_name, idx in important_features.items():
        if idx < len(columns):
            print(f"\n--- Analyse {feature_name} (colonne {idx}) ---")
            results, min_th, max_th = find_minerai_thresholds(
                model, sample_data, scaler, columns, label_encoder,
                target_feature_idx=idx, variation_range=(-3, 3), steps=100
            )
            
            thresholds_summary[feature_name] = {
                'min_threshold': min_th,
                'max_threshold': max_th,
                'column_index': idx,
                'results': results
            }
    
    return thresholds_summary

def convert_normalized_to_raw(normalized_value, feature_idx, scaler):
    """
    Convertir une valeur normalisée en valeur brute originale
    """
    # Créer un vecteur avec des zéros sauf pour la feature d'intérêt
    dummy_vector = np.zeros((1, scaler.n_features_in_))
    dummy_vector[0, feature_idx] = normalized_value
    
    # Inverse transform
    raw_vector = scaler.inverse_transform(dummy_vector)
    return raw_vector[0, feature_idx]

def get_raw_thresholds(thresholds_summary, scaler):
    """
    Convertir les seuils normalisés en valeurs brutes
    """
    print("\n=== SEUILS EN VALEURS BRUTES ===")
    
    for feature_name, data in thresholds_summary.items():
        if data['min_threshold'] is not None:
            idx = data['column_index']
            
            raw_min = convert_normalized_to_raw(data['min_threshold'], idx, scaler)
            raw_max = convert_normalized_to_raw(data['max_threshold'], idx, scaler)
            
            print(f"\n{feature_name}:")
            print(f"  Seuil minimum (brut): {raw_min:.3f}")
            print(f"  Seuil maximum (brut): {raw_max:.3f}")
            print(f"  Condition: {raw_min:.3f} <= {feature_name} <= {raw_max:.3f}")

if __name__ == "__main__":
    # Échantillon de test (votre échantillon 'data')
    sample_data = [
        57.86, 16.46, 7.96, 0.92, 3.39, 3.29, 0.11, 0.74, 0.23, 2.77,
        5.5, 1400, 117, 1287, 0.9, 20, 4, 91, 119, 123, 10, 3, 49, 12,
        43, 68, 65, 1.76, 9, 40, 20, 110, 23, 17, 164, 4.46
    ]
    
    try:
        # Charger le modèle
        model, scaler, columns, label_encoder = load_model_and_analyze()
        
        print("Colonnes du modèle:")
        for i, col in enumerate(columns):
            print(f"  {i}: {col}")
        
        # Analyser l'échantillon de base
        df_scaled, inv_map = analyze_decision_boundaries(
            model, sample_data, scaler, columns, label_encoder
        )
        
        # Analyser toutes les features importantes
        thresholds = analyze_all_features(model, sample_data, scaler, columns, label_encoder)
        
        # Convertir en valeurs brutes
        get_raw_thresholds(thresholds, scaler)
        
        print("\n=== RÈGLES SIMPLIFIÉES ===")
        print("Pour qu'un échantillon soit classé comme 'Minerai', il faut généralement:")
        print("1. Des teneurs élevées en métaux précieux (Au, Ag)")
        print("2. Des teneurs significatives en métaux de base (Cu, Pb, Zn)")
        print("3. Possiblement des indicateurs pathfinder (As, Sb, Bi)")
        
    except FileNotFoundError:
        print("Erreur: Fichier modèle non trouvé!")
        print("Assurez-vous que le chemin 'models/svm_geochem_pipeline.joblib' est correct.")
    except Exception as e:
        print(f"Erreur: {e}")