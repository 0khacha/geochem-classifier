# train.py
# la methode Randomforrest
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocess import load_and_clean


def analyze_model(model, X_test, y_test, save_path="results/"):
    """
    Analyse du modèle entraîné : scores, importance des variables, matrice de confusion
    """
    os.makedirs(save_path, exist_ok=True)

    # --- Prédictions ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # --- Scores globaux ---
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovo")
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\nRésultats d'évaluation :")
    print(" Accuracy :", acc)
    print(" Macro F1 :", f1)
    print(" ROC AUC  :", auc)

    # --- Rapport détaillé ---
    print("\nClassification report :\n", classification_report(y_test, y_pred))

    # --- Importance des variables ---
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 10 variables importantes :\n", feature_importance.head(10))

    # Sauvegarde CSV
    feature_importance.to_csv(os.path.join(save_path, "feature_importance.csv"), index=False)

    # Plot et sauvegarde
    feature_importance.head(15).plot(kind="barh", x="Feature", y="Importance", figsize=(8,6))
    plt.title("Top 15 variables importantes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "feature_importance.png"))
    plt.close()

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Matrice de confusion")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()

    print(f"\nRésultats sauvegardés dans le dossier : {save_path}")


def train_model(use_raw_data=True, save_path="models/geochem_pipeline.joblib"):
    """
    Entraîner un RandomForest sur les données géochimiques.
    
    Args:
        use_raw_data (bool): Si True, charge et préprocesse les données raw
        save_path (str): Chemin pour sauvegarder le modèle
    """
    
    if use_raw_data:
        # Charger et préprocesser depuis les données raw
        df, scaler = load_and_clean("dataset/raw/BDD_ICP.xlsx", balance=True, save=True)
    else:
        # Charger depuis données déjà preprocessées (non recommandé)
        df = pd.read_csv("dataset/processed/BDD_ICP_balanced.csv")
        scaler = None  # Pas de scaler disponible
        print("ATTENTION: Chargement sans scaler - l'inférence ne fonctionnera pas correctement")
    
    print("Données chargées pour entraînement :", df.shape)
    
    X = df.drop("Type", axis=1)
    y = df["Type"]
    
    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # --- Model & GridSearch ---
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    # --- Evaluate ---
    y_pred = grid.predict(X_test)
    print("\n Best Params:", grid.best_params_)
    print(" Macro F1 score:", f1_score(y_test, y_pred, average="macro"))
    print("\nClassification report :\n", classification_report(y_test, y_pred))
    print(" Confusion matrix :\n", confusion_matrix(y_test, y_pred))
    
    # --- Save model with scaler ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    pipeline_data = {
        "model": grid.best_estimator_,
        "scaler": scaler,  # CRUCIAL: Sauvegarder le scaler
        "columns": X.columns.tolist(),
        "label_encoder": {"Sterile": 0, "Potentiel": 1, "Minerai": 2}
    }
    
    joblib.dump(pipeline_data, save_path)
    print(f"\n Modèle et scaler sauvegardés dans {save_path}") 
    
    # --- Analyse ---
    analyze_model(grid.best_estimator_, X_test, y_test)

    return grid.best_estimator_, scaler


if __name__ == "__main__":
    model, scaler = train_model(use_raw_data=True)
# #train.py
# #la methode SVM
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
# import joblib
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from src.preprocess import load_and_clean
# import numpy as np
# import time 


# def analyze_model(model, X_test, y_test, save_path="results/"):
#     """
#     Analyse du modèle SVM entraîné : scores, coefficients (si linéaire), matrice de confusion
#     """
#     os.makedirs(save_path, exist_ok=True)

#     # --- Prédictions ---
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)

#     # --- Scores globaux ---
#     acc = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_prob, multi_class="ovo")
#     f1 = f1_score(y_test, y_pred, average="macro")

#     print("\n Résultats d'évaluation :")
#     print(" Accuracy :", acc)
#     print(" Macro F1 :", f1)
#     print(" ROC AUC  :", auc)

#     # --- Rapport détaillé ---
#     print("\nClassification report :\n", classification_report(y_test, y_pred))

#     # --- Importance des variables (pour kernel linéaire) ---
#     if model.kernel == "linear":
#         coef = np.abs(model.coef_).sum(axis=0)
#         feature_importance = pd.DataFrame({
#             "Feature": X_test.columns,
#             "Importance": coef
#         }).sort_values(by="Importance", ascending=False)
#         print("\nTop 10 variables importantes :\n", feature_importance.head(10))

#         # Sauvegarde CSV
#         feature_importance.to_csv(os.path.join(save_path, "svm_feature_importance.csv"), index=False)

#         # Plot et sauvegarde
#         feature_importance.head(10).plot(kind="barh", x="Feature", y="Importance", figsize=(8,6))
#         plt.title("Top 10 variables importantes (coef SVM)")
#         plt.gca().invert_yaxis()
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_path, "svm_feature_importance.png"))
#         plt.close()
#     else:
#         print("\n Feature importance non disponible pour kernel non-linéaire")

#     # --- Matrice de confusion ---
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
#     disp.plot(cmap="Blues")
#     plt.title("Matrice de confusion")
#     plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
#     plt.close()

#     print(f"\n Résultats sauvegardés dans le dossier : {save_path}")


# def train_model(use_raw_data=True, save_path="models/svm_geochem_pipeline.joblib"):
#     """
#     Entraîner un SVM sur les données géochimiques.
#     """
    
#     if use_raw_data:
#         df, scaler = load_and_clean("dataset/raw/BDD_ICP.xlsx", balance=True, save=True)
#     else:
#         df = pd.read_csv("dataset/processed/BDD_ICP_balanced.csv")
#         scaler = None
#         print("ATTENTION: Chargement sans scaler - l'inférence risque de mal fonctionner")
    
#     print("Données chargées pour entraînement :", df.shape)
    
#     X = df.drop("Type", axis=1)
#     y = df["Type"]
    
#     # --- Split ---
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )
    
#     # --- Model & GridSearch ---
#     svm = SVC(class_weight="balanced", probability=True, random_state=42)
#     param_grid = {
#         "C": [0.1, 1, 10],
#         "kernel": ["linear", "rbf", "poly"],
#         "gamma": ["scale", "auto"]
#     }
    
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     grid = GridSearchCV(
#         svm, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1
#     )
#     grid.fit(X_train, y_train)
#     # --- Mesure du temps d'entraînement ---
#     start_time = time.time()
#     grid.fit(X_train, y_train)
#     end_time = time.time()
#     training_time = end_time - start_time
#     print(f"\nTemps d'entraînement SVM : {training_time:.2f} secondes")

#     # --- Affichage dans les résultats globaux ---
#     print(f"\n--- Résumé des performances ---")

#     # --- Évaluation ---
#     y_pred = grid.predict(X_test)
#     print("\n Best Params:", grid.best_params_)
#     print(" Macro F1 score:", f1_score(y_test, y_pred, average="macro"))
#     print("\nClassification report :\n", classification_report(y_test, y_pred))
#     print(" Confusion matrix :\n", confusion_matrix(y_test, y_pred))


    
#     # --- Save model with scaler ---
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     pipeline_data = {
#         "model": grid.best_estimator_,
#         "scaler": scaler,
#         "columns": X.columns.tolist(),
#         "label_encoder": {"Sterile": 0, "Potentiel": 1, "Minerai": 2}
#     }
    
#     joblib.dump(pipeline_data, save_path)
#     print(f"\nModèle et scaler sauvegardés dans {save_path}") 
    
#     # --- Analyse ---
#     analyze_model(grid.best_estimator_, X_test, y_test)

#     return grid.best_estimator_, scaler


# if __name__ == "__main__":
#     model, scaler = train_model(use_raw_data=True)

