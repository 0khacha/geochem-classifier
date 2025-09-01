
# Geochem Classifier 🧪⛏️

Un système de classification géochimique basé sur l'apprentissage automatique pour analyser et prédire les types d'échantillons miniers.

## 📋 Description

Ce projet utilise des algorithmes de machine learning pour classifier automatiquement les échantillons géochimiques en trois catégories :
- **Stérile** : Échantillons sans valeur économique
- **Potentiel** : Échantillons avec potentiel minier modéré
- **Minerai** : Échantillons de haute valeur économique

Le système analyse 36 caractéristiques géochimiques différentes pour effectuer ses prédictions.

## 🚀 Fonctionnalités

- **Classification automatique** d'échantillons géochimiques
- **Analyse de seuils** pour optimiser les critères de classification
- **Préprocessing intelligent** des données brutes avec gestion des valeurs manquantes
- **Analyse comparative** entre différents algorithmes (SVM, Random Forest, etc.)
- **Prédictions en temps réel** sur de nouveaux échantillons
- **Rapports détaillés** avec métriques de performance

## 📁 Structure du projet

```
src/
├── train.py                    # Entraînement des modèles
├── infer.py                    # Prédiction sur nouveaux échantillons
├── preprocess.py               # Préprocessing des données
├── analyze_ag.py               # Analyse spécifique à l'argent
├── analyze_thresholds.py       # Analyse des seuils de classification
├── geochemical_analysis.py     # Analyses géochimiques générales
├── potential_analysis.py       # Analyse des échantillons potentiels
└── silver_analysis.py          # Analyse spécialisée argent
```

## 🛠️ Installation

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Dépendances principales
- `scikit-learn` : Algorithmes de machine learning
- `pandas` : Manipulation des données
- `numpy` : Calculs numériques
- `joblib` : Sérialisation des modèles
- `matplotlib` : Visualisations

## 💻 Utilisation

### 1. Entraînement d'un modèle

```bash
python src/train.py
```

Le script entraîne plusieurs modèles et sauvegarde le meilleur dans `models/geochem_pipeline.joblib`.

### 2. Prédiction sur un nouvel échantillon

```python
from src.infer import predict_sample

# Exemple d'échantillon (36 valeurs géochimiques)
sample = [57.86, 16.46, 7.96, 0.92, 3.39, 3.29, 0.11, 0.74, 0.23, 2.77,
          5.5, 1400, 117, 1287, 0.9, 20, 4, 91, 119, 123, 10, 3, 49, 12,
          43, 68, 65, 1.76, 9, 40, 20, 110, 23, 17, 164, 4.46]

prediction = predict_sample(sample)
print(f"Classification: {prediction}")
```

### 3. Analyse des seuils

```bash
python src/analyze_thresholds.py
```

### 4. Analyses géochimiques spécialisées

```bash
# Analyse générale
python src/geochemical_analysis.py

# Analyse de l'argent
python src/silver_analysis.py

# Analyse des échantillons potentiels
python src/potential_analysis.py
```

## 📊 Caractéristiques analysées

Le modèle analyse **36 paramètres géochimiques** incluant :

### Éléments majeurs (%)
- SiO2, Al2O3, Fe2O3, MgO, CaO, Na2O, K2O, TiO2, P2O5, MnO

### Éléments traces (ppm)
- Cu, Pb, Zn, Ag, Au, As, Sb, Bi, Cd, Co, Cr, Ni, Mo, W, Sn, V, Ba, Sr, etc.

## 🎯 Performance du modèle

Le système atteint généralement :
- **Précision** : >85%
- **Recall** : >80%
- **F1-Score** : >82%

Les métriques détaillées sont disponibles dans les rapports générés après l'entraînement.

## 📋 Format des données

Les échantillons doivent être fournis sous forme de liste de 36 valeurs dans l'ordre exact des caractéristiques. Le préprocessing gère automatiquement :
- Les virgules décimales → points
- Les valeurs de seuil (ex: "<0.1") → division par 2
- Les valeurs manquantes → 0.0

## 🔧 Configuration

Le système peut être configuré via les paramètres dans les scripts :
- **use_raw_data=True** : Utilise les données brutes avec preprocessing
- **test_size=0.3** : Proportion de données pour le test
- **random_state=42** : Seed pour la reproductibilité

## 📈 Exemples de résultats

```
TEST - Échantillon Minerai:
Données après normalisation:
Min: -1.234
Max: 2.156
Mean: 0.045

Probabilités par classe:
  Sterile: 0.123
  Potentiel: 0.234
  Minerai: 0.643

Résultat: Minerai
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Pushez vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrez une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

---

⭐ N'oubliez pas de donner une étoile au projet si vous l'avez trouvé utile !
```
