# silver_mineralization_analysis.py
"""
Analyse complète de l'argent et des minerais fortement associés
Analyse géochimique spécialisée pour la prospection argentifère
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

class SilverMineralizationAnalyzer:
    def __init__(self, file_path="dataset/raw/BDD_ICP.xlsx"):
        """
        Analyseur spécialisé pour l'argent et les minéralisations associées
        """
        self.df = None
        self.df_processed = None
        self.silver_associated_elements = []
        self.mineralization_model = None
        self.high_grade_threshold = None
        
        # Éléments typiquement associés à l'argent
        self.potential_ag_elements = [
            'Pb', 'Zn', 'Cu', 'As', 'Sb', 'Bi', 'Hg', 'Cd', 'Mo', 'W',
            'Au', 'Te', 'Se', 'S', 'Fe2O3', 'MnO'
        ]
        
        self.load_and_preprocess(file_path)
    
    def load_and_preprocess(self, file_path):
        """
        Charger et préprocesser les données géochimiques
        """
        print("🔄 Chargement et préprocessing des données...")
        
        # Charger le fichier
        self.df = pd.read_excel(file_path)
        
        # Préprocessing standard
        self.df = self.df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
        
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
        
        # Conversion numérique
        for col in self.df.columns:
            if col != "Type":
                self.df[col] = self.df[col].apply(convert_threshold)
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
        
        # Encoder le type
        if "Type" in self.df.columns:
            type_mapping = {"Sterile": 0, "Potentiel": 1, "Minerai": 2}
            self.df["Type"] = self.df["Type"].map(type_mapping)
            self.df["Type_Name"] = self.df["Type"].map({0: "Sterile", 1: "Potentiel", 2: "Minerai"})
        
        # Créer classes d'argent
        if 'Ag (g/t)' in self.df.columns:
            ag_col = 'Ag (g/t)'
        elif 'Ag' in self.df.columns:
            ag_col = 'Ag'
        else:
            raise ValueError("Colonne Argent non trouvée")
        
        # Définir seuils pour l'argent (adaptable selon votre contexte)
        ag_values = self.df[ag_col]
        self.low_grade_threshold = np.percentile(ag_values[ag_values > 0], 33)
        self.high_grade_threshold = np.percentile(ag_values[ag_values > 0], 75)
        
        self.df['Ag_Grade'] = pd.cut(self.df[ag_col], 
                                   bins=[0, self.low_grade_threshold, self.high_grade_threshold, np.inf],
                                   labels=['Low', 'Medium', 'High'])
        
        self.df['Ag_Binary'] = (self.df[ag_col] > self.high_grade_threshold).astype(int)
        
        print(f"✅ Données chargées: {self.df.shape}")
        print(f"📊 Seuils Argent: Bas={self.low_grade_threshold:.1f}, Haut={self.high_grade_threshold:.1f}")
        print(f"📈 Distribution grades Ag: \n{self.df['Ag_Grade'].value_counts()}")
    
    def identify_silver_associations(self):
        """
        Identifier les éléments fortement associés à l'argent
        """
        print("\n" + "="*60)
        print("🔍 IDENTIFICATION DES ASSOCIATIONS ARGENTIFÈRES")
        print("="*60)
        
        ag_col = 'Ag (g/t)' if 'Ag (g/t)' in self.df.columns else 'Ag'
        
        # Calculer corrélations avec l'argent
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = []
        
        for col in numeric_cols:
            if col != ag_col and col not in ['Type', 'Ag_Binary']:
                if col in self.df.columns:
                    corr = self.df[ag_col].corr(self.df[col])
                    if not np.isnan(corr):
                        correlations.append({
                            'Element': col,
                            'Correlation': corr,
                            'Abs_Correlation': abs(corr)
                        })
        
        # Trier par corrélation absolue
        corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
        
        # Sélectionner éléments fortement associés (|r| > 0.3)
        strong_associations = corr_df[corr_df['Abs_Correlation'] > 0.3]
        self.silver_associated_elements = strong_associations['Element'].tolist()
        
        print("🥈 Éléments fortement associés à l'Argent (|r| > 0.3):")
        print("-" * 50)
        for _, row in strong_associations.head(10).iterrows():
            direction = "↗️" if row['Correlation'] > 0 else "↘️"
            print(f"{row['Element']:15s}: {row['Correlation']:6.3f} {direction}")
        
        # Tests statistiques
        print("\n🧪 Tests statistiques (Argent Fort vs Faible):")
        print("-" * 50)
        
        high_ag = self.df[self.df['Ag_Binary'] == 1]
        low_ag = self.df[self.df['Ag_Binary'] == 0]
        
        significant_elements = []
        for element in self.silver_associated_elements[:10]:
            if element in self.df.columns:
                stat, p_value = stats.mannwhitneyu(
                    high_ag[element].dropna(), 
                    low_ag[element].dropna(),
                    alternative='two-sided'
                )
                if p_value < 0.05:
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
                    significant_elements.append(element)
                    
                    high_median = high_ag[element].median()
                    low_median = low_ag[element].median()
                    enrichment = high_median / low_median if low_median > 0 else float('inf')
                    
                    print(f"{element:15s}: p={p_value:.4f} {significance} (enrichissement: {enrichment:.1f}x)")
        
        return corr_df, significant_elements
    
    def advanced_silver_analysis(self):
        """
        Analyses avancées de la minéralisation argentifère
        """
        print("\n" + "="*60)
        print("🔬 ANALYSES AVANCÉES - MINÉRALISATION ARGENTIFÈRE")
        print("="*60)
        
        ag_col = 'Ag (g/t)' if 'Ag (g/t)' in self.df.columns else 'Ag'
        
        # 1. Ratios géochimiques diagnostiques
        print("\n📊 Ratios géochimiques pour l'argent:")
        print("-" * 40)
        
        ratios = {}
        if 'Pb' in self.df.columns and 'Zn' in self.df.columns:
            ratios['Ag/Pb'] = self.df[ag_col] / (self.df['Pb'] + 1)  # +1 pour éviter division par 0
            ratios['Ag/Zn'] = self.df[ag_col] / (self.df['Zn'] + 1)
        
        if 'Cu' in self.df.columns:
            ratios['Ag/Cu'] = self.df[ag_col] / (self.df['Cu'] + 1)
            
        if 'Au' in self.df.columns:
            ratios['Ag/Au'] = self.df[ag_col] / (self.df['Au'] + 1)
        
        # Analyser les ratios par grade d'argent
        for ratio_name, ratio_values in ratios.items():
            self.df[ratio_name] = ratio_values
            high_ag_ratio = ratio_values[self.df['Ag_Binary'] == 1].median()
            low_ag_ratio = ratio_values[self.df['Ag_Binary'] == 0].median()
            print(f"{ratio_name:10s}: Haut Ag = {high_ag_ratio:.3f}, Bas Ag = {low_ag_ratio:.3f}")
        
        # 2. Clustering des échantillons argentifères
        print("\n🎯 Classification des types de minéralisation:")
        print("-" * 50)
        
        # Sélectionner features pour clustering
        cluster_features = []
        for elem in ['Pb', 'Zn', 'Cu', 'As', 'Sb', 'Bi', 'Hg']:
            if elem in self.df.columns:
                cluster_features.append(elem)
        
        if len(cluster_features) >= 3:
            # Normaliser les données
            scaler = StandardScaler()
            ag_samples = self.df[self.df['Ag_Binary'] == 1]
            if len(ag_samples) > 5:
                cluster_data = scaler.fit_transform(ag_samples[cluster_features])
                
                # KMeans clustering
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(cluster_data)
                
                # Analyser les clusters
                for i in range(3):
                    cluster_mask = clusters == i
                    cluster_samples = ag_samples.iloc[cluster_mask]
                    print(f"\nCluster {i+1} ({len(cluster_samples)} échantillons):")
                    
                    for feature in cluster_features[:5]:
                        median_val = cluster_samples[feature].median()
                        print(f"  {feature}: {median_val:.1f}")
        
        return ratios
    
    def create_silver_visualizations(self, corr_df):
        """
        Créer des visualisations spécialisées pour l'argent
        """
        print("\n📊 Création des visualisations argentifères...")
        
        ag_col = 'Ag (g/t)' if 'Ag (g/t)' in self.df.columns else 'Ag'
        
        # Configuration
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribution de l'argent
        ax1 = plt.subplot(3, 4, 1)
        ag_data = self.df[ag_col][self.df[ag_col] > 0]
        ax1.hist(np.log10(ag_data), bins=30, alpha=0.7, color='silver', edgecolor='black')
        ax1.axvline(np.log10(self.low_grade_threshold), color='orange', linestyle='--', label=f'Seuil Bas ({self.low_grade_threshold:.1f})')
        ax1.axvline(np.log10(self.high_grade_threshold), color='red', linestyle='--', label=f'Seuil Haut ({self.high_grade_threshold:.1f})')
        ax1.set_xlabel('log10(Ag g/t)')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution de l\'Argent', fontweight='bold')
        ax1.legend()
        
        # 2. Corrélations avec l'argent
        ax2 = plt.subplot(3, 4, 2)
        top_corr = corr_df.head(8)
        colors = ['red' if x < 0 else 'green' for x in top_corr['Correlation']]
        bars = ax2.barh(range(len(top_corr)), top_corr['Correlation'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_corr)))
        ax2.set_yticklabels(top_corr['Element'])
        ax2.set_xlabel('Corrélation avec Ag')
        ax2.set_title('Corrélations Argentifères', fontweight='bold')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.invert_yaxis()
        
        # 3. Boxplot par grade d'argent
        ax3 = plt.subplot(3, 4, 3)
        if 'Type_Name' in self.df.columns:
            self.df.boxplot(column=ag_col, by='Type_Name', ax=ax3)
            ax3.set_ylabel('Ag (g/t)')
            ax3.set_title('Argent par Type Géologique')
            plt.suptitle('')
        
        # 4. Argent vs élément le plus corrélé
        ax4 = plt.subplot(3, 4, 4)
        if len(corr_df) > 0:
            best_element = corr_df.iloc[0]['Element']
            if best_element in self.df.columns:
                colors_map = {'Low': 'blue', 'Medium': 'orange', 'High': 'red'}
                for grade in ['Low', 'Medium', 'High']:
                    mask = self.df['Ag_Grade'] == grade
                    ax4.scatter(self.df[mask][ag_col], self.df[mask][best_element], 
                              c=colors_map[grade], label=grade, alpha=0.6, s=30)
                ax4.set_xlabel(f'Ag (g/t)')
                ax4.set_ylabel(best_element)
                ax4.set_title(f'Ag vs {best_element}')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        # 5-8. Top 4 éléments associés (scatter plots)
        for i, (_, row) in enumerate(corr_df.head(4).iterrows()):
            ax = plt.subplot(3, 4, 5+i)
            element = row['Element']
            if element in self.df.columns:
                # Scatter plot avec couleurs par grade
                for grade, color in zip(['Low', 'Medium', 'High'], ['blue', 'orange', 'red']):
                    mask = self.df['Ag_Grade'] == grade
                    ax.scatter(self.df[mask][ag_col], self.df[mask][element], 
                              c=color, label=grade, alpha=0.6, s=20)
                
                ax.set_xlabel('Ag (g/t)')
                ax.set_ylabel(element)
                ax.set_title(f'Ag vs {element}\nr={row["Correlation"]:.3f}')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
        
        # 9. Heatmap des associations
        ax9 = plt.subplot(3, 4, 9)
        # Sélectionner les éléments les plus associés
        top_elements = [ag_col] + corr_df.head(6)['Element'].tolist()
        available_elements = [elem for elem in top_elements if elem in self.df.columns]
        
        if len(available_elements) > 2:
            corr_matrix = self.df[available_elements].corr()
            im = ax9.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
            ax9.set_xticks(range(len(available_elements)))
            ax9.set_yticks(range(len(available_elements)))
            ax9.set_xticklabels([elem.replace(' (g/t)', '').replace(' (ppm)', '') for elem in available_elements], rotation=45)
            ax9.set_yticklabels([elem.replace(' (g/t)', '').replace(' (ppm)', '') for elem in available_elements])
            ax9.set_title('Matrice Corrélation\nAssociations Ag')
            
            # Ajouter valeurs
            for i in range(len(available_elements)):
                for j in range(len(available_elements)):
                    text = ax9.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                                   ha="center", va="center", color="white", fontsize=8)
        
        # 10. Ratios géochimiques
        ax10 = plt.subplot(3, 4, 10)
        ratios_to_plot = []
        for ratio_name in ['Ag/Pb', 'Ag/Zn', 'Ag/Cu']:
            if ratio_name in self.df.columns:
                ratios_to_plot.append(ratio_name)
        
        if ratios_to_plot:
            ratio_data = []
            labels = []
            for ratio_name in ratios_to_plot[:3]:
                ratio_values = self.df[ratio_name].replace([np.inf, -np.inf], np.nan).dropna()
                if len(ratio_values) > 0:
                    ratio_data.append(np.log10(ratio_values + 1))
                    labels.append(ratio_name)
            
            if ratio_data:
                ax10.boxplot(ratio_data, labels=labels)
                ax10.set_ylabel('log10(Ratio + 1)')
                ax10.set_title('Ratios Géochimiques Ag')
                ax10.tick_params(axis='x', rotation=45)
        
        # 11. Distribution spatiale (si coordonnées disponibles)
        ax11 = plt.subplot(3, 4, 11)
        # Simuler des coordonnées si pas disponibles
        if 'X' not in self.df.columns:
            self.df['X'] = np.random.uniform(0, 1000, len(self.df))
            self.df['Y'] = np.random.uniform(0, 1000, len(self.df))
        
        scatter = ax11.scatter(self.df['X'], self.df['Y'], c=self.df[ag_col], 
                             cmap='viridis', s=30, alpha=0.7)
        ax11.set_xlabel('X (coordonnée)')
        ax11.set_ylabel('Y (coordonnée)')
        ax11.set_title('Distribution Spatiale Ag')
        plt.colorbar(scatter, ax=ax11, label='Ag (g/t)')
        
        # 12. Cumulative Ag vs autres éléments
        ax12 = plt.subplot(3, 4, 12)
        ag_sorted = np.sort(self.df[ag_col])
        cumulative_ag = np.cumsum(ag_sorted)
        cumulative_pct = np.arange(1, len(ag_sorted)+1) / len(ag_sorted) * 100
        
        ax12.plot(cumulative_pct, cumulative_ag / cumulative_ag[-1] * 100, 'b-', linewidth=2)
        ax12.axhline(80, color='red', linestyle='--', alpha=0.7, label='80%')
        ax12.axvline(20, color='red', linestyle='--', alpha=0.7, label='20% échantillons')
        ax12.set_xlabel('% Échantillons (triés)')
        ax12.set_ylabel('% Ag Cumulé')
        ax12.set_title('Courbe de Lorenz - Ag')
        ax12.legend()
        ax12.grid(True, alpha=0.3)
        
        plt.suptitle('ANALYSE COMPLÈTE DE LA MINÉRALISATION ARGENTIFÈRE', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def build_silver_prediction_models(self, significant_elements):
        """
        Construire des modèles de prédiction pour l'argent
        """
        print("\n" + "="*60)
        print("🤖 MODÉLISATION PRÉDICTIVE ARGENTIFÈRE")
        print("="*60)
        
        ag_col = 'Ag (g/t)' if 'Ag (g/t)' in self.df.columns else 'Ag'
        
        # Préparer les features
        feature_cols = significant_elements[:10] if len(significant_elements) >= 10 else self.silver_associated_elements[:10]
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        if len(available_features) < 3:
            print("⚠️ Pas assez de features pour la modélisation")
            return None
        
        X = self.df[available_features].fillna(0)
        y_regression = self.df[ag_col]
        y_classification = self.df['Ag_Binary']
        
        print(f"📊 Features utilisées: {', '.join(available_features)}")
        print(f"📈 Échantillons: {len(X)}")
        
        # 1. Modèle de régression (prédire la teneur en Ag)
        print("\n🎯 Modèle de Régression (prédiction teneur Ag):")
        print("-" * 50)
        
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            X, y_regression, test_size=0.2, random_state=42
        )
        
        # Random Forest Regressor
        rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_reg.fit(X_train, y_reg_train)
        
        y_reg_pred = rf_reg.predict(X_test)
        r2 = r2_score(y_reg_test, y_reg_pred)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        
        print(f"R² Score: {r2:.3f}")
        print(f"MAE: {mae:.2f} g/t")
        
        # Importance des features pour régression
        feature_importance_reg = pd.DataFrame({
            'Feature': available_features,
            'Importance': rf_reg.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 5 Features (Régression):")
        for _, row in feature_importance_reg.head(5).iterrows():
            print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")
        
        # 2. Modèle de classification (argent fort/faible)
        print("\n🎯 Modèle de Classification (Ag Fort vs Faible):")
        print("-" * 50)
        
        X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(
            X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
        )
        
        # Random Forest Classifier
        rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        rf_clf.fit(X_train_clf, y_clf_train)
        
        y_clf_pred = rf_clf.predict(X_test_clf)
        
        print("Rapport de Classification:")
        print(classification_report(y_clf_test, y_clf_pred, 
                                  target_names=['Ag Faible', 'Ag Fort']))
        
        # Cross-validation
        cv_scores = cross_val_score(rf_clf, X, y_classification, cv=5)
        print(f"\nValidation Croisée (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance pour classification
        feature_importance_clf = pd.DataFrame({
            'Feature': available_features,
            'Importance': rf_clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 5 Features (Classification):")
        for _, row in feature_importance_clf.head(5).iterrows():
            print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")
        
        # Sauvegarder les modèles
        self.mineralization_model = {
            'regressor': rf_reg,
            'classifier': rf_clf,
            'features': available_features,
            'scaler': StandardScaler().fit(X),
            'thresholds': {
                'low': self.low_grade_threshold,
                'high': self.high_grade_threshold
            }
        }
        
        return {
            'regression_r2': r2,
            'classification_accuracy': cv_scores.mean(),
            'feature_importance_reg': feature_importance_reg,
            'feature_importance_clf': feature_importance_clf
        }
    
    def predict_silver_potential(self, sample_data):
        """
        Prédire le potentiel argentifère d'un nouvel échantillon
        """
        if self.mineralization_model is None:
            raise ValueError("Modèle non entraîné. Exécutez build_silver_prediction_models() d'abord.")
        
        features = self.mineralization_model['features']
        
        if len(sample_data) != len(features):
            raise ValueError(f"Attendu {len(features)} features, reçu {len(sample_data)}")
        
        # Préparer les données
        sample_df = pd.DataFrame([sample_data], columns=features)
        
        # Prédictions
        reg_model = self.mineralization_model['regressor']
        clf_model = self.mineralization_model['classifier']
        
        predicted_grade = reg_model.predict(sample_df)[0]
        probability_high = clf_model.predict_proba(sample_df)[0][1]
        
        # Classification
        if predicted_grade > self.high_grade_threshold:
            grade_class = "Forte Teneur"
        elif predicted_grade > self.low_grade_threshold:
            grade_class = "Teneur Moyenne"
        else:
            grade_class = "Faible Teneur"
        
        return {
            'predicted_grade': predicted_grade,
            'grade_class': grade_class,
            'probability_high_grade': probability_high,
            'recommendation': "PRIORITÉ HAUTE" if probability_high > 0.7 else 
                           "PRIORITÉ MOYENNE" if probability_high > 0.4 else "PRIORITÉ FAIBLE"
        }
    
    def generate_silver_report(self, corr_df, significant_elements, model_results):
        """
        Générer un rapport complet sur la minéralisation argentifère
        """
        ag_col = 'Ag (g/t)' if 'Ag (g/t)' in self.df.columns else 'Ag'
        
        print("\n" + "="*80)
        print("📋 RAPPORT COMPLET - MINÉRALISATION ARGENTIFÈRE")
        print("="*80)
        
        print(f"\n🥈 STATISTIQUES GÉNÉRALES DE L'ARGENT:")
        print("-" * 50)
        ag_stats = self.df[ag_col].describe()
        print(f"Échantillons analysés: {len(self.df)}")
        print(f"Teneur moyenne: {ag_stats['mean']:.2f} g/t")
        print(f"Teneur médiane: {ag_stats['50%']:.2f} g/t")
        print(f"Teneur maximum: {ag_stats['max']:.2f} g/t")
        print(f"Seuil teneur forte: >{self.high_grade_threshold:.1f} g/t ({sum(self.df['Ag_Binary'])}/{len(self.df)} échantillons)")
        
        print(f"\n🎯 ÉLÉMENTS PATHFINDERS ARGENTIFÈRES:")
        print("-" * 50)
        print("Top 5 éléments associés (corrélation):")
        for i, (_, row) in enumerate(corr_df.head(5).iterrows(), 1):
            association = "Positive" if row['Correlation'] > 0 else "Négative"
            print(f"{i}. {row['Element']:12s}: r={row['Correlation']:6.3f} ({association})")
        
        if significant_elements:
            print(f"\nÉléments statistiquement significatifs: {len(significant_elements)}")
            print(f"Pathfinders recommandés: {', '.join(significant_elements[:5])}")
        
        print(f"\n🤖 PERFORMANCE DES MODÈLES PRÉDICTIFS:")
        print("-" * 50)
        if model_results:
            print(f"Modèle de régression (R²): {model_results['regression_r2']:.3f}")
            print(f"Modèle de classification: {model_results['classification_accuracy']:.3f}")
            
            print("\nÉléments les plus prédictifs:")
            for _, row in model_results['feature_importance_clf'].head(3).iterrows():
                print(f"  • {row['Feature']:12s}: {row['Importance']:.4f}")
        
        print(f"\n💎 TYPES DE MINÉRALISATION IDENTIFIÉS:")
        print("-" * 50)
        
        # Analyser les associations pour identifier les types de minéralisation
        strong_pos_corr = corr_df[corr_df['Correlation'] > 0.4]['Element'].tolist()
        strong_neg_corr = corr_df[corr_df['Correlation'] < -0.4]['Element'].tolist()
        
        # Identifier le type de système
        system_type = "Indéterminé"
        if 'Pb' in strong_pos_corr and 'Zn' in strong_pos_corr:
            system_type = "Système Pb-Zn-Ag (épithermal/skarn)"
        elif 'As' in strong_pos_corr and 'Sb' in strong_pos_corr:
            system_type = "Système épithermal As-Sb-Ag"
        elif 'Cu' in strong_pos_corr:
            system_type = "Système Cu-Ag (porphyre/épithermal)"
        elif 'Hg' in strong_pos_corr:
            system_type = "Système épithermal Hg-Ag"
        
        print(f"Type de système probable: {system_type}")
        print(f"Éléments enrichis avec Ag: {', '.join(strong_pos_corr[:5])}")
        if strong_neg_corr:
            print(f"Éléments appauvris avec Ag: {', '.join(strong_neg_corr[:3])}")
        
        print(f"\n🔍 RECOMMANDATIONS D'EXPLORATION:")
        print("-" * 50)
        print("1. ANALYSES PRIORITAIRES:")
        priority_elements = corr_df.head(5)['Element'].tolist()
        print(f"   • Toujours analyser: Ag + {', '.join(priority_elements[:3])}")
        print(f"   • Analyses complémentaires: {', '.join(priority_elements[3:5])}")
        
        print("\n2. SEUILS DE PROSPECTION:")
        print(f"   • Anomalie Ag faible: > {self.low_grade_threshold:.1f} g/t")
        print(f"   • Anomalie Ag forte: > {self.high_grade_threshold:.1f} g/t")
        
        if 'Pb' in corr_df['Element'].values:
            pb_threshold = self.df[self.df['Ag_Binary']==1]['Pb'].quantile(0.5)
            print(f"   • Pb associé: > {pb_threshold:.1f} ppm")
        
        print("\n3. STRATÉGIE DE SUIVI:")
        print("   • Échantillons Ag > seuil fort → Priorité maximale")
        print("   • Analyser les halos géochimiques autour des anomalies Ag")
        print("   • Vérifier les associations avec la géologie structurale")
        
        print(f"\n📊 DISTRIBUTION SPATIALE:")
        print("-" * 50)
        high_ag_samples = self.df[self.df['Ag_Binary'] == 1]
        if len(high_ag_samples) > 0:
            print(f"Échantillons à forte teneur Ag: {len(high_ag_samples)} ({len(high_ag_samples)/len(self.df)*100:.1f}%)")
            
            if 'Type_Name' in self.df.columns:
                ag_by_type = high_ag_samples['Type_Name'].value_counts()
                print("\nDistribution par type géologique:")
                for rock_type, count in ag_by_type.items():
                    percentage = count / len(high_ag_samples) * 100
                    print(f"   • {rock_type}: {count} échantillons ({percentage:.1f}%)")
        
        print("\n" + "="*80)
        
        return {
            'ag_statistics': ag_stats,
            'system_type': system_type,
            'priority_elements': priority_elements,
            'thresholds': {
                'low': self.low_grade_threshold,
                'high': self.high_grade_threshold
            }
        }
    
    def run_complete_silver_analysis(self):
        """
        Exécuter l'analyse complète de la minéralisation argentifère
        """
        print("🚀 DÉMARRAGE DE L'ANALYSE ARGENTIFÈRE COMPLÈTE")
        print("="*80)
        
        # 1. Identifier les associations
        corr_df, significant_elements = self.identify_silver_associations()
        
        # 2. Analyses avancées
        ratios = self.advanced_silver_analysis()
        
        # 3. Visualisations
        self.create_silver_visualizations(corr_df)
        
        # 4. Modélisation
        model_results = self.build_silver_prediction_models(significant_elements)
        
        # 5. Rapport final
        report_data = self.generate_silver_report(corr_df, significant_elements, model_results)
        
        return {
            'correlations': corr_df,
            'significant_elements': significant_elements,
            'ratios': ratios,
            'model_results': model_results,
            'report_data': report_data,
            'analyzer': self
        }

# Classe utilitaire pour la prédiction en temps réel
class SilverProspector:
    """
    Outil de prospection argentifère en temps réel
    """
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.model = analyzer.mineralization_model
        
    def assess_sample(self, sample_data, sample_name="Échantillon"):
        """
        Évaluer un échantillon pour son potentiel argentifère
        """
        try:
            prediction = self.analyzer.predict_silver_potential(sample_data)
            
            print(f"\n🔍 ÉVALUATION ARGENTIFÈRE - {sample_name}")
            print("="*50)
            print(f"Teneur Ag prédite: {prediction['predicted_grade']:.2f} g/t")
            print(f"Classe: {prediction['grade_class']}")
            print(f"Probabilité teneur forte: {prediction['probability_high_grade']:.1%}")
            print(f"Recommandation: {prediction['recommendation']}")
            
            return prediction
            
        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation: {e}")
            return None
    
    def batch_assessment(self, samples_data, sample_names=None):
        """
        Évaluer plusieurs échantillons
        """
        if sample_names is None:
            sample_names = [f"Échantillon_{i+1}" for i in range(len(samples_data))]
        
        results = []
        for i, (sample, name) in enumerate(zip(samples_data, sample_names)):
            prediction = self.assess_sample(sample, name)
            if prediction:
                results.append({
                    'sample_name': name,
                    'predicted_grade': prediction['predicted_grade'],
                    'probability_high': prediction['probability_high_grade'],
                    'recommendation': prediction['recommendation']
                })
        
        # Trier par potentiel
        results.sort(key=lambda x: x['probability_high'], reverse=True)
        
        print(f"\n🏆 CLASSEMENT DES ÉCHANTILLONS PAR POTENTIEL ARGENTIFÈRE:")
        print("="*70)
        for i, result in enumerate(results, 1):
            print(f"{i:2d}. {result['sample_name']:15s} | "
                  f"{result['predicted_grade']:6.2f} g/t | "
                  f"{result['probability_high']:5.1%} | "
                  f"{result['recommendation']}")
        
        return results

# Exemples d'utilisation et de test
def demo_silver_analysis():
    """
    Démonstration de l'analyse argentifère
    """
    print("🎮 DÉMONSTRATION - ANALYSE ARGENTIFÈRE")
    print("="*60)
    
    # Initialiser l'analyseur
    try:
        analyzer = SilverMineralizationAnalyzer("dataset/raw/BDD_ICP.xlsx")
        
        # Exécuter l'analyse complète
        results = analyzer.run_complete_silver_analysis()
        
        # Démonstration de prédiction
        if analyzer.mineralization_model:
            print("\n DÉMONSTRATION DE PRÉDICTION:")
            print("-" * 40)
            
            # Créer un prospecteur
            prospector = SilverProspector(analyzer)
            
            # Exemples d'échantillons (adaptez selon vos colonnes)
            features = analyzer.mineralization_model['features']
            print(f"Features requises: {', '.join(features)}")
            
            # Échantillon exemple (valeurs moyennes)
            sample_moyennes = analyzer.df[features].mean().values
            sample_fort_ag = sample_moyennes * 2  # Simuler échantillon enrichi
            sample_faible_ag = sample_moyennes * 0.5  # Simuler échantillon appauvri
            
            # Évaluations
            prospector.assess_sample(sample_moyennes, "Échantillon Moyen")
            prospector.assess_sample(sample_fort_ag, "Échantillon Enrichi")
            prospector.assess_sample(sample_faible_ag, "Échantillon Appauvri")
        
        print("\nAnalyse argentifère terminée avec succès!")
        return results
        
    except FileNotFoundError:
        print("Fichier de données non trouvé. Vérifiez le chemin.")
        print("Adaptez le chemin dans SilverMineralizationAnalyzer()")
        return None
    except Exception as e:
        print(f" Erreur lors de l'analyse: {e}")
        return None

# Point d'entrée principal
if __name__ == "__main__":
    # Exécuter l'analyse complète
    results = demo_silver_analysis()
    
    # Optionnel: sauvegarder les résultats
    if results:
        import pickle
        with open("silver_analysis_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print("\nRésultats sauvegardés dans 'silver_analysis_results.pkl'")
        
        # Résumé final
        print(f"\n RÉSUMÉ FINAL:")
        print(f"✓ Éléments analysés: {len(results['correlations'])}")
        print(f"✓ Éléments significatifs: {len(results['significant_elements'])}")
        if results['model_results']:
            print(f"✓ Performance modèle: {results['model_results']['classification_accuracy']:.1%}")
        print(f"✓ Système probable: {results['report_data']['system_type']}")
        print(f"✓ Seuils définis: {results['report_data']['thresholds']['low']:.1f} - {results['report_data']['thresholds']['high']:.1f} g/t")