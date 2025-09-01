# geochemical_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class GeochemicalAnalyzer:
    def __init__(self, file_path="dataset/raw/BDD_ICP.xlsx"):
        """
        Initialiser l'analyseur géochimique
        """
        self.df = None
        self.df_clean = None
        self.load_and_prepare_data(file_path)
        
    def load_and_prepare_data(self, file_path):
        """
        Charger et nettoyer les données
        """
        # Charger le fichier Excel
        self.df = pd.read_excel(file_path)
        
        # Remplacer les virgules par des points
        self.df = self.df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
        
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
        for col in self.df.columns:
            if col != "Type":
                self.df[col] = self.df[col].apply(convert_threshold)
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
        
        # Encodage de la colonne "Type"
        type_mapping = {"Sterile": 0, "Potentiel": 1, "Minerai": 2}
        self.df["Type"] = self.df["Type"].map(type_mapping)
        
        # Créer version avec noms lisibles pour les graphiques
        self.df_clean = self.df.copy()
        inv_mapping = {0: "Sterile", 1: "Potentiel", 2: "Minerai"}
        self.df_clean["Type_Name"] = self.df_clean["Type"].map(inv_mapping)
        
        print(f"✅ Données chargées: {self.df.shape}")
        print(f"📊 Distribution des types: \n{self.df_clean['Type_Name'].value_counts()}")
    
    def statistical_analysis(self):
        """
        Analyse statistique des différences entre types
        """
        print("\n" + "="*60)
        print("ANALYSE STATISTIQUE PAR TYPE")
        print("="*60)
        
        # Statistiques descriptives par type
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.drop(['Type'])
        
        stats_by_type = {}
        for type_name in ["Sterile", "Potentiel", "Minerai"]:
            type_code = {"Sterile": 0, "Potentiel": 1, "Minerai": 2}[type_name]
            subset = self.df[self.df["Type"] == type_code]
            stats_by_type[type_name] = subset[numeric_cols].describe()
        
        # Test ANOVA pour chaque élément
        print("\n🔬 Tests ANOVA (p-values < 0.05 = différences significatives):")
        print("-" * 70)
        
        significant_elements = []
        anova_results = []
        
        for col in numeric_cols:
            sterile_vals = self.df[self.df["Type"] == 0][col]
            potentiel_vals = self.df[self.df["Type"] == 1][col]
            minerai_vals = self.df[self.df["Type"] == 2][col]
            
            # Test ANOVA
            f_stat, p_value = stats.f_oneway(sterile_vals, potentiel_vals, minerai_vals)
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            anova_results.append({
                'Element': col,
                'F_statistic': f_stat,
                'p_value': p_value,
                'Significance': significance
            })
            
            if p_value < 0.05:
                significant_elements.append(col)
                print(f"{col:15s}: p={p_value:.4f} {significance}")
        
        print(f"\n✅ {len(significant_elements)} éléments montrent des différences significatives")
        
        return stats_by_type, anova_results, significant_elements
    
    def feature_importance_analysis(self):
        """
        Analyse d'importance des features avec Random Forest
        """
        print("\n" + "="*60)
        print("IMPORTANCE DES FEATURES (Random Forest)")
        print("="*60)
        
        X = self.df.drop(["Type"], axis=1)
        y = self.df["Type"]
        
        # Entraîner Random Forest
        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        rf.fit(X, y)
        
        # Récupérer l'importance des features
        importance_df = pd.DataFrame({
            'Element': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 15 éléments les plus importants:")
        print("-" * 40)
        for idx, row in importance_df.head(15).iterrows():
            print(f"{row['Element']:15s}: {row['Importance']:.4f}")
        
        return importance_df
    
    def correlation_analysis(self):
        """
        Analyse des corrélations
        """
        print("\n" + "="*60)
        print("ANALYSE DES CORRÉLATIONS")
        print("="*60)
        
        # Matrice de corrélation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        # Corrélations avec le Type
        type_correlations = corr_matrix['Type'].drop('Type').sort_values(key=abs, ascending=False)
        
        print("Corrélations les plus fortes avec le Type:")
        print("-" * 45)
        for element, corr in type_correlations.head(15).items():
            direction = "↑" if corr > 0 else "↓"
            print(f"{element:15s}: {corr:6.3f} {direction}")
        
        return corr_matrix, type_correlations
    
    def create_visualizations(self, stats_by_type, importance_df, type_correlations):
        """
        Créer les visualisations
        """
        print("\nCréation des visualisations...")
        
        # Configuration des graphiques
        plt.style.use('default')
        colors = ['#FF6B6B', "#10C662", '#45B7D1']
        
        # Figure 1: Distribution des types
        fig = plt.figure(figsize=(20, 15))
        
        # Subplot 1: Distribution des types
        ax1 = plt.subplot(2, 3, 1)
        type_counts = self.df_clean['Type_Name'].value_counts()
        bars = ax1.bar(type_counts.index, type_counts.values, color=colors)
        ax1.set_title('Distribution des Types de Roches', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Nombre d\'échantillons')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, type_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: Top 10 Feature Importance
        ax2 = plt.subplot(2, 3, 2)
        top_features = importance_df.head(10)
        bars = ax2.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['Element'])
        ax2.set_title('Top 10 - Importance des Features', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Importance')
        ax2.invert_yaxis()
        
        # Subplot 3: Corrélations avec Type
        ax3 = plt.subplot(2, 3, 3)
        top_corr = type_correlations.head(10)
        colors_corr = ['red' if x < 0 else 'green' for x in top_corr.values]
        bars = ax3.barh(range(len(top_corr)), top_corr.values, color=colors_corr, alpha=0.7)
        ax3.set_yticks(range(len(top_corr)))
        ax3.set_yticklabels(top_corr.index)
        ax3.set_title('Top 10 - Corrélations avec Type', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Corrélation')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.invert_yaxis()
        
        # Subplot 4: Boxplot des éléments les plus importants
        ax4 = plt.subplot(2, 3, 4)
        top_element = importance_df.iloc[0]['Element']
        self.df_clean.boxplot(column=top_element, by='Type_Name', ax=ax4)
        ax4.set_title(f'Distribution de {top_element} par Type')
        ax4.set_xlabel('Type')
        ax4.set_ylabel(top_element)
        plt.suptitle('')  # Supprimer le titre automatique
        
        # Subplot 5: Scatter plot 2D des 2 éléments les plus importants
        ax5 = plt.subplot(2, 3, 5)
        elem1 = importance_df.iloc[0]['Element']
        elem2 = importance_df.iloc[1]['Element']
        
        for i, (type_name, color) in enumerate(zip(['Sterile', 'Potentiel', 'Minerai'], colors)):
            mask = self.df_clean['Type_Name'] == type_name
            ax5.scatter(self.df_clean[mask][elem1], self.df_clean[mask][elem2], 
                       c=color, label=type_name, alpha=0.7, s=50)
        
        ax5.set_xlabel(elem1)
        ax5.set_ylabel(elem2)
        ax5.set_title(f'{elem1} vs {elem2}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: Heatmap des corrélations (éléments les plus importants)
        ax6 = plt.subplot(2, 3, 6)
        top_elements = importance_df.head(8)['Element'].tolist() + ['Type']
        corr_subset = self.df[top_elements].corr()
        
        im = ax6.imshow(corr_subset.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax6.set_xticks(range(len(top_elements)))
        ax6.set_yticks(range(len(top_elements)))
        ax6.set_xticklabels(top_elements, rotation=45, ha='right')
        ax6.set_yticklabels(top_elements)
        ax6.set_title('Matrice de Corrélation\n(Top Features)')
        
        # Ajouter les valeurs de corrélation
        for i in range(len(top_elements)):
            for j in range(len(top_elements)):
                text = ax6.text(j, i, f'{corr_subset.values[i, j]:.2f}',
                               ha="center", va="center", color="white", fontsize=8)
        
        plt.colorbar(im, ax=ax6, shrink=0.8)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Comparaisons détaillées des top 6 éléments
        fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        top_6_elements = importance_df.head(6)['Element'].tolist()
        
        for i, element in enumerate(top_6_elements):
            ax = axes[i]
            
            # Boxplot
            data_by_type = [self.df_clean[self.df_clean['Type_Name'] == t][element] 
                           for t in ['Sterile', 'Potentiel', 'Minerai']]
            
            bp = ax.boxplot(data_by_type, labels=['Sterile', 'Potentiel', 'Minerai'],
                           patch_artist=True)
            
            # Colorer les boxplots
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{element}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Ajouter les moyennes
            for j, type_name in enumerate(['Sterile', 'Potentiel', 'Minerai']):
                mean_val = self.df_clean[self.df_clean['Type_Name'] == type_name][element].mean()
                ax.scatter(j+1, mean_val, color='red', s=50, marker='D', zorder=10)
        
        plt.suptitle('Distribution des 6 Éléments les Plus Discriminants', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self, stats_by_type, importance_df, type_correlations, significant_elements):
        """
        Générer un rapport de synthèse
        """
        print("\n" + "="*80)
        print("📋 RAPPORT DE SYNTHÈSE - RELATIONS GÉOCHIMIQUES")
        print("="*80)
        
        print(f"\n🎯 ÉLÉMENTS DISCRIMINANTS CLÉS:")
        print("-" * 40)
        
        # Top 5 par importance
        top_5_importance = importance_df.head(5)
        print("\n1️⃣ Top 5 par importance (Random Forest):")
        for idx, row in top_5_importance.iterrows():
            print(f"   • {row['Element']:15s}: {row['Importance']:.4f}")
        
        # Top 5 par corrélation
        top_5_corr = type_correlations.head(5)
        print("\n2️⃣ Top 5 par corrélation avec le type:")
        for element, corr in top_5_corr.items():
            trend = "favorise Minerai" if corr > 0 else "favorise Sterile"
            print(f"   • {element:15s}: {corr:6.3f} ({trend})")
        
        print(f"\n🧪 ANALYSE STATISTIQUE:")
        print("-" * 40)
        print(f"   • {len(significant_elements)} éléments montrent des différences significatives (ANOVA p<0.05)")
        print(f"   • Éléments les plus discriminants: {', '.join(significant_elements[:5])}")
        
        print(f"\n💎 CARACTÉRISTIQUES PAR TYPE:")
        print("-" * 40)
        
        # Analyser les tendances pour chaque type
        for type_name in ["Sterile", "Potentiel", "Minerai"]:
            type_code = {"Sterile": 0, "Potentiel": 1, "Minerai": 2}[type_name]
            subset = self.df[self.df["Type"] == type_code]
            
            print(f"\n{type_name.upper()}:")
            
            # Éléments enrichis (> moyenne générale)
            enriched = []
            depleted = []
            
            for element in importance_df.head(10)['Element']:
                if element in self.df.columns:
                    type_mean = subset[element].mean()
                    global_mean = self.df[element].mean()
                    
                    if type_mean > global_mean * 1.2:  # 20% au-dessus de la moyenne
                        enriched.append(f"{element} ({type_mean/global_mean:.1f}x)")
                    elif type_mean < global_mean * 0.8:  # 20% en-dessous de la moyenne
                        depleted.append(f"{element} ({type_mean/global_mean:.1f}x)")
            
            if enriched:
                print(f"   ↗️  Enrichi en: {', '.join(enriched[:3])}")
            if depleted:
                print(f"   ↘️  Appauvri en: {', '.join(depleted[:3])}")
        
        print(f"\n🔬 RECOMMANDATIONS ANALYTIQUES:")
        print("-" * 40)
        print(f"   • Éléments prioritaires à analyser: {', '.join(importance_df.head(5)['Element'])}")
        print(f"   • Ratios géochimiques utiles: {top_5_importance.iloc[0]['Element']}/{top_5_importance.iloc[1]['Element']}")
        print(f"   • Surveillez particulièrement: {', '.join([e for e in significant_elements[:3]])}")
        
        print("\n" + "="*80)
    
    def run_complete_analysis(self):
        """
        Exécuter l'analyse complète
        """
        print("🚀 DÉMARRAGE DE L'ANALYSE GÉOCHIMIQUE COMPLÈTE")
        print("="*80)
        
        # 1. Analyse statistique
        stats_by_type, anova_results, significant_elements = self.statistical_analysis()
        
        # 2. Analyse d'importance des features
        importance_df = self.feature_importance_analysis()
        
        # 3. Analyse des corrélations
        corr_matrix, type_correlations = self.correlation_analysis()
        
        # 4. Visualisations
        self.create_visualizations(stats_by_type, importance_df, type_correlations)
        
        # 5. Rapport de synthèse
        self.generate_summary_report(stats_by_type, importance_df, type_correlations, significant_elements)
        
        return {
            'stats_by_type': stats_by_type,
            'importance_df': importance_df,
            'type_correlations': type_correlations,
            'significant_elements': significant_elements,
            'anova_results': anova_results
        }

# Utilisation
if __name__ == "__main__":
    # Créer l'analyseur
    analyzer = GeochemicalAnalyzer("dataset/raw/BDD_ICP.xlsx")
    
    # Exécuter l'analyse complète
    results = analyzer.run_complete_analysis()
    
    # Optionnel: Sauvegarder les résultats
    import pickle
    with open("geochemical_analysis_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print("\n✅ Analyse terminée! Résultats sauvegardés dans 'geochemical_analysis_results.pkl'")