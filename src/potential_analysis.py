# """
# Analyse du potentiel minier des échantillons Stérile/Potentiel.

# Détermine si un échantillon classé comme Stérile ou Potentiel 
# pourrait potentiellement être Minerai en analysant :
# - Proximité géochimique avec les échantillons Minerai
# - Éléments pathfinder élevés
# - Scores de similarité
# - Recommandations d'analyses complémentaires

# Usage:
#     python potential_analysis.py --predictions predictions.csv --model models/geochem_pipeline.joblib
# """

# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
# from scipy import stats
# import argparse
# import os
# from typing import Dict, List, Tuple


# class PotentialAnalyzer:
#     """Analyse le potentiel minier caché des échantillons"""
    
#     def __init__(self, model_path: str, training_data_path: str = None):
#         """
#         Initialise l'analyseur avec les données de référence
        
#         Args:
#             model_path: Chemin vers le modèle entraîné
#             training_data_path: Chemin vers les données d'entraînement (optionnel)
#         """
#         # Charger le modèle et les métadonnées
#         self.pipeline_data = joblib.load(model_path)
#         self.model = self.pipeline_data["model"]
#         self.scaler = self.pipeline_data["scaler"]
#         self.columns = self.pipeline_data["columns"]
        
#         # Charger les données de référence (échantillons Minerai connus)
#         self.reference_data = None
#         if training_data_path:
#             self._load_reference_data(training_data_path)
        
#         # Éléments pathfinder typiques pour les gisements
#         self.pathfinder_elements = [
#             'Au', 'Ag', 'Cu', 'Pb', 'Zn', 'As', 'Sb', 'Bi', 'Te', 'Mo', 'W'
#         ]
        
#         print("✅ Analyseur de potentiel initialisé")
    
#     def _load_reference_data(self, data_path: str):
#         """Charge les données de référence (échantillons Minerai)"""
#         if data_path.endswith('.csv'):
#             df = pd.read_csv(data_path)
#         else:
#             # Assumer que c'est le chemin des données preprocessées
#             df = pd.read_csv("dataset/processed/BDD_ICP_balanced.csv")
        
#         # Filtrer seulement les échantillons Minerai
#         minerai_samples = df[df['Type'] == 'Minerai'].copy()
#         if len(minerai_samples) > 0:
#             # Sélectionner seulement les colonnes du modèle
#             available_cols = [col for col in self.columns if col in minerai_samples.columns]
#             self.reference_data = minerai_samples[available_cols]
            
#             # Normaliser avec le même scaler
#             if self.scaler:
#                 self.reference_data_scaled = pd.DataFrame(
#                     self.scaler.transform(self.reference_data),
#                     columns=self.reference_data.columns,
#                     index=self.reference_data.index
#                 )
#             else:
#                 self.reference_data_scaled = self.reference_data
                
#             print(f"📊 {len(self.reference_data)} échantillons Minerai de référence chargés")
    
#     def analyze_potential(self, sample_data: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
#         """
#         Analyse le potentiel minier des échantillons Stérile/Potentiel
        
#         Args:
#             sample_data: DataFrame avec les données géochimiques
#             predictions: DataFrame avec les prédictions du modèle
            
#         Returns:
#             Dictionnaire avec analyses détaillées
#         """
#         results = {
#             'samples_analysis': [],
#             'summary': {},
#             'recommendations': []
#         }
        
#         # Filtrer les échantillons Stérile et Potentiel
#         non_minerai = predictions[predictions['prediction'].isin(['Sterile', 'Potentiel'])].copy()
        
#         if len(non_minerai) == 0:
#             print("ℹ️  Aucun échantillon Stérile/Potentiel à analyser")
#             return results
        
#         print(f"🔍 Analyse de {len(non_minerai)} échantillons non-Minerai")
        
#         # Normaliser les données
#         sample_cols = [col for col in self.columns if col in sample_data.columns]
#         X = sample_data[sample_cols].fillna(sample_data[sample_cols].median())
        
#         if self.scaler:
#             X_scaled = pd.DataFrame(
#                 self.scaler.transform(X),
#                 columns=X.columns,
#                 index=X.index
#             )
#         else:
#             X_scaled = X
        
#         for idx, row in non_minerai.iterrows():
#             sample_analysis = self._analyze_single_sample(
#                 X_scaled.loc[idx], X.loc[idx], row, idx
#             )
#             results['samples_analysis'].append(sample_analysis)
        
#         # Générer résumé et recommandations
#         results['summary'] = self._generate_summary(results['samples_analysis'])
#         results['recommendations'] = self._generate_recommendations(results['samples_analysis'])
        
#         return results
    
#     def _analyze_single_sample(self, sample_scaled: pd.Series, sample_raw: pd.Series, 
#                              prediction_row: pd.Series, sample_id) -> Dict:
#         """Analyse détaillée d'un échantillon individuel"""
        
#         analysis = {
#             'sample_id': sample_id,
#             'current_prediction': prediction_row['prediction'],
#             'current_confidence': prediction_row['confidence'],
#             'potential_score': 0.0,
#             'similarity_to_minerai': 0.0,
#             'pathfinder_score': 0.0,
#             'anomaly_score': 0.0,
#             'key_elements': {},
#             'recommendation': 'Standard',
#             'priority': 'Low'
#         }
        
#         # 1. Similarité avec échantillons Minerai de référence
#         if self.reference_data is not None:
#             similarity_scores = cosine_similarity(
#                 sample_scaled.values.reshape(1, -1),
#                 self.reference_data_scaled.values
#             ).flatten()
#             analysis['similarity_to_minerai'] = float(np.max(similarity_scores))
        
#         # 2. Score des éléments pathfinder
#         pathfinder_scores = []
#         available_pathfinders = [elem for elem in self.pathfinder_elements 
#                                if elem in sample_raw.index]
        
#         for element in available_pathfinders:
#             value = sample_raw[element]
#             if pd.notna(value) and value > 0:
#                 # Calculer percentile par rapport aux données de référence
#                 if self.reference_data is not None and element in self.reference_data.columns:
#                     ref_values = self.reference_data[element].dropna()
#                     if len(ref_values) > 0:
#                         percentile_rank = (ref_values <= value).mean()
#                         pathfinder_scores.append(percentile_rank)
#                         analysis['key_elements'][element] = {
#                             'value': float(value),
#                             'percentile_rank': float(percentile_rank)
#                         }
        
#         if pathfinder_scores:
#             analysis['pathfinder_score'] = float(np.mean(pathfinder_scores))
        
#         # 3. Score d'anomalie géochimique
#         analysis['anomaly_score'] = self._calculate_anomaly_score(sample_raw)
        
#         # 4. Score de potentiel global
#         weights = {
#             'similarity': 0.4,
#             'pathfinder': 0.4,
#             'anomaly': 0.2
#         }
        
#         analysis['potential_score'] = (
#             analysis['similarity_to_minerai'] * weights['similarity'] +
#             analysis['pathfinder_score'] * weights['pathfinder'] +
#             analysis['anomaly_score'] * weights['anomaly']
#         )
        
#         # 5. Classification du potentiel
#         analysis['recommendation'], analysis['priority'] = self._classify_potential(analysis)
        
#         return analysis
    
#     def _calculate_anomaly_score(self, sample: pd.Series) -> float:
#         """Calcule un score d'anomalie géochimique"""
#         # Score basé sur les ratios d'éléments typiques
#         score = 0.0
#         count = 0
        
#         # Ratios typiques pour gisements
#         ratios_to_check = [
#             ('Au', 'Ag', 0.01, 0.1),  # Ratio Au/Ag
#             ('Cu', 'Zn', 0.1, 10),    # Ratio Cu/Zn
#             ('Pb', 'Zn', 0.1, 5),     # Ratio Pb/Zn
#         ]
        
#         for elem1, elem2, min_ratio, max_ratio in ratios_to_check:
#             if elem1 in sample.index and elem2 in sample.index:
#                 val1, val2 = sample[elem1], sample[elem2]
#                 if pd.notna(val1) and pd.notna(val2) and val1 > 0 and val2 > 0:
#                     ratio = val1 / val2
#                     if min_ratio <= ratio <= max_ratio:
#                         score += 1.0
#                     count += 1
        
#         return score / count if count > 0 else 0.0
    
#     def _classify_potential(self, analysis: Dict) -> Tuple[str, str]:
#         """Classifie le potentiel et la priorité d'un échantillon"""
#         score = analysis['potential_score']
#         current_pred = analysis['current_prediction']
        
#         if score >= 0.7:
#             if current_pred == 'Potentiel':
#                 return "Très prometteur - Analyses approfondies recommandées", "High"
#             else:
#                 return "Potentiel caché élevé - Réévaluation suggérée", "High"
#         elif score >= 0.5:
#             return "Potentiel modéré - Surveillance recommandée", "Medium"
#         elif score >= 0.3:
#             return "Potentiel faible - Analyses ciblées optionnelles", "Low"
#         else:
#             return "Potentiel très faible - Classification confirmée", "Very Low"
    
#     def _generate_summary(self, analyses: List[Dict]) -> Dict:
#         """Génère un résumé des analyses"""
#         if not analyses:
#             return {}
        
#         scores = [a['potential_score'] for a in analyses]
#         priorities = [a['priority'] for a in analyses]
        
#         return {
#             'total_analyzed': len(analyses),
#             'avg_potential_score': float(np.mean(scores)),
#             'high_priority_count': sum(1 for p in priorities if p == 'High'),
#             'medium_priority_count': sum(1 for p in priorities if p == 'Medium'),
#             'promising_samples': [a['sample_id'] for a in analyses if a['potential_score'] >= 0.6]
#         }
    
#     def _generate_recommendations(self, analyses: List[Dict]) -> List[str]:
#         """Génère des recommandations basées sur les analyses"""
#         recommendations = []
        
#         high_potential = [a for a in analyses if a['potential_score'] >= 0.6]
#         medium_potential = [a for a in analyses if 0.4 <= a['potential_score'] < 0.6]
        
#         if high_potential:
#             recommendations.append(
#                 f"🔥 {len(high_potential)} échantillon(s) à fort potentiel identifié(s) : "
#                 f"{[a['sample_id'] for a in high_potential[:5]]}"
#             )
#             recommendations.append(
#                 "   → Recommandation : Analyses géochimiques complémentaires (TR, métallographie)"
#             )
        
#         if medium_potential:
#             recommendations.append(
#                 f"⚡ {len(medium_potential)} échantillon(s) à potentiel modéré : "
#                 f"{[a['sample_id'] for a in medium_potential[:5]]}"
#             )
#             recommendations.append(
#                 "   → Recommandation : Surveillance lors des campagnes futures"
#             )
        
#         # Analyses spécifiques recommandées
#         pathfinder_elements = set()
#         for analysis in high_potential:
#             pathfinder_elements.update(analysis['key_elements'].keys())
        
#         if pathfinder_elements:
#             recommendations.append(
#                 f"🧪 Analyses complémentaires suggérées : {', '.join(pathfinder_elements)}"
#             )
        
#         return recommendations
    
#     def export_results(self, results: Dict, output_path: str):
#         """Exporte les résultats vers un fichier CSV"""
#         if not results['samples_analysis']:
#             print("⚠️  Aucun résultat à exporter")
#             return
        
#         # Créer DataFrame détaillé
#         detailed_results = []
#         for analysis in results['samples_analysis']:
#             row = {
#                 'sample_id': analysis['sample_id'],
#                 'current_prediction': analysis['current_prediction'],
#                 'current_confidence': analysis['current_confidence'],
#                 'potential_score': analysis['potential_score'],
#                 'similarity_to_minerai': analysis['similarity_to_minerai'],
#                 'pathfinder_score': analysis['pathfinder_score'],
#                 'anomaly_score': analysis['anomaly_score'],
#                 'recommendation': analysis['recommendation'],
#                 'priority': analysis['priority']
#             }
            
#             # Ajouter éléments clés
#             for elem, data in analysis['key_elements'].items():
#                 row[f'{elem}_value'] = data['value']
#                 row[f'{elem}_percentile'] = data['percentile_rank']
            
#             detailed_results.append(row)
        
#         df = pd.DataFrame(detailed_results)
#         df.to_csv(output_path, index=False)
#         print(f"💾 Résultats détaillés exportés vers : {output_path}")
    
#     def generate_report(self, results: Dict) -> str:
#         """Génère un rapport textuel complet"""
#         if not results['samples_analysis']:
#             return "Aucune analyse effectuée."
        
#         summary = results['summary']
#         recommendations = results['recommendations']
        
#         report = []
#         report.append("=" * 80)
#         report.append("🏗️  ANALYSE DU POTENTIEL MINIER CACHÉ")
#         report.append("=" * 80)
        
#         # Résumé statistique
#         report.append(f"\n📊 RÉSUMÉ STATISTIQUE :")
#         report.append(f"   • Échantillons analysés : {summary['total_analyzed']}")
#         report.append(f"   • Score potentiel moyen : {summary['avg_potential_score']:.2f}")
#         report.append(f"   • Priorité haute : {summary['high_priority_count']}")
#         report.append(f"   • Priorité moyenne : {summary['medium_priority_count']}")
        
#         # Échantillons prometteurs
#         if summary['promising_samples']:
#             report.append(f"\n🎯 ÉCHANTILLONS PROMETTEURS :")
#             for sample_id in summary['promising_samples'][:10]:  # Top 10
#                 analysis = next(a for a in results['samples_analysis'] if a['sample_id'] == sample_id)
#                 report.append(
#                     f"   • {sample_id}: Score {analysis['potential_score']:.2f} "
#                     f"({analysis['current_prediction']} → {analysis['priority']} priority)"
#                 )
        
#         # Recommandations
#         if recommendations:
#             report.append(f"\n💡 RECOMMANDATIONS :")
#             for rec in recommendations:
#                 report.append(f"   {rec}")
        
#         # Top éléments pathfinder
#         all_elements = {}
#         for analysis in results['samples_analysis']:
#             for elem, data in analysis['key_elements'].items():
#                 if elem not in all_elements:
#                     all_elements[elem] = []
#                 all_elements[elem].append(data['percentile_rank'])
        
#         if all_elements:
#             report.append(f"\n🧪 ÉLÉMENTS PATHFINDER LES PLUS SIGNIFICATIFS :")
#             elem_scores = {elem: np.mean(scores) for elem, scores in all_elements.items()}
#             top_elements = sorted(elem_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
#             for elem, avg_score in top_elements:
#                 report.append(f"   • {elem}: Score moyen {avg_score:.2f}")
        
#         return "\n".join(report)


# def main():
#     parser = argparse.ArgumentParser(description="Analyse du potentiel minier caché")
#     parser.add_argument("--predictions", type=str, required=True, 
#                        help="Fichier CSV avec les prédictions")
#     parser.add_argument("--data", type=str, required=True,
#                        help="Fichier CSV avec les données géochimiques originales")
#     parser.add_argument("--model", type=str, default="models/geochem_pipeline.joblib",
#                        help="Chemin du modèle entraîné")
#     parser.add_argument("--reference", type=str, 
#                        help="Données de référence (échantillons Minerai connus)")
#     parser.add_argument("--output", type=str, default="potential_analysis_results.csv",
#                        help="Fichier de sortie")
    
#     args = parser.parse_args()
    
#     # Vérifier les fichiers
#     for file_path in [args.predictions, args.data, args.model]:
#         if not os.path.exists(file_path):
#             print(f"❌ Fichier non trouvé : {file_path}")
#             return
    
#     # Charger les données
#     predictions = pd.read_csv(args.predictions)
#     sample_data = pd.read_csv(args.data)
    
#     # Initialiser l'analyseur
#     analyzer = PotentialAnalyzer(args.model, args.reference)
    
#     # Effectuer l'analyse
#     results = analyzer.analyze_potential(sample_data, predictions)
    
#     # Afficher le rapport
#     report = analyzer.generate_report(results)
#     print(report)
    
#     # Exporter les résultats
#     analyzer.export_results(results, args.output)
    
#     # Sauvegarder le rapport
#     report_path = args.output.replace('.csv', '_report.txt')
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write(report)
#     print(f"📄 Rapport complet sauvegardé : {report_path}")


# if __name__ == "__main__":
#     main()