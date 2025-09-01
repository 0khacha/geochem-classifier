#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse complète des relations entre l'argent (Ag) et les autres éléments/oxydes.

Fonctionnalités :
- Chargement du dataset (CSV préprocessé recommandé)
- Détection automatique de la colonne Ag ("Ag" ou "Ag (g/t)")
- Corrélations (Pearson & Spearman) Ag vs autres variables
- Visualisations :
    * Heatmap (colonne Ag)
    * Bar chart des top corrélations absolues
    * Scatter plots Ag vs top 6 variables corrélées (coloré par Type si dispo)
- Régression RandomForestRegressor pour expliquer Ag
    * R², RMSE
    * Importances des variables (bar chart)
- (Optionnel) Régression Lasso pour obtenir des coefficients signés (direction)
- Extraction des minimums et seuils par type (minerai, etc.)
- Export des résultats dans le dossier results/

Utilisation :
    python analyze_ag.py \
        --data dataset/processed/BDD_ICP_balanced.csv \
        --outdir results_ag \
        --with-lasso

Prérequis : pandas, numpy, scikit-learn, matplotlib

Note : Si vous voulez partir des données brutes Excel, utilisez votre
       pipeline preprocess.py pour générer le CSV équilibré d'abord.
"""

import os
import argparse
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Utilitaires
# -----------------------------

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_ag_column(df: pd.DataFrame) -> str:
    """Trouve la colonne Ag de manière robuste ("Ag" ou "Ag (g/t)")."""
    candidates = [c for c in df.columns if c.strip().lower() in {"ag", "ag (g/t)", "ag g/t"}]
    if candidates:
        return candidates[0]
    # fallback : contient "ag" et pas "mag"/"crag" etc.
    for c in df.columns:
        cl = c.strip().lower()
        if cl.startswith("ag") or cl == "ag (ppm)" or "ag (g/t)" in cl:
            return c
    raise ValueError("Impossible de localiser la colonne de l'argent (Ag). Renommez-la en 'Ag' ou 'Ag (g/t)'.")


def find_type_column(df: pd.DataFrame) -> str:
    for cand in ["Type", "type", "class", "label"]:
        if cand in df.columns:
            return cand
    return ""


def numeric_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in nums if c not in exclude]


# -----------------------------
# Corrélations
# -----------------------------

def compute_correlations(df: pd.DataFrame, ag_col: str, outdir: str) -> Tuple[pd.Series, pd.Series]:
    pearson = df.corr(numeric_only=True)[ag_col].drop(labels=[ag_col]).sort_values(ascending=False)
    spearman = df.corr(method="spearman", numeric_only=True)[ag_col].drop(labels=[ag_col]).sort_values(ascending=False)

    pearson.to_csv(os.path.join(outdir, "correlations_pearson_Ag.csv"))
    spearman.to_csv(os.path.join(outdir, "correlations_spearman_Ag.csv"))

    # Heatmap 1-col (Ag)
    cor = df.corr(numeric_only=True)[[ag_col]].sort_values(by=ag_col, ascending=False)
    plt.figure(figsize=(4, max(4, len(cor) * 0.3)))
    plt.imshow(cor.values, aspect='auto')
    plt.colorbar(label='Corrélation (Pearson)')
    plt.yticks(range(len(cor.index)), cor.index)
    plt.xticks([0], [ag_col])
    plt.title("Corrélations avec Ag")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heatmap_Ag.png"), dpi=200)
    plt.close()

    # Top corrélations absolues
    abs_sorted = pearson.abs().sort_values(ascending=False)
    top = abs_sorted.head(min(20, len(abs_sorted)))
    plt.figure(figsize=(8, max(4, len(top) * 0.35)))
    plt.barh(top.index[::-1], top.values[::-1])
    plt.title("Top corrélations absolues avec Ag (Pearson)")
    plt.xlabel("|corr|")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_abs_correlations_Ag.png"), dpi=200)
    plt.close()

    return pearson, spearman


# -----------------------------
# Visualisations de dispersion
# -----------------------------

def scatter_top_pairs(df: pd.DataFrame, ag_col: str, type_col: str, pearson: pd.Series, outdir: str, k: int = 6) -> None:
    # choisir top k par corrélation absolue
    candidates = pearson.abs().sort_values(ascending=False).head(k).index.tolist()
    # Préparer couleurs par type si dispo
    colors = None
    legend = None
    if type_col and type_col in df.columns:
        uniq = sorted(df[type_col].dropna().unique().tolist())
        color_map = {v: i for i, v in enumerate(uniq)}
        colors = df[type_col].map(color_map)
        legend = [(v, i) for i, v in enumerate(uniq)]

    n = len(candidates)
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(5*cols, 4*rows))
    for i, feat in enumerate(candidates, 1):
        ax = plt.subplot(rows, cols, i)
        if colors is None:
            ax.scatter(df[feat], df[ag_col], s=12, alpha=0.7)
        else:
            sc = ax.scatter(df[feat], df[ag_col], c=colors, s=12, alpha=0.7)
        ax.set_xlabel(feat)
        ax.set_ylabel(ag_col)
        ax.set_title(f"Ag vs {feat}")
    if legend is not None:
        handles = [plt.Line2D([0], [0], marker='o', linestyle='', label=str(v)) for v, _ in legend]
        plt.legend(handles=handles, title=type_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_Ag_top_features.png"), dpi=200)
    plt.close()


# -----------------------------
# Régression pour expliquer Ag
# -----------------------------

def regression_rf(df: pd.DataFrame, ag_col: str, type_col: str, outdir: str) -> pd.DataFrame:
    features = [c for c in df.columns if c not in [ag_col, type_col] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[features]
    y = df[ag_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    with open(os.path.join(outdir, "rf_regression_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"n_train: {len(X_train)}, n_test: {len(X_test)}\n")

    imp = pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
    imp.to_csv(os.path.join(outdir, "rf_feature_importance_for_Ag.csv"), index=False)

    top = imp.head(min(15, len(imp)))
    plt.figure(figsize=(8, max(4, len(top)*0.35)))
    plt.barh(top["Feature"][::-1], top["Importance"][::-1])
    plt.title("Importances (RandomForest) pour expliquer Ag")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rf_feature_importance_for_Ag.png"), dpi=200)
    plt.close()

    return imp


def regression_lasso(df: pd.DataFrame, ag_col: str, type_col: str, outdir: str) -> pd.DataFrame:
    features = [c for c in df.columns if c not in [ag_col, type_col] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[features].values
    y = df[ag_col].values

    lasso = LassoCV(alphas=None, cv=5, random_state=42, max_iter=10000)
    lasso.fit(X, y)

    coefs = pd.DataFrame({"Feature": features, "Coef": lasso.coef_}).sort_values("Coef", ascending=False)
    coefs.to_csv(os.path.join(outdir, "lasso_coefficients_for_Ag.csv"), index=False)

    k = min(12, len(coefs))
    top_pos = coefs.sort_values("Coef", ascending=False).head(k//2)
    top_neg = coefs.sort_values("Coef", ascending=True).head(k//2)
    top_combined = pd.concat([top_neg, top_pos])

    plt.figure(figsize=(8, max(4, len(top_combined)*0.4)))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in top_combined["Coef"].values]
    plt.barh(top_combined["Feature"], top_combined["Coef"], color=colors)
    plt.title("Lasso : coefficients (signés) pour Ag")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lasso_top_signed_coeffs_for_Ag.png"), dpi=200)
    plt.close()

    with open(os.path.join(outdir, "lasso_alpha.txt"), "w", encoding="utf-8") as f:
        f.write(f"Alpha choisi: {lasso.alpha_:.6f}\n")

    return coefs


# -----------------------------
# Extraction des minimums par type
# -----------------------------

def compute_minimums(df: pd.DataFrame, type_col: str, outdir: str) -> pd.DataFrame:
    if not type_col:
        print("⚠️ Pas de colonne de type trouvée, impossible de calculer les seuils par type.")
        return pd.DataFrame()

    results = {}
    for t, subdf in df.groupby(type_col):
        results[t] = subdf.min(numeric_only=True)
        results[f"{t}_p10"] = subdf.quantile(0.10, numeric_only=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(outdir, "minimums_by_type.csv"))
    return res_df


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse des relations entre Ag et les autres variables.")
    parser.add_argument("--data", type=str, default="dataset/processed/BDD_ICP_balanced.csv", help="Chemin du CSV préprocessé")
    parser.add_argument("--outdir", type=str, default="results_ag", help="Dossier de sortie")
    parser.add_argument("--with-lasso", action="store_true", help="Inclure une régression Lasso (coefficients signés)")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    df = pd.read_csv(args.data)

    ag_col = find_ag_column(df)
    type_col = find_type_column(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[ag_col])
    for c in df.columns:
        if c != ag_col and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())

    pearson, spearman = compute_correlations(df, ag_col, args.outdir)

    scatter_top_pairs(df, ag_col, type_col, pearson, args.outdir, k=6)

    imp_rf = regression_rf(df, ag_col, type_col, args.outdir)

    if args.with_lasso:
        _ = regression_lasso(df, ag_col, type_col, args.outdir)

    res_min = compute_minimums(df, type_col, args.outdir)

    with open(os.path.join(args.outdir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("# Résultats — Analyse Ag\n\n")
        f.write("## Corrélations (Pearson — top 10)\n\n")
        f.write(pearson.abs().sort_values(ascending=False).head(10).to_string())
        f.write("\n\n## Minimums par type (bruts et p10)\n\n")
        if not res_min.empty:
            f.write(res_min.to_string())
        f.write("\n\n## Fichiers générés\n\n")
        f.write("- correlations_pearson_Ag.csv\n")
        f.write("- correlations_spearman_Ag.csv\n")
        f.write("- heatmap_Ag.png\n")
        f.write("- top_abs_correlations_Ag.png\n")
        f.write("- scatter_Ag_top_features.png\n")
        f.write("- rf_regression_metrics.txt\n")
        f.write("- rf_feature_importance_for_Ag.csv\n")
        f.write("- rf_feature_importance_for_Ag.png\n")
        if args.with_lasso:
            f.write("- lasso_coefficients_for_Ag.csv\n")
            f.write("- lasso_top_signed_coeffs_for_Ag.png\n")
            f.write("- lasso_alpha.txt\n")
        f.write("- minimums_by_type.csv\n")

    print("✅ Analyse terminée. Résultats dans:", args.outdir)


if __name__ == "__main__":
    main()
