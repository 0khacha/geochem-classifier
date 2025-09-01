# Geochem Classifier 🧪⛏️

A **machine learning–powered geochemical classification tool** designed to analyze and predict the type of mining samples.

## 📋 Overview

This project leverages machine learning algorithms to automatically classify geochemical samples into three categories:
- **Sterile** – No economic interest
- **Potential** – Moderate mining potential
- **Ore** – High economic value

The system bases its predictions on **36 geochemical features**.

## 🚀 Key Features

- **Automatic classification** of geochemical samples  
- **Threshold analysis** to refine classification criteria  
- **Smart preprocessing** with handling of missing or censored values  
- **Algorithm benchmarking** (SVM, Random Forest, etc.)  
- **Real-time predictions** on new samples  
- **Detailed reports** with performance metrics  

## 📁 Project Structure

```
src/
├── train.py                  # Model training
├── infer.py                  # Predictions on new samples
├── preprocess.py             # Data preprocessing
├── analyze_ag.py             # Silver-focused analysis
├── analyze_thresholds.py     # Threshold optimization
├── geochemical_analysis.py   # General geochemical analyses
├── potential_analysis.py     # Analysis of potential samples
└── silver_analysis.py        # Silver-specific analysis
```

## 🛠️ Installation

### Requirements
- Python 3.8+  
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### Core dependencies
- `scikit-learn` – Machine learning algorithms  
- `pandas` – Data handling  
- `numpy` – Numerical computations  
- `joblib` – Model serialization  
- `matplotlib` – Data visualization  

## 💻 Usage

### 1. Train a model
```bash
python src/train.py
```
This trains multiple models and saves the best one under `models/geochem_pipeline.joblib`.

### 2. Run a prediction
```python
from src.infer import predict_sample

# Example input (36 geochemical values)
sample = [57.86, 16.46, 7.96, 0.92, 3.39, 3.29, 0.11, 0.74, 0.23, 2.77,
          5.5, 1400, 117, 1287, 0.9, 20, 4, 91, 119, 123, 10, 3, 49, 12,
          43, 68, 65, 1.76, 9, 40, 20, 110, 23, 17, 164, 4.46]

prediction = predict_sample(sample)
print(f"Classification: {prediction}")
```

### 3. Run threshold analysis
```bash
python src/analyze_thresholds.py
```

### 4. Specialized analyses
```bash
# General analysis
python src/geochemical_analysis.py

# Silver analysis
python src/silver_analysis.py

# Potential sample analysis
python src/potential_analysis.py
```

## 📊 Features Analyzed

The model uses **36 geochemical parameters**, including:

### Major elements (%)
SiO₂, Al₂O₃, Fe₂O₃, MgO, CaO, Na₂O, K₂O, TiO₂, P₂O₅, MnO

### Trace elements (ppm)
Cu, Pb, Zn, Ag, Au, As, Sb, Bi, Cd, Co, Cr, Ni, Mo, W, Sn, V, Ba, Sr, etc.

## 🎯 Model Performance

Typical results:
- **Accuracy**: >85%  
- **Recall**: >80%  
- **F1-score**: >82%  

Detailed metrics are stored in the training reports.

## 📋 Data Format

Samples must be provided as a **list of 36 values** in the exact expected order.  
The preprocessing step automatically:
- Converts decimal commas → points  
- Handles values like `<0.1` → converted to half (0.05)  
- Fills missing values → 0.0  

## 🔧 Configuration

Configurable parameters in the scripts:
- `use_raw_data=True` → Enables preprocessing of raw input  
- `test_size=0.3` → Share of data reserved for testing  
- `random_state=42` → Seed for reproducibility  

## 📈 Example Output

```
TEST - Ore Sample:
Normalized data:
Min: -1.234
Max: 2.156
Mean: 0.045

Class probabilities:
  Sterile:   0.123
  Potential: 0.234
  Ore:       0.643

Result: Ore
```

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/new-feature`)  
3. Commit your changes (`git commit -m 'Add new feature'`)  
4. Push your branch (`git push origin feature/new-feature`)  
5. Open a Pull Request  

## 📝 License

This project is under the **MIT License**. See the `LICENSE` file for details.

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ If you find this project useful, don’t forget to give it a star!
