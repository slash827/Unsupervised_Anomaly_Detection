# Unsupervised Anomaly Detection on BETH Dataset

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project is a comprehensive study on **unsupervised anomaly detection** techniques applied to cybersecurity data, specifically the **BETH (Behavior-based Endpoint Threat Hunting) dataset**. It was developed as the final project for the Unsupervised Learning course at **Bar Ilan University**.

The project explores various machine learning approaches to detect malicious activities in system process logs without relying on labeled training data, making it applicable to real-world cybersecurity scenarios where labeled attack data is scarce.

## 🎯 Objectives

- **Primary Goal**: Develop and compare unsupervised anomaly detection methods for cybersecurity threat detection
- **Dataset**: BETH dataset containing Linux kernel process logs with labeled benign/suspicious/malicious activities
- **Techniques**: Traditional ML, deep learning autoencoders, clustering, and dimensionality reduction approaches
- **Evaluation**: Comprehensive performance analysis using multiple metrics and visualization techniques

## 📊 Dataset Information

### BETH Dataset
- **Source**: Kernel-level process monitoring logs from Linux systems
- **Size**: 1,141,078 records with 16 features
- **Classes**: 
  - Benign: 967,564 samples (84.79%)
  - Suspicious: 15,082 samples (1.32%)
  - Malicious: 158,432 samples (13.88%)

### Key Features
- `timestamp`: Time since system boot
- `processId`, `threadId`, `parentProcessId`: Process identifiers
- `userId`, `mountNamespace`: System context
- `processName`, `hostName`: Process and host information
- `eventId`, `eventName`: System call details
- `stackAddresses`, `argsNum`, `returnValue`, `args`: Execution details
- `sus`, `evil`: Target labels (suspicious/malicious indicators)

## 🚀 Quick Start

### Prerequisites
- Python 3.11.2 or compatible version
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/slash827/Unsupervised_Anomaly_Detection.git
cd Unsupervised_Anomaly_Detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**:
```bash
python main.py
```

**Alternative**: Download manually from [Kaggle BETH Dataset](https://www.kaggle.com/datasets/katehighnam/beth-dataset/data?select=labelled_training_data.csv) and extract to `data/` directory.

## 📁 Project Structure

### Core Modules

| File | Description | Key Features |
|------|-------------|--------------|
| **main.py** | Entry point and data downloader | Automated dataset acquisition from DropBox |
| **utils.py** | Data preprocessing utilities | Feature engineering, scaling, sampling functions |
| **eda.py** | Exploratory Data Analysis | Comprehensive statistical analysis and visualization |

### Anomaly Detection Methods

| Method | File | Approach | Key Capabilities |
|--------|------|----------|------------------|
| **Traditional ML** | `traditional_anomaly_detection.py` | Isolation Forest, One-Class SVM | Baseline anomaly detection methods |
| **Autoencoders** | `autoencoder.py` | Neural network reconstruction | Deep learning-based anomaly scoring |
| **Variational AE** | `variational_autoencoder.py` | VAE with KL divergence | Probabilistic latent space modeling |
| **PCA** | `pca.py` | Principal Component Analysis | Linear dimensionality reduction anomalies |

### Clustering & Analysis

| File | Purpose | Techniques |
|------|---------|------------|
| **prep_cluster.py** | Clustering data preparation | Feature engineering, MCA, balanced sampling |
| **full_analysis.py** | Comprehensive clustering pipeline | K-means, DBSCAN, GMM, Hierarchical, Fuzzy C-means |
| **cluster_visualizer.py** | Cluster visualization and investigation | PCA, t-SNE, UMAP projections |
| **dimension_reduction.py** | Dimensionality reduction analysis | UMAP, t-SNE with clustering evaluation |

### Specialized Analysis

| File | Focus | Methods |
|------|-------|---------|
| **dataset_explainability.py** | Model interpretability | SHAP analysis, decision trees, feature importance |
| **dataset_reduction.py** | Data efficiency | Smart sampling strategies, information loss measurement |

## 🔬 Methodology

### 1. Data Preprocessing
- **Feature Engineering**: Binary system indicators, statistical features
- **Normalization**: StandardScaler for numerical features
- **Categorical Encoding**: MCA (Multiple Correspondence Analysis) for categorical data
- **Class-aware Sampling**: Stratified sampling maintaining class distributions

### 2. Anomaly Detection Approaches

#### Traditional Methods
- **Isolation Forest**: Tree-based anomaly detection
- **One-Class SVM**: Support vector-based outlier detection

#### Deep Learning Methods
- **Autoencoders**: Reconstruction error-based detection
- **Variational Autoencoders**: Probabilistic encoding with KL divergence loss

#### Unsupervised Learning
- **PCA Reconstruction**: Linear dimensionality reduction anomalies
- **Clustering-based**: Outlier detection using cluster distances

### 3. Evaluation Metrics
- **ROC-AUC**: Area under receiver operating characteristic curve
- **PR-AUC**: Area under precision-recall curve
- **F1-Score**: Harmonic mean of precision and recall
- **Silhouette Score**: Clustering quality measurement
- **Mutual Information**: Cluster-label correspondence

## 📈 Key Results

### Best Performing Methods
1. **Variational Autoencoder**: Achieved highest anomaly detection performance with combined reconstruction + KL divergence loss
2. **Isolation Forest**: Strong baseline performance with fast training
3. **DBSCAN Clustering**: Effective at identifying anomalous behavior patterns

### Feature Importance Findings
- **Process identifiers** (processId, parentProcessId) showed high discriminative power
- **System calls** (eventId, eventName) crucial for detecting malicious patterns
- **Temporal features** revealed time-based attack patterns
- **User context** (userId, mountNamespace) important for insider threat detection

### Performance Highlights
- **Detection Rate**: Up to 85%+ true positive rate on malicious samples
- **False Positives**: Maintained low false positive rates (< 5%) for production viability
- **Scalability**: Efficient processing of 1M+ records using optimized sampling strategies

## 📊 Visualizations & Analysis

### Generated Outputs

#### EDA Outputs (`eda_output/`)
- Statistical distributions of all features
- Correlation matrices and feature relationships
- Target variable analysis and class imbalance visualization
- User behavior patterns and temporal analysis
- Feature importance rankings

#### Algorithm-Specific Results
- **ROC and PR curves** for all anomaly detection methods
- **Confusion matrices** with optimal threshold analysis
- **Latent space visualizations** for autoencoder methods
- **Cluster analysis** with silhouette scores and separation metrics
- **Feature contribution analysis** using SHAP values

#### Comparative Analysis
- **Method comparison** across multiple evaluation metrics
- **Hyperparameter optimization** results and trade-offs
- **Computational efficiency** analysis (training time vs. performance)

## 🛠️ Usage Examples

### Basic Anomaly Detection
```python
# Run comprehensive EDA
python eda.py

# Train and evaluate traditional methods
python traditional_anomaly_detection.py

# Deep learning approach
python variational_autoencoder.py

# Clustering analysis
python full_analysis.py
```

### Custom Analysis
```python
from utils import load_and_preprocess_beth_data
from autoencoder import Autoencoder

# Load and preprocess data
df, features = load_and_preprocess_beth_data(csv_files, "data/")

# Train custom autoencoder
autoencoder = Autoencoder(encoding_dim=10, max_iter=100)
results = autoencoder.fit_transform(df[features])
```

## 📋 Requirements

### Core Dependencies
```
pandas==2.1.4          # Data manipulation and analysis
numpy==1.26.4           # Numerical computing
matplotlib==3.7.2       # Plotting and visualization
seaborn==0.12.2         # Statistical data visualization
scikit-learn==1.4.2     # Machine learning algorithms
scipy==1.11.4           # Scientific computing
```

### Specialized Libraries
```
scikit-fuzzy==0.5.0     # Fuzzy clustering methods
umap==0.1.1             # UMAP dimensionality reduction
shap==0.47.1            # Model explainability
prince==0.16.0          # Multiple Correspondence Analysis
kneed==0.8.5            # Knee point detection for clustering
joblib==1.3.2           # Model serialization
tqdm==4.66.1            # Progress bars
```

## 🔧 Advanced Configuration

### Hyperparameter Tuning
- **Autoencoder**: Adjust `encoding_dim`, `hidden_layers`, `max_iter`
- **Clustering**: Optimize `eps`, `min_samples` for DBSCAN
- **Sampling**: Configure `n_samples` for large dataset processing

### Performance Optimization
- **Memory Management**: Built-in data sampling for large datasets
- **Parallel Processing**: Multi-core support for clustering algorithms
- **Efficient Storage**: Automatic result caching and model serialization

## 📚 Academic Context

### Course Information
- **Institution**: Bar Ilan University
- **Course**: Unsupervised Learning
- **Focus**: Advanced machine learning techniques for pattern discovery
- **Application Domain**: Cybersecurity and threat detection

### Research Contributions
- **Comparative Study**: Systematic evaluation of multiple unsupervised methods
- **Feature Engineering**: Domain-specific features for cybersecurity data
- **Scalability Analysis**: Techniques for handling large-scale security logs
- **Interpretability**: Focus on explainable AI for security applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Additional anomaly detection algorithms
- Performance optimizations
- Visualization improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BETH Dataset**: Thanks to the creators of the BETH dataset for providing high-quality cybersecurity data
- **Bar Ilan University**: For the academic framework and research support
- **Open Source Community**: For the excellent machine learning libraries that made this project possible

## 📞 Contact

For questions or collaboration opportunities, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and research purposes. The anomaly detection methods explored here provide a foundation for understanding unsupervised learning in cybersecurity contexts. 