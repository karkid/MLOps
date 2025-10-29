# 🧠 ReML — Rebuilding Machine Learning from Scratch

**ReML** (Rebuild Machine Learning) is a comprehensive educational machine learning library that reimplements core algorithms from **Scikit-learn**, **NumPy**, and **SciPy** entirely from scratch using only basic Python.

**🎯 Mission**: Understand machine learning algorithms at their core by building them from first principles, comparing with industry standards, and creating a structured learning resource.

---

## ✨ Features

### 🤖 **Implemented Algorithms**
- **Classification**: K-Nearest Neighbors, Decision Trees, Logistic Regression
- **Regression**: Linear Regression  
- **Preprocessing**: StandardScaler, MinMaxScaler, Normalizer
- **Metrics**: Accuracy, Confusion Matrix
- **Distance Functions**: Euclidean, Manhattan, Cosine, Chebyshev, Canberra

### 📊 **Built-in Validation**
- **95% test coverage** with comprehensive unit tests
- **Benchmarking scripts** comparing against scikit-learn
- **Interactive notebooks** for algorithm visualization
- **Systematic experiments** for performance analysis

### 🛠️ **Development Tools**
- **Just commands** for streamlined development (`just test`, `just lint`, `just check`)
- **UV package management** for fast dependency resolution
- **Automated formatting** with Ruff
- **CI/CD ready** with comprehensive testing

---

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/karkid/ReML.git
cd ReML

# Setup environment (requires just and uv)
just init

# Run tests to verify installation
just test
```

### Basic Usage
```python
from reml.neighbors import KNeighborsClassifier
from reml.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
knn = KNeighborsClassifier(k=5)
knn.fit(X_scaled, y)

# Make predictions
predictions = knn.predict(X_scaled)
```  

## 📁 Project Structure

```bash
ReML/
├── 📦 reml/                           # Core library implementation
│   ├── linear_model/                  # LinearRegression, LogisticRegression
│   ├── neighbors/                     # KNeighborsClassifier
│   ├── tree/                          # DecisionTree
│   ├── preprocessing/                 # Scalers and normalizers
│   ├── spatial/                       # Distance functions
│   ├── utils/                         # Helper functions and decorators
│   └── metrics.py                     # Evaluation metrics
│
├── 📓 notebooks/                      # Interactive exploration & tutorials
│   ├── iris_knn_classification.ipynb # KNN demo on Iris dataset
│   └── decision_tree_visualization.ipynb # Tree visualization guide
│
├── 🧪 experiments/                    # Systematic testing & benchmarking
│   └── decision_tree_analysis/        # Decision tree performance comparison
│
├── 📄 reports/                        # Documentation & analysis results
│   └── algorithm_documentation/       # Detailed algorithm analysis
│
├── ✅ tests/                          # Comprehensive test suite (95% coverage)
├── 📊 data/                          # Sample datasets
└── 🛠️ Development files (Justfile, pyproject.toml, etc.)
```

## 🧠 Learning Workflow

### 1. 📓 **Explore** (`notebooks/`)
Start with interactive notebooks to understand algorithms:
```bash
# Open Jupyter Lab with all dependencies
just notebook-deps
jupyter lab notebooks/
```

### 2. 🧪 **Experiment** (`experiments/`)
Run systematic comparisons and benchmarks:
```bash
# Run decision tree benchmark
python experiments/decision_tree_analysis/benchmark_vs_sklearn.py
```

### 3. 📄 **Document** (`reports/`)
Review analysis and implementation details:
- Algorithm explanations
- Performance comparisons  
- Implementation insights

### 4. 🔧 **Develop** (`reml/`)
Implement new algorithms following established patterns:
```bash
just test        # Run all tests
just lint        # Format and lint code  
just check       # Full quality check
just coverage    # Generate coverage report
```

## 🔧 Development Commands

ReML uses [Just](https://just.systems/) for streamlined development:

```bash
just --list                    # Show all available commands

# Development
just init                      # Create virtual environment and install dependencies
just test                      # Run pytest test suite  
just lint                      # Run linting and formatting (Ruff)
just check                     # Run all quality checks (formatting, linting, tests)

# Analysis  
just coverage                  # Run tests with coverage report
just coverage-html             # Generate HTML coverage report

# Package Management
just add PACKAGE              # Add a new dependency
just remove PACKAGE           # Remove a dependency  
just update-deps              # Update all dependencies
just reinstall                # Clean reinstall of all dependencies

# Notebooks
just notebook-deps            # Install notebook dependencies
```

### Requirements
- **Python** ≥ 3.10
- **Just** (command runner) - `brew install just` 
- **UV** (package manager) - `brew install uv`

## 🎯 Current Status

### ✅ **Completed**
- **5 core algorithms** implemented and tested
- **95% test coverage** with comprehensive unit tests
- **Interactive notebooks** for visualization and learning
- **Benchmarking framework** for performance comparison
- **Modern development setup** with UV, Just, and Ruff

### 🚧 **In Progress**  
- Decision tree visualization enhancements
- Additional distance metrics
- Performance optimization

### 📋 **Roadmap**
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Neural Networks**: Basic multi-layer perceptron
- **Clustering**: K-Means, DBSCAN
- **Dimensionality Reduction**: PCA, t-SNE
- **More Metrics**: Precision, Recall, F1-Score, ROC-AUC

## 🤝 Contributing

This is an educational project, but contributions are welcome! Please:

1. **Follow the workflow**: notebooks → experiments → reports
2. **Add comprehensive tests** for new algorithms
3. **Include benchmarks** against scikit-learn
4. **Document with examples** in notebooks

```bash
# Setup development environment
git clone https://github.com/karkid/ReML.git
cd ReML  
just init
just check  # Ensure everything works
```

## � License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Scikit-learn** team for the excellent API design patterns
- **NumPy** and **SciPy** communities for mathematical foundations  
- **UV** and **Just** for modern Python tooling
- All the educational resources that made this learning journey possible

## 🧑‍💻 Author

**Dheeraj Karki** - Passionate about AI, machine learning, and interdisciplinary research.

*"The best way to understand something deeply is to build it from scratch."* 🧩

---

⭐ **Star this repo** if you find it helpful for learning ML algorithms!  
🐛 **Report issues** or suggest improvements via GitHub Issues  
🤝 **Contribute** by following the established workflow