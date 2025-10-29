# ReML Centralized Benchmarking System

## Overview

The ReML centralized benchmarking system eliminates code duplication across algorithm implementations by providing standardized utilities for:

- **Dataset Loading**: Consistent dataset management across benchmarks
- **Model Comparison**: Automated comparison between ReML and sklearn implementations  
- **Results Management**: Standardized saving, loading, and analysis of results
- **Visualization**: Professional plots and reports with VS Code compatibility

## 🚀 Quick Start

```python
from reml.benchmarking import (
    BenchmarkRunner, BenchmarkConfig, 
    get_standard_datasets, BenchmarkVisualizer
)

# Configure benchmark
config = BenchmarkConfig(
    algorithm_name="KNN",
    param_name="k",                     # ReML parameter name
    param_values=[3, 5, 7, 9, 11],
    sklearn_param_name="n_neighbors",   # sklearn parameter name
    additional_params={'weights': 'uniform'},
    sklearn_additional_params={'weights': 'uniform'}
)

# Load datasets
datasets = get_standard_datasets('classification', max_samples=1000)

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run_benchmark(ReMLClass, SklearnClass, datasets)

# Create visualizations
visualizer = BenchmarkVisualizer()
visualizer.create_comprehensive_report(results, "KNN", "k", "plots/")
```

## 📊 Core Components

### 1. BenchmarkConfig

Configures all aspects of the benchmark:

```python
config = BenchmarkConfig(
    algorithm_name="DecisionTree",
    param_name="max_depth",
    param_values=[3, 5, 7, 10, None],
    n_trials=3,
    test_size=0.3,
    random_state=42,
    additional_params={
        'criterion': 'gini',
        'min_samples_split': 2
    },
    # Handle different parameter names between ReML and sklearn
    sklearn_param_name="max_depth",  # Usually same, but can differ
    sklearn_additional_params={
        'criterion': 'gini',
        'min_samples_split': 2,
        'random_state': 42
    }
)
```

### 2. Dataset Management

```python
from reml.benchmarking import get_standard_datasets, create_custom_dataset

# Load standard datasets
datasets = get_standard_datasets(
    task_type='classification',    # 'classification' or 'regression'
    include_synthetic=True,
    include_real=True,
    max_samples=1000              # Limit for faster testing
)

# Create custom dataset
custom_dataset = create_custom_dataset(
    dataset_type='classification',
    n_samples=500,
    n_features=10,
    n_classes=3
)
```

### 3. Preprocessing

```python
from reml.benchmarking import create_standard_preprocessor

# Automatic preprocessing based on algorithm type
preprocessor = create_standard_preprocessor('knn')        # StandardScaler
preprocessor = create_standard_preprocessor('svm')        # StandardScaler  
preprocessor = create_standard_preprocessor('tree')       # None
preprocessor = create_standard_preprocessor('naive_bayes') # None
```

### 4. Results Management

```python
from reml.benchmarking import ResultsManager

# Save results in multiple formats
manager = ResultsManager("results/benchmarks")
saved_files = manager.save_results(
    results=results,
    algorithm_name="KNN",
    metadata={'config': config.__dict__},
    save_formats=['json', 'csv', 'excel']
)

# Load and compare multiple algorithms
comparison_df = manager.compare_algorithms([
    'results/knn_results.json',
    'results/tree_results.json'
])
```

### 5. Visualization

```python
from reml.benchmarking import BenchmarkVisualizer

visualizer = BenchmarkVisualizer(figsize=(12, 8))

# Individual plots
fig1 = visualizer.plot_accuracy_comparison(results, "KNN")
fig2 = visualizer.plot_performance_comparison(results, "KNN")
fig3 = visualizer.plot_parameter_sensitivity(results, "KNN", "k")
fig4 = visualizer.plot_summary_heatmap(results, "KNN")

# Comprehensive report (creates all plots)
plot_files = visualizer.create_comprehensive_report(
    results=results,
    algorithm_name="KNN",
    param_name="k",
    output_dir="reports/benchmarks/plots",
    show_plots=False  # Save but don't display
)
```

## 🔧 Parameter Mapping

The system handles different parameter names between ReML and sklearn:

| ReML Parameter | sklearn Parameter | Example Usage |
|---------------|-------------------|---------------|
| `k` | `n_neighbors` | KNeighborsClassifier |
| `max_depth` | `max_depth` | DecisionTree |
| `C` | `C` | LogisticRegression |

```python
# KNN Example with parameter mapping
config = BenchmarkConfig(
    algorithm_name="KNN",
    param_name="k",                    # ReML uses 'k'
    param_values=[3, 5, 7, 9],
    sklearn_param_name="n_neighbors",  # sklearn uses 'n_neighbors'
    additional_params={'weights': 'uniform'},
    sklearn_additional_params={'weights': 'uniform'}
)
```

## 📈 Example Results

The system provides comprehensive analysis:

```
🎯 BENCHMARK SUMMARY
========================================
iris:
  ReML Accuracy: 0.9526
  Sklearn Accuracy: 0.9526
  Accuracy Diff: 0.0000
  Speed Ratio: 11.91x

🔍 OVERALL STATISTICS:
   Average Accuracy Difference: 0.0007
   Max Accuracy Difference: 0.0033
   Average Speed Ratio: 91.33x
```

## 📁 Output Structure

```
results/benchmarks/
├── KNN_benchmark_20251029_230825.json
├── KNN_benchmark_20251029_230825.csv
└── KNN_benchmark_20251029_230825.xlsx

reports/benchmarks/
├── KNN_benchmark_report.txt
└── plots/
    ├── KNN_accuracy_comparison.png
    ├── KNN_performance_analysis.png
    ├── KNN_parameter_sensitivity.png
    └── KNN_summary_heatmap.png
```

## 🎨 Visualization Features

- **Accuracy Comparison**: Bar charts comparing ReML vs sklearn accuracy
- **Performance Analysis**: Combined accuracy and timing plots
- **Parameter Sensitivity**: How parameters affect performance across datasets
- **Summary Heatmap**: Comprehensive overview of all results
- **VS Code Compatibility**: Optimized for VS Code's Simple Browser

## 🔍 Advanced Usage

### Custom Datasets

```python
from reml.benchmarking import DatasetManager

manager = DatasetManager(random_state=42)

# Load specific dataset types
classification_datasets = manager.load_classification_datasets(
    include_synthetic=True,
    include_real=False,
    max_samples=500
)

# Get dataset summary
summary = manager.get_dataset_summary(classification_datasets)
print(summary)
```

### Results Analysis

```python
from reml.benchmarking import load_and_compare

# Compare multiple algorithm results
comparison_df = load_and_compare([
    'results/knn_results.json',
    'results/tree_results.json',
    'results/svm_results.json'
])

# Analyze specific metrics
best_accuracy = comparison_df.groupby('algorithm')['reml_accuracy'].mean()
speed_comparison = comparison_df.groupby('algorithm')['speed_ratio'].mean()
```

## 🛠️ Migration Guide

### Before (Old Template System)
```python
# Duplicated code in each benchmark
def load_datasets():
    # 50+ lines of dataset loading code
    pass

def compare_models(X, y, param_value):
    # 30+ lines of model comparison code
    pass

def run_benchmark():
    # 100+ lines of benchmark orchestration
    pass
```

### After (Centralized System)
```python
# Clean, reusable code
config = BenchmarkConfig(...)
datasets = get_standard_datasets('classification')
runner = BenchmarkRunner(config)
results = runner.run_benchmark(ReMLClass, SklearnClass, datasets)
```

## 📚 Complete Examples

See these files for complete working examples:

- `experiments/benchmark/benchmark_knn_centralized.py` - KNN with parameter mapping
- `experiments/benchmark/benchmark_decision_tree_updated.py` - Decision Tree example
- `reml/benchmarking/` - Core implementation modules

## 🎯 Benefits

1. **Eliminated Code Duplication**: 200+ lines reduced to ~20 lines per benchmark
2. **Consistent Results**: Standardized metrics and evaluation procedures
3. **Professional Visualization**: Automated plot generation with consistent styling
4. **Multiple Output Formats**: JSON, CSV, Excel, and text reports
5. **Parameter Mapping**: Handles different API names between ReML and sklearn
6. **VS Code Integration**: Optimized for VS Code development environment
7. **Extensible Design**: Easy to add new algorithms and datasets

The centralized benchmarking system transforms ReML development from manual, error-prone processes to automated, professional-grade analysis workflows.