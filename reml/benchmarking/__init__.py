"""
Centralized benchmarking utilities for ReML implementations.

This package provides standardized benchmarking infrastructure that eliminates
code duplication across algorithm implementations and ensures consistent
comparison against sklearn.

Key Components:
- BenchmarkRunner: Main orchestrator for running benchmarks
- ModelComparator: Handles model comparison logic
- DatasetManager: Loads and manages standard datasets
- ResultsManager: Saves, loads, and analyzes results
- BenchmarkVisualizer: Creates standardized plots and reports

Usage Example:
    ```python
    from reml.benchmarking import (
        BenchmarkRunner, BenchmarkConfig, 
        get_standard_datasets, BenchmarkVisualizer
    )
    
    # Configure benchmark
    config = BenchmarkConfig(
        algorithm_name="KNN",
        param_name="n_neighbors", 
        param_values=[3, 5, 7, 9],
        n_trials=3
    )
    
    # Load datasets
    datasets = get_standard_datasets('classification')
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_benchmark(ReMLKNN, SklearnKNN, datasets)
    
    # Visualize results
    visualizer = BenchmarkVisualizer()
    visualizer.create_comprehensive_report(results, "KNN", "n_neighbors", "plots/")
    ```
"""

from .core import BenchmarkRunner, ModelComparator, BenchmarkConfig, create_standard_preprocessor
from .datasets import DatasetManager, get_standard_datasets, create_custom_dataset
from .results import ResultsManager, quick_save_results, load_and_compare
from .visualization import BenchmarkVisualizer, quick_plot_comparison, configure_vscode_matplotlib

__all__ = [
    # Core benchmarking
    'BenchmarkRunner',
    'ModelComparator', 
    'BenchmarkConfig',
    'create_standard_preprocessor',
    
    # Dataset management
    'DatasetManager',
    'get_standard_datasets',
    'create_custom_dataset',
    
    # Results management
    'ResultsManager',
    'quick_save_results',
    'load_and_compare',
    
    # Visualization
    'BenchmarkVisualizer',
    'quick_plot_comparison',
    'configure_vscode_matplotlib'
]

# Version info
__version__ = '1.0.0'