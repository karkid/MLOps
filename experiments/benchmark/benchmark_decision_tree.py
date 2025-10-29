#!/usr/bin/env python3
"""
Decision Tree Benchmark: ReML vs Scikit-learn

This script performs systematic comparison between ReML and sklearn
decision tree implementations across multiple datasets and parameters.

Usage:
    python benchmark_vs_sklearn.py --output results.json

Author: ReML Project
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import json
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import our implementation
from reml.tree import DecisionTree
from reml.metrics import accuracy_score as reml_accuracy


def load_datasets():
    """Load multiple datasets for comprehensive testing"""
    datasets_dict = {}
    
    # Iris dataset
    iris = datasets.load_iris()
    datasets_dict['iris'] = {
        'X': iris.data,
        'y': iris.target,
        'name': 'Iris'
    }
    
    # Wine dataset
    wine = datasets.load_wine()
    datasets_dict['wine'] = {
        'X': wine.data,
        'y': wine.target,
        'name': 'Wine'
    }
    
    # Breast cancer dataset
    cancer = datasets.load_breast_cancer()
    datasets_dict['breast_cancer'] = {
        'X': cancer.data,
        'y': cancer.target,
        'name': 'Breast Cancer'
    }
    
    return datasets_dict


def benchmark_single_dataset(dataset_name, X, y, max_depths=[3, 5, 7, 10]):
    """
    Benchmark ReML vs sklearn on a single dataset
    
    Returns:
        dict: Results containing accuracy and timing comparisons
    """
    results = {
        'dataset': dataset_name,
        'max_depths': {},
        'summary': {}
    }
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    reml_accuracies = []
    sklearn_accuracies = []
    reml_times = []
    sklearn_times = []
    
    for max_depth in max_depths:
        print(f"  Testing max_depth={max_depth}")
        
        # Test ReML implementation
        start_time = time.time()
        reml_tree = DecisionTree(min_samples_split=2, max_depth=max_depth)
        reml_tree.fit(X_train, y_train)
        reml_pred = reml_tree.predict(X_test)
        reml_time = time.time() - start_time
        reml_acc = reml_accuracy(y_test, reml_pred)
        
        # Test sklearn implementation
        start_time = time.time()
        sklearn_tree = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_split=2, random_state=42
        )
        sklearn_tree.fit(X_train, y_train)
        sklearn_pred = sklearn_tree.predict(X_test)
        sklearn_time = time.time() - start_time
        sklearn_acc = accuracy_score(y_test, sklearn_pred)
        
        # Store results
        results['max_depths'][max_depth] = {
            'reml': {'accuracy': reml_acc, 'time': reml_time},
            'sklearn': {'accuracy': sklearn_acc, 'time': sklearn_time},
            'accuracy_diff': abs(reml_acc - sklearn_acc)
        }
        
        reml_accuracies.append(reml_acc)
        sklearn_accuracies.append(sklearn_acc)
        reml_times.append(reml_time)
        sklearn_times.append(sklearn_time)
    
    # Summary statistics
    results['summary'] = {
        'avg_accuracy_reml': np.mean(reml_accuracies),
        'avg_accuracy_sklearn': np.mean(sklearn_accuracies),
        'avg_time_reml': np.mean(reml_times),
        'avg_time_sklearn': np.mean(sklearn_times),
        'max_accuracy_diff': max([results['max_depths'][d]['accuracy_diff'] 
                                 for d in max_depths])
    }
    
    return results


def main():
    """Run comprehensive benchmark"""
    print("🌳 Decision Tree Benchmark: ReML vs Scikit-learn")
    print("=" * 50)
    
    # Load datasets
    datasets_dict = load_datasets()
    all_results = []
    
    # Run benchmarks
    for dataset_name, dataset_info in datasets_dict.items():
        print(f"\n📊 Testing on {dataset_info['name']} dataset...")
        
        results = benchmark_single_dataset(
            dataset_info['name'], 
            dataset_info['X'], 
            dataset_info['y']
        )
        all_results.append(results)
        
        # Print summary
        summary = results['summary']
        print(f"  ✅ Avg Accuracy - ReML: {summary['avg_accuracy_reml']:.3f}, "
              f"Sklearn: {summary['avg_accuracy_sklearn']:.3f}")
        print(f"  ⏱️  Avg Time - ReML: {summary['avg_time_reml']:.4f}s, "
              f"Sklearn: {summary['avg_time_sklearn']:.4f}s")
    
    # Save results
    output_file = './decision_tree_vs_sklearn.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print("📋 Next: Generate report in reports/")


if __name__ == "__main__":
    main()