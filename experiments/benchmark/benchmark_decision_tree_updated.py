#!/usr/bin/env python3
"""
Decision Tree Benchmark - Updated with Centralized Utilities

Uses the new reml.benchmarking utilities to eliminate code duplication
and provide standardized comparison against sklearn.

Usage:
    python benchmark_decision_tree_updated.py

Author: ReML Project
"""

import sys
from pathlib import Path

# Add ReML to path
reml_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(reml_path))

# Core imports using centralized utilities
from reml.benchmarking import (
    BenchmarkRunner, BenchmarkConfig, 
    get_standard_datasets, ResultsManager, 
    BenchmarkVisualizer
)

# Algorithm-specific imports
from reml.tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier

def main():
    """Main benchmark execution function"""
    print("🚀 Decision Tree Benchmark (Updated)")
    print("=" * 50)
    
    # Configure benchmark
    config = BenchmarkConfig(
        algorithm_name="DecisionTree",
        param_name="max_depth",
        param_values=[3, 5, 7, 10, None],
        n_trials=3,
        test_size=0.3,
        random_state=42,
        additional_params={
            'criterion': 'gini',
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
    )
    
    print(f"📋 Configuration:")
    print(f"   Parameter: {config.param_name}")
    print(f"   Values: {config.param_values}")
    print(f"   Trials per test: {config.n_trials}")
    print(f"   Additional params: {config.additional_params}")
    
    # Load datasets
    print(f"\n📊 Loading datasets...")
    datasets = get_standard_datasets(
        task_type='classification',
        include_synthetic=True,
        include_real=True,
        max_samples=1000  # Limit for faster testing
    )
    
    print(f"   Loaded {len(datasets)} datasets:")
    for name, dataset in datasets.items():
        print(f"   - {name}: {dataset['X'].shape}")
    
    # Decision trees don't need preprocessing
    print(f"\n🔧 No preprocessing needed for Decision Trees")
    
    # Initialize runner and execute benchmark
    print(f"\n⚡ Starting benchmark...")
    runner = BenchmarkRunner(config)
    
    try:
        results = runner.run_benchmark(
            reml_class=DecisionTree,
            sklearn_class=DecisionTreeClassifier,
            datasets=datasets,
            preprocessor=None  # Trees don't need scaling
        )
        
        print(f"\n✅ Benchmark completed successfully!")
        
        # Save results
        print(f"\n💾 Saving results...")
        results_manager = ResultsManager("results/benchmarks")
        saved_files = results_manager.save_results(
            results=results,
            algorithm_name=config.algorithm_name,
            metadata={
                'config': config.__dict__,
                'reml_class': 'DecisionTree',
                'sklearn_class': 'DecisionTreeClassifier',
                'datasets': list(datasets.keys()),
                'preprocessing': 'None - Trees handle raw features'
            },
            save_formats=['json', 'csv', 'excel']
        )
        
        # Generate text report
        print(f"\n📄 Generating report...")
        text_report = results_manager.generate_report(results, config.algorithm_name)
        
        report_path = Path("reports/benchmarks") / f"{config.algorithm_name}_benchmark_report.txt"
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w') as f:
            f.write(text_report)
        
        print(f"   Text report: {report_path}")
        
        # Create visualizations
        print(f"\n📊 Creating visualizations...")
        visualizer = BenchmarkVisualizer(figsize=(12, 8))
        
        plots_dir = Path("reports/benchmarks/plots")
        plot_files = visualizer.create_comprehensive_report(
            results=results,
            algorithm_name=config.algorithm_name,
            param_name=config.param_name,
            output_dir=plots_dir,
            show_plots=False  # Don't display, just save
        )
        
        # Print summary
        print(f"\n🎯 BENCHMARK SUMMARY")
        print(f"=" * 40)
        
        overall_accuracy_diffs = []
        overall_speed_ratios = []
        
        for dataset_result in results:
            summary = dataset_result['summary']
            dataset_name = dataset_result['dataset']
            
            print(f"{dataset_name}:")
            print(f"  ReML Accuracy: {summary['avg_accuracy_reml']:.4f}")
            print(f"  Sklearn Accuracy: {summary['avg_accuracy_sklearn']:.4f}")
            print(f"  Accuracy Diff: {summary['avg_accuracy_diff']:.4f}")
            print(f"  Speed Ratio: {summary['speed_ratio']:.2f}x")
            print()
            
            overall_accuracy_diffs.append(summary['avg_accuracy_diff'])
            overall_speed_ratios.append(summary['speed_ratio'])
        
        # Overall statistics
        import numpy as np
        print(f"🔍 OVERALL STATISTICS:")
        print(f"   Average Accuracy Difference: {np.mean(overall_accuracy_diffs):.4f}")
        print(f"   Max Accuracy Difference: {np.max(overall_accuracy_diffs):.4f}")
        print(f"   Average Speed Ratio: {np.mean(overall_speed_ratios):.2f}x")
        
        print(f"\n📁 All results saved to:")
        print(f"   Results: results/benchmarks/")
        print(f"   Reports: reports/benchmarks/")
        print(f"   Plots: {plots_dir}/")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n🎉 {len(results)} dataset benchmarks completed successfully!")
        print(f"\nTo compare with old implementation:")
        print(f"   python experiments/benchmark/benchmark_decision_tree.py")
    else:
        print(f"\n💥 Benchmark execution failed!")
        sys.exit(1)