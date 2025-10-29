"""
Core benchmarking framework for ReML implementations.

Provides standardized benchmarking infrastructure that can be reused
across all ReML algorithms for consistent comparison against sklearn.
"""

import time
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    algorithm_name: str
    param_name: str
    param_values: List[Any]
    n_trials: int = 3
    test_size: float = 0.3
    random_state: int = 42
    output_dir: Optional[str] = None
    additional_params: Dict[str, Any] = None
    # Parameter mapping for different API names
    sklearn_param_name: Optional[str] = None
    sklearn_additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}
        if self.sklearn_additional_params is None:
            self.sklearn_additional_params = {}
        # If no sklearn param name specified, assume same as ReML
        if self.sklearn_param_name is None:
            self.sklearn_param_name = self.param_name


class ModelComparator:
    """Handles comparison between ReML and sklearn implementations"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def compare_models(
        self,
        reml_class: Callable,
        sklearn_class: Callable,
        X: np.ndarray,
        y: np.ndarray,
        param_value: Any
    ) -> Dict[str, Any]:
        """
        Compare ReML and sklearn models for a specific parameter value
        
        Args:
            reml_class: ReML model class
            sklearn_class: Sklearn model class  
            X: Feature matrix
            y: Target vector
            param_value: Parameter value to test
            
        Returns:
            Dictionary with comparison results
        """
        from sklearn.model_selection import train_test_split
        from reml.metrics import accuracy_score
        
        results = {
            'reml': {'accuracies': [], 'times': []},
            'sklearn': {'accuracies': [], 'times': []},
            'param_value': param_value
        }
        
        for trial in range(self.config.n_trials):
            # Split data with different random state for each trial
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state + trial,
                stratify=y
            )
            
            # Test ReML implementation
            reml_result = self._test_model(
                reml_class, X_train, X_test, y_train, y_test, param_value, 'reml'
            )
            results['reml']['accuracies'].append(reml_result['accuracy'])
            results['reml']['times'].append(reml_result['time'])
            
            # Test sklearn implementation
            sklearn_result = self._test_model(
                sklearn_class, X_train, X_test, y_train, y_test, param_value, 'sklearn'
            )
            results['sklearn']['accuracies'].append(sklearn_result['accuracy'])
            results['sklearn']['times'].append(sklearn_result['time'])
        
        # Calculate averages and statistics
        results['reml']['avg_accuracy'] = np.mean(results['reml']['accuracies'])
        results['reml']['std_accuracy'] = np.std(results['reml']['accuracies'])
        results['reml']['avg_time'] = np.mean(results['reml']['times'])
        results['reml']['std_time'] = np.std(results['reml']['times'])
        
        results['sklearn']['avg_accuracy'] = np.mean(results['sklearn']['accuracies'])
        results['sklearn']['std_accuracy'] = np.std(results['sklearn']['accuracies'])
        results['sklearn']['avg_time'] = np.mean(results['sklearn']['times'])
        results['sklearn']['std_time'] = np.std(results['sklearn']['times'])
        
        results['accuracy_diff'] = abs(
            results['reml']['avg_accuracy'] - results['sklearn']['avg_accuracy']
        )
        results['speed_ratio'] = (
            results['reml']['avg_time'] / results['sklearn']['avg_time']
        )
        
        return results
    
    def _test_model(
        self,
        model_class: Callable,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        param_value: Any,
        model_type: str
    ) -> Dict[str, float]:
        """Test a single model and return results"""
        from reml.metrics import accuracy_score
        
        # Prepare parameters based on model type
        if model_type == 'reml':
            params = {self.config.param_name: param_value}
            params.update(self.config.additional_params)
        else:  # sklearn
            params = {self.config.sklearn_param_name: param_value}
            params.update(self.config.sklearn_additional_params)
            
            # Only add random_state if it's not already specified and the model supports it
            if 'random_state' not in params:
                # Check if the model class supports random_state parameter
                import inspect
                sig = inspect.signature(model_class.__init__)
                if 'random_state' in sig.parameters:
                    params['random_state'] = self.config.random_state
        
        # Train and test model
        start_time = time.time()
        model = model_class(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        end_time = time.time()
        
        # Calculate accuracy using ReML metrics for consistency
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'time': end_time - start_time
        }


class BenchmarkRunner:
    """Main benchmark runner that orchestrates the entire process"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.comparator = ModelComparator(config)
    
    def run_benchmark(
        self,
        reml_class: Callable,
        sklearn_class: Callable,
        datasets: Dict[str, Dict[str, np.ndarray]],
        preprocessor: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Run complete benchmark across datasets and parameters
        
        Args:
            reml_class: ReML model class
            sklearn_class: Sklearn model class
            datasets: Dictionary of datasets {name: {'X': X, 'y': y, ...}}
            preprocessor: Optional preprocessing function
            
        Returns:
            List of benchmark results for each dataset
        """
        print(f"🚀 Starting {self.config.algorithm_name} Benchmark...")
        print("=" * 60)
        
        all_results = []
        
        for dataset_name, dataset_info in datasets.items():
            print(f"\n📊 Testing dataset: {dataset_name}")
            print(f"    Shape: {dataset_info['X'].shape}")
            print(f"    Classes: {len(np.unique(dataset_info['y']))}")
            
            # Apply preprocessing if provided
            X = dataset_info['X']
            y = dataset_info['y']
            
            if preprocessor:
                X = preprocessor(X)
            
            # Run parameter sweep
            dataset_results = self._run_parameter_sweep(
                reml_class, sklearn_class, X, y, dataset_name, dataset_info
            )
            
            all_results.append(dataset_results)
            
            # Print summary
            summary = dataset_results['summary']
            print(f"    ✅ ReML Accuracy: {summary['avg_accuracy_reml']:.3f}")
            print(f"    ✅ Sklearn Accuracy: {summary['avg_accuracy_sklearn']:.3f}")
            print(f"    📈 Max Accuracy Diff: {summary['max_accuracy_diff']:.3f}")
            print(f"    ⚡ Speed Ratio: {summary['speed_ratio']:.1f}x")
        
        return all_results
    
    def _run_parameter_sweep(
        self,
        reml_class: Callable,
        sklearn_class: Callable,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str,
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run parameter sweep for a single dataset"""
        
        results = {
            'dataset': dataset_name,
            'shape': X.shape,
            'n_classes': len(np.unique(y)),
            'target_names': dataset_info.get('target_names', []),
            'param_values': {},
            'summary': {}
        }
        
        all_reml_accuracies = []
        all_sklearn_accuracies = []
        all_reml_times = []
        all_sklearn_times = []
        all_accuracy_diffs = []
        
        for param_value in self.config.param_values:
            print(f"    Testing {self.config.param_name}={param_value}...")
            
            # Compare models for this parameter value
            comparison_result = self.comparator.compare_models(
                reml_class, sklearn_class, X, y, param_value
            )
            
            # Store detailed results
            results['param_values'][str(param_value)] = {
                'reml': {
                    'accuracy': comparison_result['reml']['avg_accuracy'],
                    'time': comparison_result['reml']['avg_time'],
                    'std_accuracy': comparison_result['reml']['std_accuracy'],
                    'std_time': comparison_result['reml']['std_time']
                },
                'sklearn': {
                    'accuracy': comparison_result['sklearn']['avg_accuracy'],
                    'time': comparison_result['sklearn']['avg_time'],
                    'std_accuracy': comparison_result['sklearn']['std_accuracy'],
                    'std_time': comparison_result['sklearn']['std_time']
                },
                'accuracy_diff': comparison_result['accuracy_diff'],
                'speed_ratio': comparison_result['speed_ratio']
            }
            
            # Collect for summary
            all_reml_accuracies.append(comparison_result['reml']['avg_accuracy'])
            all_sklearn_accuracies.append(comparison_result['sklearn']['avg_accuracy'])
            all_reml_times.append(comparison_result['reml']['avg_time'])
            all_sklearn_times.append(comparison_result['sklearn']['avg_time'])
            all_accuracy_diffs.append(comparison_result['accuracy_diff'])
        
        # Calculate summary statistics
        results['summary'] = {
            'avg_accuracy_reml': np.mean(all_reml_accuracies),
            'avg_accuracy_sklearn': np.mean(all_sklearn_accuracies),
            'avg_time_reml': np.mean(all_reml_times),
            'avg_time_sklearn': np.mean(all_sklearn_times),
            'max_accuracy_diff': np.max(all_accuracy_diffs),
            'min_accuracy_diff': np.min(all_accuracy_diffs),
            'avg_accuracy_diff': np.mean(all_accuracy_diffs),
            'speed_ratio': np.mean(all_reml_times) / np.mean(all_sklearn_times)
        }
        
        return results


def create_standard_preprocessor(algorithm_type: str) -> Optional[Callable]:
    """
    Create standard preprocessor for different algorithm types
    
    Args:
        algorithm_type: Type of algorithm ('knn', 'svm', 'tree', etc.)
        
    Returns:
        Preprocessing function or None
    """
    if algorithm_type.lower() in ['knn', 'svm', 'logistic']:
        from sklearn.preprocessing import StandardScaler
        
        def preprocess(X):
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        
        return preprocess
    
    elif algorithm_type.lower() in ['tree', 'forest', 'naive_bayes']:
        # Tree-based algorithms don't need scaling
        return None
    
    else:
        # Default: no preprocessing
        return None