"""
Standard dataset utilities for benchmarking.

Provides consistent dataset loading and management functionality
across all ReML algorithm benchmarks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn import datasets
from sklearn.model_selection import train_test_split


class DatasetManager:
    """Manages loading and preparation of standard benchmarking datasets"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def load_classification_datasets(
        self,
        include_synthetic: bool = True,
        include_real: bool = True,
        max_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load standard classification datasets for benchmarking
        
        Args:
            include_synthetic: Include synthetic datasets
            include_real: Include real-world datasets
            max_samples: Maximum samples per dataset (for speed)
            
        Returns:
            Dictionary of datasets with metadata
        """
        datasets_dict = {}
        
        if include_real:
            # Iris dataset
            iris = datasets.load_iris()
            datasets_dict['iris'] = {
                'X': iris.data,
                'y': iris.target,
                'target_names': iris.target_names.tolist(),
                'feature_names': iris.feature_names,
                'description': 'Classic iris flower classification'
            }
            
            # Wine dataset
            wine = datasets.load_wine()
            datasets_dict['wine'] = {
                'X': wine.data,
                'y': wine.target,
                'target_names': wine.target_names.tolist(),
                'feature_names': wine.feature_names,
                'description': 'Wine classification based on chemical analysis'
            }
            
            # Breast cancer dataset
            cancer = datasets.load_breast_cancer()
            datasets_dict['breast_cancer'] = {
                'X': cancer.data,
                'y': cancer.target,
                'target_names': cancer.target_names.tolist(),
                'feature_names': cancer.feature_names,
                'description': 'Breast cancer diagnosis classification'
            }
        
        if include_synthetic:
            # Synthetic classification dataset
            X_syn, y_syn = datasets.make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=3,
                random_state=self.random_state
            )
            datasets_dict['synthetic_classification'] = {
                'X': X_syn,
                'y': y_syn,
                'target_names': ['Class_0', 'Class_1', 'Class_2'],
                'feature_names': [f'feature_{i}' for i in range(20)],
                'description': 'Synthetic multi-class classification dataset'
            }
            
            # Synthetic binary classification
            X_bin, y_bin = datasets.make_classification(
                n_samples=800,
                n_features=10,
                n_informative=8,
                n_redundant=2,
                n_classes=2,
                random_state=self.random_state + 1
            )
            datasets_dict['synthetic_binary'] = {
                'X': X_bin,
                'y': y_bin,
                'target_names': ['Class_0', 'Class_1'],
                'feature_names': [f'feature_{i}' for i in range(10)],
                'description': 'Synthetic binary classification dataset'
            }
        
        # Apply sample limit if specified
        if max_samples:
            for name, dataset in datasets_dict.items():
                if dataset['X'].shape[0] > max_samples:
                    indices = np.random.RandomState(self.random_state).choice(
                        dataset['X'].shape[0], max_samples, replace=False
                    )
                    dataset['X'] = dataset['X'][indices]
                    dataset['y'] = dataset['y'][indices]
        
        return datasets_dict
    
    def load_regression_datasets(
        self,
        include_synthetic: bool = True,
        include_real: bool = True,
        max_samples: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load standard regression datasets for benchmarking
        
        Args:
            include_synthetic: Include synthetic datasets
            include_real: Include real-world datasets
            max_samples: Maximum samples per dataset (for speed)
            
        Returns:
            Dictionary of datasets with metadata
        """
        datasets_dict = {}
        
        if include_real:
            # Boston housing (deprecated but still useful for benchmarking)
            try:
                boston = datasets.load_boston()
                datasets_dict['boston_housing'] = {
                    'X': boston.data,
                    'y': boston.target,
                    'feature_names': boston.feature_names,
                    'description': 'Boston housing price prediction'
                }
            except:
                # If Boston is not available, skip it
                pass
            
            # Diabetes dataset
            diabetes = datasets.load_diabetes()
            datasets_dict['diabetes'] = {
                'X': diabetes.data,
                'y': diabetes.target,
                'feature_names': diabetes.feature_names,
                'description': 'Diabetes progression prediction'
            }
        
        if include_synthetic:
            # Synthetic regression dataset
            X_reg, y_reg = datasets.make_regression(
                n_samples=1000,
                n_features=15,
                n_informative=10,
                noise=0.1,
                random_state=self.random_state
            )
            datasets_dict['synthetic_regression'] = {
                'X': X_reg,
                'y': y_reg,
                'feature_names': [f'feature_{i}' for i in range(15)],
                'description': 'Synthetic regression dataset'
            }
        
        # Apply sample limit if specified
        if max_samples:
            for name, dataset in datasets_dict.items():
                if dataset['X'].shape[0] > max_samples:
                    indices = np.random.RandomState(self.random_state).choice(
                        dataset['X'].shape[0], max_samples, replace=False
                    )
                    dataset['X'] = dataset['X'][indices]
                    dataset['y'] = dataset['y'][indices]
        
        return datasets_dict
    
    def get_dataset_summary(self, datasets: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a formatted summary of loaded datasets
        
        Args:
            datasets: Dictionary of datasets from load_*_datasets methods
            
        Returns:
            Formatted string summary
        """
        summary = "📊 Loaded Datasets Summary:\n"
        summary += "=" * 50 + "\n"
        
        for name, dataset in datasets.items():
            X, y = dataset['X'], dataset['y']
            summary += f"\n🔹 {name.replace('_', ' ').title()}:\n"
            summary += f"   Shape: {X.shape}\n"
            summary += f"   Features: {X.shape[1]}\n"
            summary += f"   Samples: {X.shape[0]}\n"
            
            if 'target_names' in dataset:
                summary += f"   Classes: {len(dataset['target_names'])} {dataset['target_names']}\n"
            else:
                summary += f"   Target range: [{y.min():.2f}, {y.max():.2f}]\n"
            
            summary += f"   Description: {dataset.get('description', 'No description')}\n"
        
        return summary
    
    def prepare_dataset_splits(
        self,
        datasets: Dict[str, Dict[str, Any]],
        test_size: float = 0.3,
        stratify: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare train/test splits for all datasets
        
        Args:
            datasets: Dictionary of datasets
            test_size: Fraction for test set
            stratify: Whether to stratify splits (for classification)
            
        Returns:
            Dictionary with train/test splits for each dataset
        """
        splits = {}
        
        for name, dataset in datasets.items():
            X, y = dataset['X'], dataset['y']
            
            # Determine if this is classification (discrete y) or regression
            is_classification = len(np.unique(y)) < 50 and y.dtype in [int, np.int32, np.int64]
            
            stratify_y = y if (stratify and is_classification) else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=stratify_y
            )
            
            splits[name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'is_classification': is_classification,
                'metadata': dataset
            }
        
        return splits


def get_standard_datasets(
    task_type: str = 'classification',
    include_synthetic: bool = True,
    include_real: bool = True,
    max_samples: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to get standard datasets for benchmarking
    
    Args:
        task_type: 'classification' or 'regression'
        include_synthetic: Include synthetic datasets
        include_real: Include real-world datasets
        max_samples: Maximum samples per dataset
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of datasets ready for benchmarking
    """
    manager = DatasetManager(random_state=random_state)
    
    if task_type.lower() == 'classification':
        return manager.load_classification_datasets(
            include_synthetic=include_synthetic,
            include_real=include_real,
            max_samples=max_samples
        )
    elif task_type.lower() == 'regression':
        return manager.load_regression_datasets(
            include_synthetic=include_synthetic,
            include_real=include_real,
            max_samples=max_samples
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'classification' or 'regression'")


def create_custom_dataset(
    dataset_type: str,
    n_samples: int = 1000,
    n_features: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Create custom synthetic dataset for testing
    
    Args:
        dataset_type: 'classification', 'regression', or 'clustering'
        n_samples: Number of samples
        n_features: Number of features
        **kwargs: Additional parameters for dataset generation
        
    Returns:
        Dataset dictionary with X, y, and metadata
    """
    random_state = kwargs.get('random_state', 42)
    
    if dataset_type == 'classification':
        n_classes = kwargs.get('n_classes', 3)
        n_informative = kwargs.get('n_informative', max(2, n_features // 2))
        
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=n_classes,
            random_state=random_state,
            **{k: v for k, v in kwargs.items() if k not in ['n_classes', 'n_informative', 'random_state']}
        )
        
        return {
            'X': X,
            'y': y,
            'target_names': [f'Class_{i}' for i in range(n_classes)],
            'feature_names': [f'feature_{i}' for i in range(n_features)],
            'description': f'Custom {n_classes}-class classification dataset'
        }
    
    elif dataset_type == 'regression':
        n_informative = kwargs.get('n_informative', max(2, n_features // 2))
        
        X, y = datasets.make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            random_state=random_state,
            **{k: v for k, v in kwargs.items() if k not in ['n_informative', 'random_state']}
        )
        
        return {
            'X': X,
            'y': y,
            'feature_names': [f'feature_{i}' for i in range(n_features)],
            'description': f'Custom regression dataset'
        }
    
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")