"""
Results management and storage utilities for benchmarking.

Handles saving, loading, and analysis of benchmark results with
support for multiple formats and statistical analysis.
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import warnings


class ResultsManager:
    """Manages benchmark results storage, loading, and analysis"""
    
    def __init__(self, output_dir: Union[str, Path] = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        algorithm_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_formats: List[str] = ['json', 'pickle', 'csv']
    ) -> Dict[str, Path]:
        """
        Save benchmark results in multiple formats
        
        Args:
            results: List of benchmark results from BenchmarkRunner
            algorithm_name: Name of the algorithm tested
            metadata: Additional metadata to include
            save_formats: List of formats to save ('json', 'pickle', 'csv', 'excel')
            
        Returns:
            Dictionary mapping format to saved file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{algorithm_name}_benchmark_{timestamp}"
        
        # Prepare complete results with metadata
        complete_results = {
            'algorithm': algorithm_name,
            'timestamp': timestamp,
            'metadata': metadata or {},
            'results': results,
            'summary': self._generate_summary(results)
        }
        
        saved_files = {}
        
        # Save JSON format
        if 'json' in save_formats:
            json_path = self.output_dir / f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump(complete_results, f, indent=2, default=self._json_serializer)
            saved_files['json'] = json_path
        
        # Save pickle format (preserves numpy arrays)
        if 'pickle' in save_formats:
            pickle_path = self.output_dir / f"{base_filename}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(complete_results, f)
            saved_files['pickle'] = pickle_path
        
        # Save CSV format (flattened)
        if 'csv' in save_formats:
            csv_path = self.output_dir / f"{base_filename}.csv"
            df = self._results_to_dataframe(results)
            df.to_csv(csv_path, index=False)
            saved_files['csv'] = csv_path
        
        # Save Excel format with multiple sheets
        if 'excel' in save_formats:
            excel_path = self.output_dir / f"{base_filename}.xlsx"
            self._save_excel_report(results, complete_results['summary'], excel_path)
            saved_files['excel'] = excel_path
        
        print(f"📁 Results saved to: {self.output_dir}")
        for format_name, path in saved_files.items():
            print(f"   {format_name.upper()}: {path.name}")
        
        return saved_files
    
    def load_results(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load benchmark results from file
        
        Args:
            filepath: Path to results file
            
        Returns:
            Complete results dictionary
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def compare_algorithms(
        self,
        result_files: List[Union[str, Path]],
        metrics: List[str] = ['accuracy', 'time']
    ) -> pd.DataFrame:
        """
        Compare results across multiple algorithms
        
        Args:
            result_files: List of result file paths
            metrics: Metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for file_path in result_files:
            results = self.load_results(file_path)
            algorithm = results['algorithm']
            
            for dataset_result in results['results']:
                dataset_name = dataset_result['dataset']
                summary = dataset_result['summary']
                
                row = {
                    'algorithm': algorithm,
                    'dataset': dataset_name,
                    'n_samples': dataset_result['shape'][0],
                    'n_features': dataset_result['shape'][1],
                    'n_classes': dataset_result['n_classes']
                }
                
                # Add metric columns
                for metric in metrics:
                    if metric == 'accuracy':
                        row['reml_accuracy'] = summary.get('avg_accuracy_reml', np.nan)
                        row['sklearn_accuracy'] = summary.get('avg_accuracy_sklearn', np.nan)
                        row['accuracy_diff'] = summary.get('avg_accuracy_diff', np.nan)
                    elif metric == 'time':
                        row['reml_time'] = summary.get('avg_time_reml', np.nan)
                        row['sklearn_time'] = summary.get('avg_time_sklearn', np.nan)
                        row['speed_ratio'] = summary.get('speed_ratio', np.nan)
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(
        self,
        results: Union[List[Dict[str, Any]], str, Path],
        algorithm_name: Optional[str] = None,
        include_plots: bool = True
    ) -> str:
        """
        Generate a formatted text report from results
        
        Args:
            results: Results data or path to results file
            algorithm_name: Algorithm name (if not in results)
            include_plots: Whether to include ASCII plots
            
        Returns:
            Formatted report string
        """
        # Load results if path provided
        if isinstance(results, (str, Path)):
            loaded_results = self.load_results(results)
            results_data = loaded_results['results']
            algorithm_name = loaded_results.get('algorithm', algorithm_name)
            metadata = loaded_results.get('metadata', {})
        else:
            results_data = results
            metadata = {}
        
        report = []
        report.append("🔬 BENCHMARK REPORT")
        report.append("=" * 60)
        report.append(f"Algorithm: {algorithm_name or 'Unknown'}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if metadata:
            report.append(f"Configuration: {metadata}")
        
        report.append("")
        
        # Dataset summaries
        for i, dataset_result in enumerate(results_data, 1):
            report.append(f"{i}. {dataset_result['dataset'].upper()} DATASET")
            report.append("-" * 40)
            
            shape = dataset_result['shape']
            report.append(f"   Shape: {shape[0]} samples × {shape[1]} features")
            report.append(f"   Classes: {dataset_result['n_classes']}")
            
            summary = dataset_result['summary']
            report.append(f"   ReML Accuracy: {summary['avg_accuracy_reml']:.4f}")
            report.append(f"   Sklearn Accuracy: {summary['avg_accuracy_sklearn']:.4f}")
            report.append(f"   Accuracy Difference: {summary['avg_accuracy_diff']:.4f}")
            report.append(f"   Speed Ratio: {summary['speed_ratio']:.2f}x")
            
            # Parameter-wise results
            if include_plots and dataset_result['param_values']:
                report.append("\n   Parameter Performance:")
                param_results = dataset_result['param_values']
                
                for param_val, result in param_results.items():
                    report.append(f"      {param_val}: ReML={result['reml']['accuracy']:.3f}, "
                               f"Sklearn={result['sklearn']['accuracy']:.3f}")
            
            report.append("")
        
        # Overall summary
        report.append("📊 OVERALL SUMMARY")
        report.append("-" * 40)
        
        all_accuracy_diffs = []
        all_speed_ratios = []
        
        for dataset_result in results_data:
            summary = dataset_result['summary']
            all_accuracy_diffs.append(summary['avg_accuracy_diff'])
            all_speed_ratios.append(summary['speed_ratio'])
        
        report.append(f"Average Accuracy Difference: {np.mean(all_accuracy_diffs):.4f}")
        report.append(f"Max Accuracy Difference: {np.max(all_accuracy_diffs):.4f}")
        report.append(f"Average Speed Ratio: {np.mean(all_speed_ratios):.2f}x")
        
        return "\n".join(report)
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall summary statistics"""
        all_accuracy_diffs = []
        all_speed_ratios = []
        dataset_count = len(results)
        
        for dataset_result in results:
            summary = dataset_result['summary']
            all_accuracy_diffs.append(summary['avg_accuracy_diff'])
            all_speed_ratios.append(summary['speed_ratio'])
        
        return {
            'total_datasets': dataset_count,
            'overall_avg_accuracy_diff': np.mean(all_accuracy_diffs),
            'overall_max_accuracy_diff': np.max(all_accuracy_diffs),
            'overall_min_accuracy_diff': np.min(all_accuracy_diffs),
            'overall_avg_speed_ratio': np.mean(all_speed_ratios),
            'accuracy_consistency': np.std(all_accuracy_diffs),
            'speed_consistency': np.std(all_speed_ratios)
        }
    
    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to flattened DataFrame"""
        rows = []
        
        for dataset_result in results:
            dataset_name = dataset_result['dataset']
            base_row = {
                'dataset': dataset_name,
                'n_samples': dataset_result['shape'][0],
                'n_features': dataset_result['shape'][1],
                'n_classes': dataset_result['n_classes']
            }
            
            # Add summary statistics
            summary = dataset_result['summary']
            base_row.update({
                'avg_accuracy_reml': summary['avg_accuracy_reml'],
                'avg_accuracy_sklearn': summary['avg_accuracy_sklearn'],
                'avg_time_reml': summary['avg_time_reml'],
                'avg_time_sklearn': summary['avg_time_sklearn'],
                'avg_accuracy_diff': summary['avg_accuracy_diff'],
                'speed_ratio': summary['speed_ratio']
            })
            
            # Add parameter-specific results
            for param_val, param_result in dataset_result['param_values'].items():
                row = base_row.copy()
                row['parameter_value'] = param_val
                row['param_reml_accuracy'] = param_result['reml']['accuracy']
                row['param_sklearn_accuracy'] = param_result['sklearn']['accuracy']
                row['param_reml_time'] = param_result['reml']['time']
                row['param_sklearn_time'] = param_result['sklearn']['time']
                row['param_accuracy_diff'] = param_result['accuracy_diff']
                row['param_speed_ratio'] = param_result['speed_ratio']
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _save_excel_report(
        self,
        results: List[Dict[str, Any]], 
        summary: Dict[str, Any],
        filepath: Path
    ):
        """Save comprehensive Excel report with multiple sheets"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results
            df = self._results_to_dataframe(results)
            df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # Dataset-wise summaries
            dataset_summaries = []
            for dataset_result in results:
                row = {
                    'dataset': dataset_result['dataset'],
                    'shape': f"{dataset_result['shape'][0]}×{dataset_result['shape'][1]}",
                    'n_classes': dataset_result['n_classes']
                }
                row.update(dataset_result['summary'])
                dataset_summaries.append(row)
            
            dataset_df = pd.DataFrame(dataset_summaries)
            dataset_df.to_excel(writer, sheet_name='Dataset_Summary', index=False)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj)


def quick_save_results(
    results: List[Dict[str, Any]],
    algorithm_name: str,
    output_dir: str = "results"
) -> Path:
    """
    Quick utility to save results in JSON format
    
    Args:
        results: Benchmark results
        algorithm_name: Algorithm name
        output_dir: Output directory
        
    Returns:
        Path to saved JSON file
    """
    manager = ResultsManager(output_dir)
    saved_files = manager.save_results(
        results, algorithm_name, save_formats=['json']
    )
    return saved_files['json']


def load_and_compare(
    result_files: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Quick utility to load and compare multiple algorithm results
    
    Args:
        result_files: List of result file paths
        output_path: Optional path to save comparison CSV
        
    Returns:
        Comparison DataFrame
    """
    manager = ResultsManager()
    comparison_df = manager.compare_algorithms(result_files)
    
    if output_path:
        comparison_df.to_csv(output_path, index=False)
        print(f"Comparison saved to: {output_path}")
    
    return comparison_df