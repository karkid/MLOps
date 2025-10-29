"""
Visualization utilities for benchmark results.

Provides standardized plotting and visualization functions for
benchmark analysis with consistent styling and VS Code compatibility.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class BenchmarkVisualizer:
    """Creates standardized visualizations for benchmark results"""
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer with consistent styling
        
        Args:
            style: Matplotlib style ('default', 'seaborn', 'ggplot')
            figsize: Default figure size
        """
        self.figsize = figsize
        self.colors = {
            'reml': '#2E86AB',      # Blue
            'sklearn': '#A23B72',    # Purple
            'accuracy': '#F18F01',   # Orange
            'time': '#C73E1D'        # Red
        }
        
        # Configure matplotlib for VS Code compatibility
        self._configure_matplotlib()
        
        # Set style
        if style == 'seaborn':
            plt.style.use('seaborn-v0_8')
        elif style == 'ggplot':
            plt.style.use('ggplot')
        else:
            plt.style.use('default')
    
    def _configure_matplotlib(self):
        """Configure matplotlib for VS Code compatibility"""
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14
    
    def plot_accuracy_comparison(
        self,
        results: List[Dict[str, Any]],
        algorithm_name: str,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Plot accuracy comparison between ReML and sklearn
        
        Args:
            results: Benchmark results from BenchmarkRunner
            algorithm_name: Name of the algorithm
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        fig.suptitle(f'{algorithm_name} - Accuracy Comparison', fontsize=14, fontweight='bold')
        
        for idx, dataset_result in enumerate(results):
            ax = axes[idx]
            dataset_name = dataset_result['dataset']
            
            # Extract parameter values and accuracies
            param_values = []
            reml_accuracies = []
            sklearn_accuracies = []
            
            for param_val, result in dataset_result['param_values'].items():
                param_values.append(param_val)
                reml_accuracies.append(result['reml']['accuracy'])
                sklearn_accuracies.append(result['sklearn']['accuracy'])
            
            # Create x positions
            x = np.arange(len(param_values))
            width = 0.35
            
            # Plot bars
            ax.bar(x - width/2, reml_accuracies, width, 
                  label='ReML', color=self.colors['reml'], alpha=0.8)
            ax.bar(x + width/2, sklearn_accuracies, width,
                  label='sklearn', color=self.colors['sklearn'], alpha=0.8)
            
            # Customize subplot
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{dataset_name.replace("_", " ").title()}')
            ax.set_xticks(x)
            ax.set_xticklabels(param_values, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (reml_acc, sklearn_acc) in enumerate(zip(reml_accuracies, sklearn_accuracies)):
                ax.text(i - width/2, reml_acc + 0.01, f'{reml_acc:.3f}', 
                       ha='center', va='bottom', fontsize=8)
                ax.text(i + width/2, sklearn_acc + 0.01, f'{sklearn_acc:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Accuracy plot saved: {save_path}")
        
        return fig
    
    def plot_performance_comparison(
        self,
        results: List[Dict[str, Any]],
        algorithm_name: str,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Plot comprehensive performance comparison (accuracy + time)
        
        Args:
            results: Benchmark results from BenchmarkRunner
            algorithm_name: Name of the algorithm
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_datasets = len(results)
        fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 10))
        
        if n_datasets == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'{algorithm_name} - Performance Analysis', fontsize=16, fontweight='bold')
        
        for idx, dataset_result in enumerate(results):
            dataset_name = dataset_result['dataset']
            
            # Extract data
            param_values = list(dataset_result['param_values'].keys())
            reml_accuracies = [result['reml']['accuracy'] 
                             for result in dataset_result['param_values'].values()]
            sklearn_accuracies = [result['sklearn']['accuracy'] 
                                for result in dataset_result['param_values'].values()]
            reml_times = [result['reml']['time'] 
                         for result in dataset_result['param_values'].values()]
            sklearn_times = [result['sklearn']['time'] 
                           for result in dataset_result['param_values'].values()]
            
            # Accuracy subplot
            ax_acc = axes[0, idx]
            x = np.arange(len(param_values))
            
            ax_acc.plot(x, reml_accuracies, 'o-', color=self.colors['reml'], 
                       label='ReML', linewidth=2, markersize=6)
            ax_acc.plot(x, sklearn_accuracies, 's-', color=self.colors['sklearn'], 
                       label='sklearn', linewidth=2, markersize=6)
            
            ax_acc.set_title(f'{dataset_name.replace("_", " ").title()} - Accuracy')
            ax_acc.set_xlabel('Parameter Value')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.set_xticks(x)
            ax_acc.set_xticklabels(param_values, rotation=45)
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)
            
            # Time subplot
            ax_time = axes[1, idx]
            
            ax_time.plot(x, reml_times, 'o-', color=self.colors['reml'], 
                        label='ReML', linewidth=2, markersize=6)
            ax_time.plot(x, sklearn_times, 's-', color=self.colors['sklearn'], 
                        label='sklearn', linewidth=2, markersize=6)
            
            ax_time.set_title(f'{dataset_name.replace("_", " ").title()} - Training Time')
            ax_time.set_xlabel('Parameter Value')
            ax_time.set_ylabel('Time (seconds)')
            ax_time.set_xticks(x)
            ax_time.set_xticklabels(param_values, rotation=45)
            ax_time.legend()
            ax_time.grid(True, alpha=0.3)
            ax_time.set_yscale('log')  # Log scale for time
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Performance plot saved: {save_path}")
        
        return fig
    
    def plot_parameter_sensitivity(
        self,
        results: List[Dict[str, Any]],
        algorithm_name: str,
        param_name: str,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Plot parameter sensitivity analysis
        
        Args:
            results: Benchmark results
            algorithm_name: Algorithm name
            param_name: Parameter name being varied
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{algorithm_name} - {param_name} Sensitivity Analysis', 
                    fontsize=14, fontweight='bold')
        
        # Collect data across all datasets
        all_param_values = []
        all_reml_accuracies = []
        all_sklearn_accuracies = []
        all_accuracy_diffs = []
        dataset_labels = []
        
        for dataset_result in results:
            dataset_name = dataset_result['dataset']
            
            for param_val, result in dataset_result['param_values'].items():
                all_param_values.append(param_val)
                all_reml_accuracies.append(result['reml']['accuracy'])
                all_sklearn_accuracies.append(result['sklearn']['accuracy'])
                all_accuracy_diffs.append(result['accuracy_diff'])
                dataset_labels.append(dataset_name)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame({
            'param_value': all_param_values,
            'reml_accuracy': all_reml_accuracies,
            'sklearn_accuracy': all_sklearn_accuracies,
            'accuracy_diff': all_accuracy_diffs,
            'dataset': dataset_labels
        })
        
        # Plot 1: Accuracy comparison
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            ax1.plot(dataset_data['param_value'], dataset_data['reml_accuracy'], 
                    'o-', label=f'{dataset} (ReML)', alpha=0.7)
            ax1.plot(dataset_data['param_value'], dataset_data['sklearn_accuracy'], 
                    's--', label=f'{dataset} (sklearn)', alpha=0.7)
        
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Parameter Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy difference
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            ax2.plot(dataset_data['param_value'], dataset_data['accuracy_diff'], 
                    'o-', label=dataset)
        
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel(param_name)
        ax2.set_ylabel('Accuracy Difference |ReML - sklearn|')
        ax2.set_title('Accuracy Difference vs Parameter Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Sensitivity plot saved: {save_path}")
        
        return fig
    
    def plot_summary_heatmap(
        self,
        results: List[Dict[str, Any]],
        algorithm_name: str,
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Create summary heatmap of performance across datasets and parameters
        
        Args:
            results: Benchmark results
            algorithm_name: Algorithm name
            save_path: Optional save path
            
        Returns:
            Matplotlib figure
        """
        # Prepare data for heatmap
        datasets = [result['dataset'] for result in results]
        param_values = list(results[0]['param_values'].keys())
        
        # Create matrices for ReML and sklearn accuracies
        reml_matrix = np.zeros((len(datasets), len(param_values)))
        sklearn_matrix = np.zeros((len(datasets), len(param_values)))
        diff_matrix = np.zeros((len(datasets), len(param_values)))
        
        for i, dataset_result in enumerate(results):
            for j, param_val in enumerate(param_values):
                result = dataset_result['param_values'][str(param_val)]
                reml_matrix[i, j] = result['reml']['accuracy']
                sklearn_matrix[i, j] = result['sklearn']['accuracy']
                diff_matrix[i, j] = result['accuracy_diff']
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{algorithm_name} - Performance Heatmaps', fontsize=14, fontweight='bold')
        
        # ReML heatmap
        sns.heatmap(reml_matrix, 
                   xticklabels=param_values,
                   yticklabels=datasets,
                   annot=True, fmt='.3f',
                   cmap='Blues', ax=axes[0])
        axes[0].set_title('ReML Accuracy')
        axes[0].set_xlabel('Parameter Value')
        
        # sklearn heatmap
        sns.heatmap(sklearn_matrix,
                   xticklabels=param_values,
                   yticklabels=datasets,
                   annot=True, fmt='.3f',
                   cmap='Purples', ax=axes[1])
        axes[1].set_title('sklearn Accuracy')
        axes[1].set_xlabel('Parameter Value')
        
        # Difference heatmap
        sns.heatmap(diff_matrix,
                   xticklabels=param_values,
                   yticklabels=datasets,
                   annot=True, fmt='.3f',
                   cmap='Reds', ax=axes[2])
        axes[2].set_title('Accuracy Difference')
        axes[2].set_xlabel('Parameter Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Heatmap saved: {save_path}")
        
        return fig
    
    def create_comprehensive_report(
        self,
        results: List[Dict[str, Any]],
        algorithm_name: str,
        param_name: str,
        output_dir: Union[str, Path],
        show_plots: bool = True
    ) -> Dict[str, Path]:
        """
        Create comprehensive visual report with all plots
        
        Args:
            results: Benchmark results
            algorithm_name: Algorithm name
            param_name: Parameter name
            output_dir: Output directory for plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of created plot files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        plot_files = {}
        
        # Accuracy comparison
        fig1 = self.plot_accuracy_comparison(results, algorithm_name)
        accuracy_path = output_dir / f"{algorithm_name}_accuracy_comparison.png"
        fig1.savefig(accuracy_path, dpi=150, bbox_inches='tight')
        plot_files['accuracy'] = accuracy_path
        if not show_plots:
            plt.close(fig1)
        
        # Performance comparison
        fig2 = self.plot_performance_comparison(results, algorithm_name)
        performance_path = output_dir / f"{algorithm_name}_performance_analysis.png"
        fig2.savefig(performance_path, dpi=150, bbox_inches='tight')
        plot_files['performance'] = performance_path
        if not show_plots:
            plt.close(fig2)
        
        # Parameter sensitivity
        fig3 = self.plot_parameter_sensitivity(results, algorithm_name, param_name)
        sensitivity_path = output_dir / f"{algorithm_name}_parameter_sensitivity.png"
        fig3.savefig(sensitivity_path, dpi=150, bbox_inches='tight')
        plot_files['sensitivity'] = sensitivity_path
        if not show_plots:
            plt.close(fig3)
        
        # Summary heatmap
        fig4 = self.plot_summary_heatmap(results, algorithm_name)
        heatmap_path = output_dir / f"{algorithm_name}_summary_heatmap.png"
        fig4.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plot_files['heatmap'] = heatmap_path
        if not show_plots:
            plt.close(fig4)
        
        print(f"📊 Visual report created in: {output_dir}")
        for plot_type, path in plot_files.items():
            print(f"   {plot_type.upper()}: {path.name}")
        
        return plot_files


def quick_plot_comparison(
    results: List[Dict[str, Any]],
    algorithm_name: str,
    plot_type: str = 'accuracy',
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Quick utility to create a single comparison plot
    
    Args:
        results: Benchmark results
        algorithm_name: Algorithm name
        plot_type: Type of plot ('accuracy', 'performance', 'sensitivity', 'heatmap')
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    visualizer = BenchmarkVisualizer()
    
    if plot_type == 'accuracy':
        return visualizer.plot_accuracy_comparison(results, algorithm_name, save_path)
    elif plot_type == 'performance':
        return visualizer.plot_performance_comparison(results, algorithm_name, save_path)
    elif plot_type == 'heatmap':
        return visualizer.plot_summary_heatmap(results, algorithm_name, save_path)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


# VS Code compatibility fixes
def configure_vscode_matplotlib():
    """Configure matplotlib for optimal VS Code display"""
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Ensure plots don't exceed VS Code's display limits
    plt.rcParams['figure.max_open_warning'] = 0

# Auto-configure when module is imported
configure_vscode_matplotlib()