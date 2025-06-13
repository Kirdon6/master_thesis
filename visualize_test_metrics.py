import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import os

class TestMetricsVisualizer:
    def __init__(self, api_key=None, project="cluster-structure-prediction", entity="kirdon6-university-of-copenhagen"):
        """
        Initialize the visualizer with wandb API access.
        
        Parameters:
        -----------
        api_key : str, optional
            Wandb API key. If None, will try to use from environment
        project : str
            Wandb project name
        entity : str
            Wandb entity/username
        """
        self.api = wandb.Api(api_key=api_key)
        self.project = project
        self.entity = entity
        self.runs_data = []
        
    def fetch_runs(self, run_name_filter=None):
        """
        Fetch all runs from the wandb project.
        
        Parameters:
        -----------
        run_name_filter : str, optional
            Filter runs by name pattern
        """
        print(f"Fetching runs from {self.entity}/{self.project}...")
        runs = self.api.runs(f"{self.entity}/{self.project}")
        
        print(f"Found {len(runs)} total runs")
        
        for run in runs:
            # Skip runs without test metrics
            if not any(key.startswith('test/') for key in run.summary.keys()):
                continue
                
            # Apply name filter if specified
            if run_name_filter and run_name_filter not in run.name:
                continue
                
            # Extract run information
            run_info = {
                'run_id': run.id,
                'run_name': run.name,
                'tags': run.tags,
                'config': run.config,
                'summary': run.summary,
                'state': run.state
            }
            
            # Extract key parameters from config
            config = run.config
            run_info['model_type'] = self._extract_model_type(run, config)
            run_info['cond_type'] = config.get('Model_config', {}).get('cond_type', 'unknown')
            run_info['pos_type'] = config.get('Model_config', {}).get('model_type', 'unknown')
            run_info['T'] = config.get('Model_config', {}).get('T', None)
            run_info['epochs'] = config.get('Train_config', {}).get('epochs', None)
            run_info['batch_size'] = config.get('Train_config', {}).get('batch_size', None)
            run_info['learning_rate'] = config.get('Train_config', {}).get('learning_rate', None)
            
            # Extract test metrics
            summary = run.summary
            run_info['test_mae'] = summary.get('test/mae', None)
            run_info['test_hausdorff'] = summary.get('test/hausdorff', None)
            run_info['test_optimized_mae'] = summary.get('test/optimized_mae', None)
            run_info['test_optimized_typed_mae'] = summary.get('test/optimized_typed_mae', None)
            run_info['test_atom_type_accuracy'] = summary.get('test/atom_type_accuracy', None)
            
            # Only add runs that have at least some test metrics
            if any(run_info[key] is not None for key in ['test_mae', 'test_hausdorff', 'test_optimized_mae']):
                self.runs_data.append(run_info)
        
        print(f"Collected {len(self.runs_data)} runs with test metrics")
        return self.runs_data
    
    def _extract_model_type(self, run, config):
        """Extract model type from run name, tags, or config."""
        # Check run name first
        name_lower = run.name.lower()
        if 'mlp' in name_lower:
            return 'MLP'
        elif 'diffusion' in name_lower or 'ddpm' in name_lower:
            return 'Diffusion'
        
        # Check tags
        for tag in run.tags:
            if 'mlp' in tag.lower():
                return 'MLP'
            elif 'diffusion' in tag.lower() or 'ddpm' in tag.lower():
                return 'Diffusion'
        
        # Check config
        model_from_config = config.get('model', '').lower()
        if 'mlp' in model_from_config:
            return 'MLP'
        elif 'diffusion' in model_from_config:
            return 'Diffusion'
        
        return 'Unknown'
    
    def create_comparison_dataframe(self):
        """Create a pandas DataFrame with all runs for easy analysis."""
        df = pd.DataFrame(self.runs_data)
        
        # Clean up the data
        df = df.dropna(subset=['test_mae'])  # Remove runs without basic test metrics
        
        # Create a combined identifier for grouping
        df['model_config'] = df.apply(lambda row: self._create_model_identifier(row), axis=1)
        
        return df
    
    def _create_model_identifier(self, row):
        """Create a descriptive identifier for each model configuration."""
        model_type = row['model_type']
        cond_type = row['cond_type']
        pos_type = row['pos_type']
        T = row['T']
        
        if model_type == 'Diffusion' and T is not None:
            return f"{model_type}_T{T}_{cond_type}_{pos_type}"
        else:
            return f"{model_type}_{cond_type}_{pos_type}"
    
    def plot_test_metrics_comparison(self, save_dir="plots", figsize=(20, 15)):
        """
        Create comprehensive comparison plots of test metrics.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save plots
        figsize : tuple
            Figure size for the plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        df = self.create_comparison_dataframe()
        
        if len(df) == 0:
            print("No data to plot!")
            return
        
        print(f"Plotting data for {len(df)} runs")
        print(f"Model configurations found: {df['model_config'].unique()}")
        
        # Set style
        sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
        
        # Define metrics to plot
        metrics = [
            {'column': 'test_mae', 'title': 'Test MAE', 'ylabel': 'Mean Absolute Error'},
            {'column': 'test_hausdorff', 'title': 'Test Hausdorff Distance', 'ylabel': 'Hausdorff Distance'},
            {'column': 'test_optimized_mae', 'title': 'Test Optimized MAE', 'ylabel': 'Optimized MAE'},
            {'column': 'test_optimized_typed_mae', 'title': 'Test Optimized Typed MAE', 'ylabel': 'Optimized Typed MAE'},
            {'column': 'test_atom_type_accuracy', 'title': 'Test Atom Type Accuracy', 'ylabel': 'Accuracy (%)'},
        ]
        
        # Create the main comparison plot
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Test Metrics Comparison Across Different Model Configurations', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            
            # Filter out None values for this metric
            metric_df = df.dropna(subset=[metric['column']])
            
            if len(metric_df) == 0:
                ax.text(0.5, 0.5, f"No data for {metric['title']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric['title'])
                continue
            
            # Create box plot
            sns.boxplot(data=metric_df, x='model_config', y=metric['column'], ax=ax)
            
            # Customize the plot
            ax.set_title(metric['title'], fontweight='bold')
            ax.set_ylabel(metric['ylabel'], fontweight='bold')
            ax.set_xlabel('Model Configuration', fontweight='bold')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add value annotations
            for i, config in enumerate(metric_df['model_config'].unique()):
                config_data = metric_df[metric_df['model_config'] == config][metric['column']]
                if len(config_data) > 0:
                    mean_val = config_data.mean()
                    ax.text(i, mean_val, f'{mean_val:.3f}', 
                           ha='center', va='bottom', fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Remove the empty subplot
        axes_flat[5].remove()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.15)
        
        # Save the plot
        plot_path = os.path.join(save_dir, "test_metrics_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved comparison plot to {plot_path}")
        
        # Create separate plots for detailed analysis
        self._create_detailed_plots(df, save_dir)
        
        return df
    
    def _create_detailed_plots(self, df, save_dir):
        """Create additional detailed plots."""
        
        # 1. Model Type Comparison
        self._plot_by_model_type(df, save_dir)
        
        # 2. Conditioning Type Comparison
        self._plot_by_conditioning_type(df, save_dir)
        
        # 3. Position Type Comparison
        self._plot_by_position_type(df, save_dir)
        
        # 4. Diffusion Steps Comparison (T values)
        self._plot_by_diffusion_steps(df, save_dir)
        
        # 5. Performance summary table
        self._create_summary_table(df, save_dir)
    
    def _plot_by_model_type(self, df, save_dir):
        """Create plots comparing different model types."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['test_mae', 'test_hausdorff', 'test_optimized_mae']
        metric_names = ['Test MAE', 'Test Hausdorff', 'Test Optimized MAE']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            metric_df = df.dropna(subset=[metric])
            if len(metric_df) > 0:
                sns.boxplot(data=metric_df, x='model_type', y=metric, ax=axes[idx])
                axes[idx].set_title(f'{name} by Model Type', fontweight='bold')
                axes[idx].set_ylabel(name, fontweight='bold')
                axes[idx].set_xlabel('Model Type', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "model_type_comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_by_conditioning_type(self, df, save_dir):
        """Create plots comparing different conditioning types."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['test_mae', 'test_hausdorff', 'test_optimized_mae']
        metric_names = ['Test MAE', 'Test Hausdorff', 'Test Optimized MAE']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            metric_df = df.dropna(subset=[metric])
            if len(metric_df) > 0:
                sns.boxplot(data=metric_df, x='cond_type', y=metric, ax=axes[idx])
                axes[idx].set_title(f'{name} by Conditioning Type', fontweight='bold')
                axes[idx].set_ylabel(name, fontweight='bold')
                axes[idx].set_xlabel('Conditioning Type', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "conditioning_type_comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_by_position_type(self, df, save_dir):
        """Create plots comparing different position types."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['test_mae', 'test_hausdorff', 'test_optimized_mae']
        metric_names = ['Test MAE', 'Test Hausdorff', 'Test Optimized MAE']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            metric_df = df.dropna(subset=[metric])
            if len(metric_df) > 0:
                sns.boxplot(data=metric_df, x='pos_type', y=metric, ax=axes[idx])
                axes[idx].set_title(f'{name} by Position Type', fontweight='bold')
                axes[idx].set_ylabel(name, fontweight='bold')
                axes[idx].set_xlabel('Position Type', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "position_type_comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_by_diffusion_steps(self, df, save_dir):
        """Create plots comparing different diffusion steps (T values)."""
        # Filter only diffusion models
        diffusion_df = df[df['model_type'] == 'Diffusion'].dropna(subset=['T'])
        
        if len(diffusion_df) == 0:
            print("No diffusion model data found for T comparison")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['test_mae', 'test_hausdorff', 'test_optimized_mae']
        metric_names = ['Test MAE', 'Test Hausdorff', 'Test Optimized MAE']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            metric_df = diffusion_df.dropna(subset=[metric])
            if len(metric_df) > 0:
                sns.boxplot(data=metric_df, x='T', y=metric, ax=axes[idx])
                axes[idx].set_title(f'{name} by Diffusion Steps (T)', fontweight='bold')
                axes[idx].set_ylabel(name, fontweight='bold')
                axes[idx].set_xlabel('Diffusion Steps (T)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "diffusion_steps_comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_summary_table(self, df, save_dir):
        """Create a summary table of results."""
        # Group by model configuration and calculate statistics
        summary_stats = []
        
        for config in df['model_config'].unique():
            config_df = df[df['model_config'] == config]
            
            stats = {
                'Model_Config': config,
                'Runs': len(config_df),
                'MAE_mean': config_df['test_mae'].mean() if 'test_mae' in config_df else None,
                'MAE_std': config_df['test_mae'].std() if 'test_mae' in config_df else None,
                'Hausdorff_mean': config_df['test_hausdorff'].mean() if 'test_hausdorff' in config_df else None,
                'Hausdorff_std': config_df['test_hausdorff'].std() if 'test_hausdorff' in config_df else None,
                'Optimized_MAE_mean': config_df['test_optimized_mae'].mean() if 'test_optimized_mae' in config_df else None,
                'Optimized_MAE_std': config_df['test_optimized_mae'].std() if 'test_optimized_mae' in config_df else None,
                'Atom_Accuracy_mean': config_df['test_atom_type_accuracy'].mean() if 'test_atom_type_accuracy' in config_df else None,
                'Atom_Accuracy_std': config_df['test_atom_type_accuracy'].std() if 'test_atom_type_accuracy' in config_df else None,
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save to CSV
        csv_path = os.path.join(save_dir, "results_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary table to {csv_path}")
        
        # Create a formatted table plot
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Format the table for display
        display_df = summary_df.round(4)
        
        table = ax.table(cellText=display_df.values, colLabels=display_df.columns, 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color header
        for i in range(len(display_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Test Metrics Summary by Model Configuration', 
                 fontsize=16, fontweight='bold', pad=20)
        
        table_path = os.path.join(save_dir, "results_summary_table.png")
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return summary_df

def main():
    """Main function to run the visualization."""
    # Initialize visualizer
    visualizer = TestMetricsVisualizer(
        api_key="6da3cfca616fa8b7e812fc8fcf54ef2a08870da2",  # Replace with your API key or set as env var
        project="cluster-structure-prediction",
        entity="kirdon6-university-of-copenhagen"
    )
    
    # Fetch runs (you can add filters here)
    runs_data = visualizer.fetch_runs()
    
    if len(runs_data) == 0:
        print("No runs found!")
        return
    
    # Create plots
    df = visualizer.plot_test_metrics_comparison(save_dir="test_metrics_plots")
    
    print("\nSummary of collected data:")
    print(f"Total runs: {len(df)}")
    print(f"Model types: {df['model_type'].unique()}")
    print(f"Conditioning types: {df['cond_type'].unique()}")
    print(f"Position types: {df['pos_type'].unique()}")
    print(f"Diffusion steps (T): {df['T'].unique()}")
    print(f"Model configurations: {df['model_config'].unique()}")

if __name__ == "__main__":
    main() 