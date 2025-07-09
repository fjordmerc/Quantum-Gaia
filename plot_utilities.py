#!/usr/bin/env python3
"""
plot_utilities.py - Shared utilities for consistent academic plot styling.
Contains functions and constants for maintaining consistent appearance across all plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator, LogFormatter
from scipy import stats

# Professional color scheme matching academic publications
ACADEMIC_COLORS = {
    'primary_black': '#000000',
    'primary_red': '#FF0000', 
    'primary_blue': '#0000FF',
    'primary_green': '#00AA00',
    'secondary_orange': '#FF8000',
    'secondary_purple': '#800080',
    'secondary_brown': '#8B4513',
    'secondary_pink': '#FF69B4',
    'classical': '#000000',
    'quantum': '#0000FF',
    'hybrid': '#FF0000',
    'optimal': '#00AA00'
}

# Algorithm-specific color mapping
ALGORITHM_COLORS = {
    'Classical': '#000000',
    'Algorithm 1': '#00AA00',
    'Algorithm 2': '#000000', 
    'Algorithm 3': '#FF0000',
    'Grover': '#0000FF',
    'Oracle': '#800080'
}

# Global plot settings
PLOT_CONFIG = {
    'figsize': (8, 6),
    'dpi': 300,
    'font_size': 12,
    'legend_size': 10,
    'tick_size': 10,
    'line_width': 2.0,
    'marker_size': 6,
    'error_alpha': 0.3,
    'grid_alpha': 0.3
}

def setup_academic_style():
    """
    Setup matplotlib parameters for academic publication quality plots.
    Call this function at the beginning of your plotting script.
    """
    # Set style parameters
    plt.rcParams.update({
        'font.size': PLOT_CONFIG['font_size'],
        'axes.labelsize': PLOT_CONFIG['font_size'],
        'axes.titlesize': PLOT_CONFIG['font_size'] + 2,
        'xtick.labelsize': PLOT_CONFIG['tick_size'],
        'ytick.labelsize': PLOT_CONFIG['tick_size'],
        'legend.fontsize': PLOT_CONFIG['legend_size'],
        'figure.titlesize': PLOT_CONFIG['font_size'] + 4,
        
        # Grid and axes styling
        'axes.grid': True,
        'grid.alpha': PLOT_CONFIG['grid_alpha'],
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        
        # Line and marker styling
        'lines.linewidth': PLOT_CONFIG['line_width'],
        'lines.markersize': PLOT_CONFIG['marker_size'],
        'lines.markeredgewidth': 1,
        'lines.markeredgecolor': 'black',
        
        # Font and text
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'text.usetex': False,  # Set to True if LaTeX is available
        
        # Figure settings
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def create_figure_with_style(figsize=None, subplot_params=None):
    """
    Create a figure with academic styling.
    
    Args:
        figsize: Tuple of (width, height) for figure size
        subplot_params: Dictionary of subplot parameters
    
    Returns:
        fig, ax: Figure and axes objects
    """
    if figsize is None:
        figsize = PLOT_CONFIG['figsize']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if subplot_params:
        plt.subplots_adjust(**subplot_params)
    
    return fig, ax

def add_confidence_bounds(ax, x_data, y_data, color, alpha=None, method='std'):
    """
    Add confidence bounds to a plot with dashed lines and fill.
    
    Args:
        ax: Matplotlib axes object
        x_data: X-axis data
        y_data: Y-axis data
        color: Color for bounds
        alpha: Transparency for fill
        method: Method for calculating bounds ('std', 'ci', 'minmax')
    """
    if alpha is None:
        alpha = PLOT_CONFIG['error_alpha']
    
    if method == 'std':
        # Standard deviation bounds
        y_mean = np.mean(y_data) if hasattr(y_data, '__iter__') else y_data
        y_std = np.std(y_data) if hasattr(y_data, '__iter__') else y_data * 0.1
        y_lower = y_mean - y_std
        y_upper = y_mean + y_std
        
    elif method == 'ci':
        # Confidence interval bounds
        if hasattr(y_data, '__iter__') and len(y_data) > 1:
            confidence_level = 0.95
            alpha_ci = 1 - confidence_level
            y_mean = np.mean(y_data)
            y_sem = stats.sem(y_data)
            t_val = stats.t.ppf(1 - alpha_ci/2, len(y_data) - 1)
            margin_error = t_val * y_sem
            y_lower = y_mean - margin_error
            y_upper = y_mean + margin_error
        else:
            y_lower = y_data * 0.95
            y_upper = y_data * 1.05
            
    elif method == 'minmax':
        # Min-max bounds
        if hasattr(y_data, '__iter__') and len(y_data) > 1:
            y_lower = np.min(y_data)
            y_upper = np.max(y_data)
        else:
            y_lower = y_data * 0.9
            y_upper = y_data * 1.1
    
    # Ensure arrays for plotting
    if not hasattr(y_lower, '__iter__'):
        y_lower = np.full_like(x_data, y_lower)
    if not hasattr(y_upper, '__iter__'):
        y_upper = np.full_like(x_data, y_upper)
    
    # Add dashed boundary lines
    ax.plot(x_data, y_lower, '--', color=color, alpha=0.8, linewidth=1.5)
    ax.plot(x_data, y_upper, '--', color=color, alpha=0.8, linewidth=1.5)
    
    # Add fill between bounds
    ax.fill_between(x_data, y_lower, y_upper, color=color, alpha=alpha)

def plot_algorithm_comparison(ax, data_dict, x_col, y_col, title, xlabel, ylabel, 
                             log_scale=True, add_bounds=True, bound_method='std'):
    """
    Create a comparison plot with multiple algorithms matching target style.
    
    Args:
        ax: Matplotlib axes object
        data_dict: Dictionary with algorithm names as keys and (data, color) tuples as values
        x_col: Column name for x-axis data
        y_col: Column name for y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Whether to use log scale
        add_bounds: Whether to add confidence bounds
        bound_method: Method for calculating bounds
    """
    for label, (data, color) in data_dict.items():
        if len(data) > 0:
            x_data = data[x_col]
            y_data = data[y_col]
            
            # Main line with markers
            ax.plot(x_data, y_data, 'o-', color=color, label=label, 
                   linewidth=PLOT_CONFIG['line_width'], 
                   markersize=PLOT_CONFIG['marker_size'])
            
            # Add confidence bounds if requested
            if add_bounds and len(y_data) > 1:
                add_confidence_bounds(ax, x_data, y_data, color, method=bound_method)
            
            # Add error bars for discrete points
            if len(y_data) > 1:
                if bound_method == 'std':
                    y_err = np.std(y_data) * 0.1
                elif bound_method == 'ci':
                    y_err = stats.sem(y_data) * 1.96  # 95% CI
                else:
                    y_err = (np.max(y_data) - np.min(y_data)) * 0.1
                
                ax.errorbar(x_data, y_data, yerr=y_err, fmt='none', 
                          color=color, capsize=3, alpha=0.7, capthick=1.5)
    
    # Set scale and labels
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])
    else:
        ax.grid(True, linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])
    
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

def create_subplot_comparison(data_dict, figsize=(12, 8), suptitle=None):
    """
    Create a multi-subplot comparison similar to target images.
    
    Args:
        data_dict: Dictionary with subplot configurations
        figsize: Figure size
        suptitle: Overall figure title
    
    Returns:
        fig, axes: Figure and axes objects
    """
    n_subplots = len(data_dict)
    n_cols = min(2, n_subplots)
    n_rows = (n_subplots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_subplots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for i, (subplot_key, subplot_config) in enumerate(data_dict.items()):
        if i < len(axes_flat):
            ax = axes_flat[i]
            
            # Extract configuration
            algorithms = subplot_config.get('algorithms', {})
            x_col = subplot_config.get('x_col', 'x')
            y_col = subplot_config.get('y_col', 'y')
            title = subplot_config.get('title', f'Subplot {i+1}')
            xlabel = subplot_config.get('xlabel', 'X')
            ylabel = subplot_config.get('ylabel', 'Y')
            log_scale = subplot_config.get('log_scale', True)
            
            # Create the plot
            plot_algorithm_comparison(ax, algorithms, x_col, y_col, title, 
                                    xlabel, ylabel, log_scale)
    
    # Hide unused subplots
    for i in range(n_subplots, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=PLOT_CONFIG['font_size'] + 4, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, axes

def save_publication_figure(fig, filepath, dpi=None):
    """
    Save figure with publication-quality settings.
    
    Args:
        fig: Matplotlib figure object
        filepath: Output file path
        dpi: Resolution (defaults to PLOT_CONFIG['dpi'])
    """
    if dpi is None:
        dpi = PLOT_CONFIG['dpi']
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                format='pdf', transparent=False)

def create_bar_plot_with_errors(ax, categories, values, errors=None, colors=None, 
                               title=None, xlabel=None, ylabel=None):
    """
    Create a bar plot with error bars and professional styling.
    
    Args:
        ax: Matplotlib axes object
        categories: List of category names
        values: List of values for each category
        errors: List of error values (optional)
        colors: List of colors for each bar (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if colors is None:
        colors = [ACADEMIC_COLORS['primary_blue']] * len(categories)
    
    x_pos = np.arange(len(categories))
    
    bars = ax.bar(x_pos, values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.2)
    
    # Add error bars if provided
    if errors is not None:
        ax.errorbar(x_pos, values, yerr=errors, fmt='none', 
                   color='black', capsize=4, capthick=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{value:.1f}' if isinstance(value, float) else f'{value}',
               ha='center', va='bottom', fontsize=PLOT_CONFIG['tick_size']-1, 
               fontweight='bold')
    
    # Customize axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    
    if title:
        ax.set_title(title, fontweight='bold', pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    
    ax.grid(True, axis='y', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])
    ax.set_axisbelow(True)

def apply_log_scale_formatting(ax, axis='both'):
    """
    Apply professional log scale formatting to axes.
    
    Args:
        ax: Matplotlib axes object
        axis: Which axis to format ('x', 'y', or 'both')
    """
    if axis in ['x', 'both']:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=50))
    
    if axis in ['y', 'both']:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=50))
    
    ax.grid(True, which='both', linestyle='--', alpha=PLOT_CONFIG['grid_alpha'])
    ax.grid(True, which='minor', linestyle=':', alpha=PLOT_CONFIG['grid_alpha']/2)

# Example usage and testing functions
def demo_plot_styles():
    """
    Generate demonstration plots showing different styling options.
    """
    setup_academic_style()
    
    # Generate sample data
    x = np.logspace(0, 3, 20)
    y1 = x**0.5 + np.random.normal(0, x**0.3 * 0.1, len(x))
    y2 = x**0.3 + np.random.normal(0, x**0.2 * 0.1, len(x))
    y3 = x**0.4 + np.random.normal(0, x**0.25 * 0.1, len(x))
    
    # Create sample data dictionary
    import pandas as pd
    data1 = pd.DataFrame({'x': x, 'y': y1})
    data2 = pd.DataFrame({'x': x, 'y': y2})
    data3 = pd.DataFrame({'x': x, 'y': y3})
    
    algorithms = {
        'Algorithm 1': (data1, ALGORITHM_COLORS['Algorithm 1']),
        'Algorithm 2': (data2, ALGORITHM_COLORS['Algorithm 2']),
        'Algorithm 3': (data3, ALGORITHM_COLORS['Algorithm 3'])
    }
    
    # Create demonstration plot
    fig, ax = create_figure_with_style(figsize=(10, 6))
    plot_algorithm_comparison(ax, algorithms, 'x', 'y', 
                            'Algorithm Performance Comparison',
                            'Input Size', 'Processing Time (s)')
    
    return fig, ax

if __name__ == "__main__":
    # Run demonstration
    fig, ax = demo_plot_styles()
    plt.show()
    print("Academic plotting utilities loaded successfully!")
    print("Use setup_academic_style() at the beginning of your plotting scripts.") 