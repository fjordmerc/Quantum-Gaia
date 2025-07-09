#!/usr/bin/env python3
"""
Final publication-ready plot generation script.
Generates Figures F1–F7 with professional academic styling.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap, linregress, shapiro
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style parameters
plt.style.use('default')
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Helvetica', 'Arial'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.6,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out'
})

# Professional color palette
colors = {
    'primary': '#2E86AB',    # Blue
    'secondary': '#A23B72',  # Magenta  
    'accent1': '#F18F01',    # Orange
    'accent2': '#C73E1D',    # Red
    'neutral': '#6C757D',    # Gray
    'success': '#198754'     # Green
}

def setup_directories():
    """Create output directories"""
    out_dir = os.path.expanduser('~/env_QGA/results/plots')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_data():
    """Load all required data files"""
    try:
        base = pd.read_csv(os.path.expanduser('~/env_QGA/results/baseline_full.csv'))
        grov = pd.read_csv(os.path.expanduser('~/env_QGA/results/grover_full.csv'))
        return base, grov
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def compute_ci(data, confidence=0.95):
    """Compute bootstrap confidence intervals"""
    try:
        res = bootstrap((data,), np.mean, confidence_level=confidence, random_state=42)
        return res.confidence_interval.low, res.confidence_interval.high
    except:
        return data.min(), data.max()

def add_power_law_fit(ax, x_data, y_data, color='gray', label_prefix=''):
    """Add power law trend line with statistics"""
    try:
        # Remove any zeros or negative values for log fitting
        mask = (x_data > 0) & (y_data > 0)
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) < 2:
            return None, None, None
            
        log_x = np.log10(x_clean)
        log_y = np.log10(y_clean)
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        
        # Generate trend line
        x_trend = np.logspace(np.log10(x_clean.min()), np.log10(x_clean.max()), 100)
        y_trend = 10**(slope * np.log10(x_trend) + intercept)
        
        # Plot trend line
        ax.loglog(x_trend, y_trend, '--', color=color, alpha=0.7, linewidth=1.5,
                  label=f'{label_prefix}∝ x^{slope:.2f} (R² = {r_value**2:.3f})')
        
        # Shapiro-Wilk test on residuals
        residuals = log_y - (slope * log_x + intercept)
        if len(residuals) >= 3:
            _, shapiro_p = shapiro(residuals)
        else:
            shapiro_p = None
            
        return slope, r_value**2, shapiro_p
    except Exception as e:
        print(f"Error in power law fit: {e}")
        return None, None, None

def create_figure_f1(base, out_dir):
    """Figure 1: μ (pairs) vs. radius"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Main data plot
    ax.loglog(base['radius'], base['pairs'], 'o-', color=colors['primary'], 
              linewidth=2, markersize=7, markerfacecolor='white', 
              markeredgewidth=2, markeredgecolor=colors['primary'],
              label='Observed data')
    
    # Add trend line
    slope, r2, shapiro_p = add_power_law_fit(ax, base['radius'], base['pairs'], 
                                           colors['neutral'], 'μ ')
    
    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel('μ = Number of close pairs')
    ax.set_title('Neighbour Count Growth vs. Angular Radius')
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/F1_pairs.png', format='png')
    plt.close()
    
    return slope, r2, shapiro_p

def create_figure_f2(base, out_dir):
    """Figure 2: Classical runtime vs. radius"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Confidence intervals (approximate)
    ci_lower, ci_upper = compute_ci(base['secs'])
    ci_rel = (ci_upper - ci_lower) / base['secs'].mean()
    err_low = base['secs'] * (1 - ci_rel/2)
    err_high = base['secs'] * (1 + ci_rel/2)
    
    # Main plot with confidence band
    ax.loglog(base['radius'], base['secs'], 'o-', color=colors['secondary'], 
              linewidth=2, markersize=7, markerfacecolor='white',
              markeredgewidth=2, markeredgecolor=colors['secondary'], 
              label='Classical algorithm')
    
    ax.fill_between(base['radius'], err_low, err_high, alpha=0.25, 
                    color=colors['secondary'], label='95% CI estimate')
    
    # Add trend line
    slope, r2, shapiro_p = add_power_law_fit(ax, base['radius'], base['secs'], 
                                           colors['neutral'], 'T ')
    
    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel('Runtime T_classical (s)')
    ax.set_title('Classical Algorithm Runtime Scaling')
    
    # Add statistics annotation
    if shapiro_p:
        ax.text(0.05, 0.95, f'Shapiro-Wilk p = {shapiro_p:.3f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/F2_classical.png', format='png')
    plt.close()
    
    return slope, r2, shapiro_p

def create_figure_f3(grov, out_dir):
    """Figure 3: Grover oracle calls vs. radius"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.loglog(grov['radius'], grov['oracle_calls'], 's-', color=colors['accent1'],
              linewidth=2, markersize=7, markerfacecolor='white',
              markeredgewidth=2, markeredgecolor=colors['accent1'],
              label='Grover algorithm')
    
    # Add trend line
    slope, r2, shapiro_p = add_power_law_fit(ax, grov['radius'], grov['oracle_calls'], 
                                           colors['neutral'], 'Calls ')
    
    ax.set_xlabel('Radius (arcsec)')
    ax.set_ylabel('Oracle Calls')
    ax.set_title('Grover Algorithm Oracle Call Scaling')
    
    # Add statistics annotation
    if shapiro_p:
        ax.text(0.05, 0.95, f'Shapiro-Wilk p = {shapiro_p:.3f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/F3_grover_calls.png', format='png')
    plt.close()
    
    return slope, r2, shapiro_p

def create_figure_f4(base, grov, out_dir):
    """Figure 4: Speed-up ratio vs. μ"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ratio = base['secs'] / grov['grover_sec']
    
    ax.loglog(base['pairs'], ratio, '^-', color=colors['accent2'],
              linewidth=2, markersize=7, markerfacecolor='white',
              markeredgewidth=2, markeredgecolor=colors['accent2'],
              label='Speed-up ratio')
    
    # Add trend line
    slope, r2, shapiro_p = add_power_law_fit(ax, base['pairs'], ratio, 
                                           colors['neutral'], 'Speedup ')
    
    # Add theoretical O(√μ) line for reference
    mu_range = np.logspace(np.log10(base['pairs'].min()), 
                          np.log10(base['pairs'].max()), 100)
    theoretical = 0.1 * np.sqrt(mu_range)  # Scaled for visibility
    ax.loglog(mu_range, theoretical, ':', color=colors['neutral'], alpha=0.6,
              label='O(√μ) reference')
    
    ax.set_xlabel('μ (Number of close pairs)')
    ax.set_ylabel('Speed-up Ratio (T_classical / T_Grover)')
    ax.set_title('Quantum-Inspired Algorithm Speed-up vs. Problem Size')
    
    # Add statistics annotation
    if shapiro_p:
        ax.text(0.05, 0.05, f'Shapiro-Wilk p = {shapiro_p:.3f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/F4_speedup.png', format='png')
    plt.close()
    
    return slope, r2, shapiro_p

def create_figure_f5(out_dir):
    """Figure 5: Runtime scaling comparison or simple comparison"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    try:
        # Try to load subsample data
        sub_b = pd.read_csv(os.path.expanduser('~/env_QGA/results/baseline_sub.csv'))
        sub_g = pd.read_csv(os.path.expanduser('~/env_QGA/results/grover_sub.csv'))
        
        ax.loglog(sub_b['N'], sub_b['secs'], 'o-', color=colors['secondary'],
                  linewidth=2, markersize=7, markerfacecolor='white',
                  markeredgewidth=2, markeredgecolor=colors['secondary'], 
                  label='Classical')
        ax.loglog(sub_g['N'], sub_g['grover_sec'], 's-', color=colors['accent1'],
                  linewidth=2, markersize=7, markerfacecolor='white',
                  markeredgewidth=2, markeredgecolor=colors['accent1'], 
                  label='Grover')
        
        ax.set_xlabel('Catalogue Size N')
        ax.set_ylabel('Runtime (s)')
        ax.set_title('Runtime Scaling Comparison at 10 arcsec')
        
    except FileNotFoundError:
        # Fallback: Simple runtime comparison
        base = pd.read_csv(os.path.expanduser('~/env_QGA/results/baseline_full.csv'))
        grov = pd.read_csv(os.path.expanduser('~/env_QGA/results/grover_full.csv'))
        
        methods = ['Classical', 'Grover']
        times = [base.iloc[0]['secs'], grov.iloc[0]['grover_sec']]
        
        bars = ax.bar(methods, times, color=[colors['secondary'], colors['accent1']], 
                     alpha=0.8, edgecolor='black', linewidth=1.0, width=0.6)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{time:.3f}s', ha='center', va='bottom', fontweight='bold',
                    fontsize=10)
        
        ax.set_ylabel('Wall-time (s)')
        ax.set_title('Runtime Comparison at 10 arcsec')
        ax.set_ylim(0, max(times) * 1.2)
    
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/F5_runtime.png', format='png')
    plt.close()

def create_figure_f6(out_dir):
    """Figure 6: Oracle fidelity validation"""
    try:
        fid = pd.read_csv(os.path.expanduser('~/env_QGA/results/oracle_truth/oracle_fidelity.csv'))
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        bars = ax.bar(fid['N'], fid['fidelity'], color=colors['success'], alpha=0.8,
                      edgecolor='black', linewidth=1.0, width=0.6)
        
        # Add value labels
        for bar, fidelity in zip(bars, fid['fidelity']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{fidelity:.2f}', ha='center', va='bottom', fontweight='bold',
                    fontsize=9)
        
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Catalogue Size N')
        ax.set_ylabel('Logical Fidelity')
        ax.set_title('Oracle Circuit Validation Results')
        ax.axhline(y=1.0, color=colors['accent2'], linestyle='--', alpha=0.7, 
                   linewidth=1.5, label='Perfect Fidelity')
        ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{out_dir}/F6_oracle_validation.png', format='png')
        plt.close()
        
    except FileNotFoundError:
        print("Oracle fidelity CSV not found, skipping F6")

def create_figure_f7(out_dir):
    """Figure 7: Resource Usage Comparison"""
    try:
        res_df = pd.read_csv(os.path.expanduser('~/env_QGA/results/resource_usage.csv'))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Qubit comparison
        qubit_data = res_df.set_index('component')['qubits']
        bars1 = ax1.bar(qubit_data.index, qubit_data.values, 
                        color=[colors['neutral'], colors['primary']], alpha=0.8,
                        edgecolor='black', linewidth=1.0, width=0.6)
        ax1.set_ylabel('Number of Qubits')
        ax1.set_title('Qubit Usage')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, qubit_data.values):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{value}', ha='center', va='bottom', fontweight='bold',
                        fontsize=10)
        
        # Gate count comparison
        gate_data = res_df.set_index('component')['gate_count']
        bars2 = ax2.bar(gate_data.index, gate_data.values,
                        color=[colors['neutral'], colors['primary']], alpha=0.8,
                        edgecolor='black', linewidth=1.0, width=0.6)
        ax2.set_ylabel('Number of Gates')
        ax2.set_title('Gate Count')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, gate_data.values):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                        f'{value}', ha='center', va='bottom', fontweight='bold',
                        fontsize=10)
        
        plt.suptitle('Resource Usage: Classical vs. Quantum Oracle', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/F7_resource_usage.png', format='png')
        plt.close()
        
    except FileNotFoundError:
        print("Resource usage CSV not found, skipping F7")

def save_confidence_intervals(base, grov):
    """Save bootstrap confidence intervals"""
    try:
        ci_data = {
            'classical_CI': list(compute_ci(base['secs'])),
            'grover_CI': list(compute_ci(grov['grover_sec']))
        }
        with open(os.path.expanduser('~/env_QGA/results/runtime_ci.json'), 'w') as f:
            json.dump(ci_data, f, indent=2)
    except Exception as e:
        print(f"CI computation failed: {e}")

def main():
    """Main plotting function"""
    print("Generating publication-ready plots...")
    
    # Setup
    out_dir = setup_directories()
    base, grov = load_data()
    
    if base is None or grov is None:
        print("Failed to load data. Exiting.")
        return
    
    # Generate all figures
    print("Creating Figure F1...")
    f1_stats = create_figure_f1(base, out_dir)
    
    print("Creating Figure F2...")
    f2_stats = create_figure_f2(base, out_dir)
    
    print("Creating Figure F3...")
    f3_stats = create_figure_f3(grov, out_dir)
    
    print("Creating Figure F4...")
    f4_stats = create_figure_f4(base, grov, out_dir)
    
    print("Creating Figure F5...")
    create_figure_f5(out_dir)
    
    print("Creating Figure F6...")
    create_figure_f6(out_dir)
    
    print("Creating Figure F7...")
    create_figure_f7(out_dir)
    
    # Save confidence intervals
    print("Computing confidence intervals...")
    save_confidence_intervals(base, grov)
    
    # Print statistics summary
    print("\n=== STATISTICAL SUMMARY ===")
    print(f"F1 (μ vs radius): slope={f1_stats[0]:.3f}, R²={f1_stats[1]:.3f}")
    print(f"F2 (Classical runtime): slope={f2_stats[0]:.3f}, R²={f2_stats[1]:.3f}")
    print(f"F3 (Grover calls): slope={f3_stats[0]:.3f}, R²={f3_stats[1]:.3f}")
    print(f"F4 (Speed-up): slope={f4_stats[0]:.3f}, R²={f4_stats[1]:.3f}")
    
    print(f"\nAll figures saved to: {out_dir}")
    print("✅ Publication-ready plots generated successfully!")

if __name__ == "__main__":
    main()
