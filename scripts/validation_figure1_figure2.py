import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.append(str(SRC_DIR))

from validation_model import solve_cell_model

val_file_extr = DATA_DIR / "Experimental_extracellular_data.csv"
val_file_intr = DATA_DIR / "experimental_intracellular_data.csv"

df_extr = pd.read_csv(val_file_extr, sep=";", header=None, decimal=",")
df_intra = pd.read_csv(val_file_intr, sep=";", header=None, decimal=",")
    
# Przykładowe wyciągnięcie kolumn (np. czas i pomiary stężenia in vitro)

PARAM_NAMES = {
        'pH_in': r'$pH_{\mathrm{in}}$', 
        'psi': r'$\psi$',
        'Vexcl': r'$k_{\mathrm{bulk}}$', 
        'k_fac_met': r'$k_{\mathrm{met}}$', 
        'D': r'$D$', 
        'k_fac_svct': r'$k_{\mathrm{svct}}$', 
        'cell_vol_fac': r'$k_{\mathrm{surf}}$'
    }
PARAM_UNITS = ['', ' [V]', '', '', r' [$\mathrm{\mu}$m]', '', '']

def calculate_metrics(t_model, y_model, df_exp):
    '''
    Parameters
    ----------
    t_model : numpy array
        timesteps of simulation.
    
    y_model : numpy array
        initial conditions.
    
    df_exp : data frame
        experimental results of intracellular concentration.

    Returns
    -------
    nrmse : float
        normalized root mean square error
    
    err_C : float
        normalized concnetration peak height error
    
    err_t : float
        normalized concnetration peak time error
        
    The function calculates metrics used for model evaluation
    '''
    exp_t = df_exp.iloc[:, 0].values
    exp_y = df_exp.iloc[:, 1].values
    
    exp_max_y = np.max(exp_y)
    exp_max_t = exp_t[np.argmax(exp_y)]
    max_t = np.max(exp_t)
   
    mod_max_y = np.max(y_model)
    mod_max_t = t_model[np.argmax(y_model)]
    
    # Signed Relative Errors 
    err_C = (mod_max_y - exp_max_y) / exp_max_y
    err_t = (mod_max_t - exp_max_t) / max_t
    
    # NRMSE 
    interpolator = interp1d(t_model, y_model, kind='cubic', fill_value='extrapolate')
    y_pred = interpolator(exp_t)
    rmse = np.sqrt(np.mean((exp_y - y_pred) ** 2))
    nrmse = rmse / exp_max_y
    
    return nrmse, err_C, err_t


def plot_baseline_fit(params, df_exp):
    """Figure 1 (A): model basic validation on experimental intracellular data from HCT116 cell culture"""
    t, y_in, y_out = solve_cell_model((0, 48), params)
    nrmse, err_C, err_t = calculate_metrics(t, y_in, df_exp)
    print('Baseline fit metrics ------------------')
    print(f'NRMSE: {nrmse}, dC_max = {err_C}, dt_max = {err_t}')
    
    with plt.rc_context({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
    }):
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.set_xlabel('$t$ [h]')
        ax1.set_ylabel(r'$[Asc_{\mathrm{out}}]$ [$\mathrm{\mu}$M]', color='black')
        l1, = ax1.plot(t, y_out, 'k-', linewidth=2, label='Environment')
        ax1.set_ylim(0, 1100)
        ax1.set_xlim(0, 48)
        ax1.xaxis.set_major_locator(MultipleLocator(6))
        
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'$[Asc_{\mathrm{in}}]$ [$\mathrm{\mu}$M]', color='black')
        l2, = ax2.plot(t, y_in, 'k--', linewidth=2, label='Intracellular (Model)')
        l3, = ax2.plot(df_exp.iloc[:, 0], df_exp.iloc[:, 1], 'r--', marker='o', markersize=4, label='Experiment (HCT116)')
        ax2.set_ylim(0, max(np.max(y_in), np.max(df_exp.iloc[:, 1])) * 1.1)
        
        ax2.legend(handles=[l1, l2, l3], loc='upper right', frameon=False)
        ax1.grid(True, alpha=0.4)
        plt.title('Baseline model validation with experimental data', pad=15)
        plt.tight_layout()
      
        output_path = RESULTS_DIR / "figure_1_A.png"
        plt.savefig(output_path, dpi=300)
        print(f"Wykres zapisano pomyślnie do: {output_path}")


def run_sensitivity_and_plot(params, bounds_dict, df_exp):
    """
    Figure 2: sensitivity analysis of the model - comparison of kinetic profiles 
    """
    res_min, res_max = {}, {}
    
    # Przygotowanie siatki wykresów (2 wiersze x 4 kolumny dla 7 parametrów)
    with plt.rc_context({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 10
    }):
        cols = 3
        rows = 3
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = axes.flatten() 
        
        print("\n" + "="*60)
        print("Sensitivity analysis results")
        print("="*60)
        
        for i, (var, bounds) in enumerate(bounds_dict.items()):
            # Min extreme
            p_min = params.copy()
            p_min[var] = bounds[0]
            t_min, y_in_min, y_out_min = solve_cell_model((0, 48), p_min)
            res_min[var] = calculate_metrics(t_min, y_in_min, df_exp)
            
            # Max extreme
            p_max = params.copy()
            p_max[var] = bounds[1]
            t_max, y_in_max, y_out_max = solve_cell_model((0, 48), p_max)
            res_max[var] = calculate_metrics(t_max, y_in_max, df_exp)
            
            print(f"Parametr: {var} {PARAM_NAMES[var]}")
            print(f"  [-] MIN ({bounds[0]:<5}): NRMSE = {res_min[var][0]:.3f} | dC_max = {res_min[var][1]:>6.3f} | dt_max = {res_min[var][2]:>6.3f}")
            print(f"  [+] MAX ({bounds[1]:<5}): NRMSE = {res_max[var][0]:.3f} | dC_max = {res_max[var][1]:>6.3f} | dt_max = {res_max[var][2]:>6.3f}")
            print("-" * 60)
            
            ax1 = axes[i]
            ax1.set_xlabel('$t$ [h]')
            ax1.set_ylabel(r'$[Asc_{\mathrm{out}}]$ [$\mathrm{\mu}$M]')
            l1, = ax1.plot(t_min, y_out_min, 'k-', label='Environment')
            ax1.set_ylim(0, 1100)
            ax1.set_xlim(0, 48)
            
            ax2 = ax1.twinx()
            ax2.set_ylabel(r'$[Asc_{\mathrm{in}}]$ [$\mathrm{\mu}$M]')
            l2 = ax2.fill_between(t_min, y_in_min, y_in_max, color='royalblue', alpha=0.3, label='Envelope')
            l3, = ax2.plot(df_exp.iloc[:, 0], df_exp.iloc[:, 1], 'r--', label='Experiment')
            ax2.set_ylim(0, max(np.max(y_in_max), np.max(df_exp.iloc[:, 1])) * 1.1)
            
            ax1.set_title(f"Sensitivity: {PARAM_NAMES[var]}", pad=10)
            ax1.grid(True, alpha=0.3)
            
            
            if i == 0:
                ax2.legend(handles=[l1, l2, l3], loc='upper right', frameon=False)
                
       
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()

        output_path = RESULTS_DIR / "figure_2.png"
        plt.savefig(output_path, dpi=300)
        print(f"Wykres zapisano pomyślnie do: {output_path}")
    
    return res_min, res_max

def plot_sensitivity_heatmap(res_min, res_max, bounds_dict):
    """Figure 1 (B, C, D): The results of sensitivity analysis presented with normalized metrics"""
    df_min = pd.DataFrame.from_dict(res_min, orient='index', columns=['NRMSE', r'$\Delta C_{\mathrm{max}}$', r'$\Delta t_{\mathrm{max}}$'])
    df_max = pd.DataFrame.from_dict(res_max, orient='index', columns=['NRMSE', r'$\Delta C_{\mathrm{max}}$', r'$\Delta t_{\mathrm{max}}$'])
    
    df_min.index = [PARAM_NAMES[k] for k in df_min.index]
    df_max.index = [PARAM_NAMES[k] for k in df_max.index]
    
    with plt.rc_context({
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 12
    }):
        fig, (ax1, ax_center, ax2) = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [4, 1.5, 4]}) 
        
        # Heatmapa dla wartości MINIMUM
        sns.heatmap(df_min, annot=True, fmt=".2f", cmap="RdBu_r", center=0, 
                    linewidths=1, linecolor='white', cbar_kws={'label': 'Relative Error'}, ax=ax1)
        ax1.set_title('Minimal Boundary Effect', pad=20, fontsize=18)
        ax1.set_ylabel('Parameter')
        
        # Heatmapa dla wartości MAXIMUM
        sns.heatmap(df_max, annot=True, fmt=".2f", cmap="RdBu_r", center=0, 
                    linewidths=1, linecolor='white', cbar_kws={'label': 'Relative Error'}, ax=ax2)
        ax2.set_title('Maximal Boundary Effect', pad=20, fontsize=18)
        
        # Środkowa kolumna informacyjna (Mini-tabela)
        ax_center.axis('off') 
        ax_center.set_ylim(ax1.get_ylim())
        y_coords = ax1.get_yticks() 
    
        ax_center.axvline(x=0.1, color='lightgrey', linestyle='-', linewidth=1.5, zorder=0)
        ax_center.axvline(x=0.9, color='lightgrey', linestyle='-', linewidth=1.5, zorder=0)
    
        ax_center.text(0.5, 1.05, 'Evaluated Bounds', ha='center', va='bottom', transform=ax_center.transAxes, fontsize=13)
        ax_center.text(0.2, 1.0, 'Min', ha='center', va='top', transform=ax_center.transAxes, fontweight='bold', color='dimgrey')
        ax_center.text(0.8, 1.0, 'Max', ha='center', va='top', transform=ax_center.transAxes, fontweight='bold', color='dimgrey')
    
        for y, key, unit in zip(y_coords, bounds_dict.keys(), PARAM_UNITS):
            w_min, w_max = bounds_dict[key]
            ax_center.text(-0.2, y, PARAM_NAMES[key] + unit, ha='center', va='center', fontsize=11)
            ax_center.text(0.2, y, str(w_min), ha='center', va='center', fontsize=11)
            ax_center.text(0.8, y, str(w_max), ha='center', va='center', fontsize=11)
            
        plt.tight_layout()
        output_path = RESULTS_DIR / "figure_1_BCD.png"
        plt.savefig(output_path, dpi=300)
        print(f"Wykres zapisano pomyślnie do: {output_path}")


def main():

    PARAMS = {
    'pKa': 4.2,
    'K_m1': 233.8,  # uM
    'V_max1_um': (6.85e-7) * 1e-6 * 60,  # umol/h/um2, the original value of 13.7 pmol/min/oocyte was recalculated according to methodology presented in the SI
    'K_m2': 22.6,   # uM
    'V_max2_um': (2e-9) * 1e-6 * 60,     # umol/h/um2, the original value of 0.2 pmol/min/5oocyte was recalculated according to methodology presented in the SI
    'P': (1.1e-8) * 0.1 * 3600 * 5.2,    # dm/h, scaling by 5.2 represents recalculation presented in the SI
    'P_Asc': (3e-12) * 0.1 * 3600 * 11.9, # dm/h, scaling by 11.9 represents recalculation presented in the SI
    'V_water': 0.65, # (this value is represented by 'alpha' in the manuscript and SI)
    'k_AV': 7,      # area-to-volume scaling constant
    'F': 96485,     # C/mol
    'T': 310.15,    # K
    'R_const': 8.314, # J/K/mol (Zmieniono nazwę, aby nie mylić z promieniem R)
    'pH_out': 7.4, 
    'pH_in': 7.2, 
    'psi': -0.0462, # V
    'Vexcl': 0.2, 
    'k_fac_met': 1.0, 
    'D': 12.675, # um, the mean value of HCT116 cell diameter
    'k_fac_svct': 0.32, 
    'cell_vol_fac': 1.5, 
    'initial_cond': [0, 1100] # uM
      }

    SENSITIVITY_BOUNDS = {
        'pH_in': [7.0, 7.4], 
        'psi': [-0.07, -0.01],
        'Vexcl': [0.2, 0.5], 
        'k_fac_met': [1, 10], 
        'D': [13, 18], 
        'k_fac_svct': [0.1, 0.5], 
        'cell_vol_fac': [1, 3]
    }
    
    # Figure 1 (A)
    plot_baseline_fit(PARAMS, df_intra)
     # Figure 2 
    results_min, results_max = run_sensitivity_and_plot(PARAMS, SENSITIVITY_BOUNDS, df_intra)
        
    # Figure 1 (B, C, D)
    plot_sensitivity_heatmap(results_min, results_max, SENSITIVITY_BOUNDS)

if __name__ == "__main__":
    main()
