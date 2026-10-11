import numpy as np
import sys
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.append(str(SRC_DIR))

from intestinal_wall_model import simulate_transport, J_impulse


def kinetics(conc_params, t_max, uptake, plots=True):
    '''
    Parameters
    ----------
    conc_params : dict
        dictionary of parameters.
    t_max : float
        simulation total time.
    uptake : string
        the compartment of vitamin C concnetration impulse ('intestine' or 'blood').
    plots : bool, optional
        Do or do not generate plots. The default is True.

    Returns
    -------
    avg_flux : float
        Total integrated amount of vitamin C transported through endothelial barrier or mucus-lumen interface, normalized
        with respect to impulse duration.
    res : dict
        dictionary of model results.

    '''
    t_span = (0, t_max) 
    
    res = simulate_transport(conc_params['initial_cond'], t_span, conc_params, uptake_side=uptake)
    
    t = res['t']
    k_scale = conc_params['V_water'] * conc_params['V_water_bulk']
    
    # indexing for timescale
    idx_pulse_start = np.argmax(t >= 250)
    idx_short_end = np.argmax(t >= 258) if max(t) >= 258 else len(t)-1
    idx_supshort_end = np.argmax(t >= 253) if max(t) >= 253 else len(t)-1
    idx_long_end = len(t)-1

    t_short = t[idx_pulse_start:idx_short_end] - 250.0
    t_supshort = t[idx_pulse_start:idx_supshort_end] - 250.0
    t_long = t[idx_pulse_start:idx_long_end] - 250.0
    
    if plots:
        
        
        with plt.rc_context({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 14,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11
        }):
            # FIGURE 4/5 A: 
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.35, wspace=0.45)
            fnt=14
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[0, 3])
            ax5 = fig.add_subplot(gs[0, 4])
            
            pulse_cmax = conc_params.get('ADM_int', 0) if uptake == 'intestine' else conc_params.get('ADM_blood_impulse', 0)
            input_pulse_short = [J_impulse(time, 'square', pulse_cmax, conc_params['impuls_time']) for time in t[idx_pulse_start:idx_short_end]]
            input_pulse_long = [J_impulse(time, 'square', pulse_cmax, conc_params['impuls_time']) for time in t[idx_pulse_start:idx_long_end]]
            
           
            ax1.plot(t_short, input_pulse_short if uptake=='intestine' else np.zeros_like(t_short), 'k-')
            ax1.set(title='Intestine', xlabel='t [h]', ylabel=r'$[Asc_{\mathrm{int}}]$ [$\mathrm{\mu}$M]')
            ax1.set_xticks(np.arange(0, 10, 2))
            ax1.grid()
            
            if uptake=='intestine':
                
                ax2.plot(t_short, res['mucus'][-1, idx_pulse_start:idx_short_end], 'k-')
                ax2.set(title='Apical Mucus', xlabel='t [h]', ylabel=r'$[Asc_{\mathrm{muc}}]$ [$\mathrm{\mu}$M]')
                ax2.set_xticks(np.arange(0, 10, 2))
                
                ax2.grid()
            elif uptake=='blood':
               
                ax2.plot(t_long, res['mucus'][-1, idx_pulse_start:idx_long_end], 'k-')
                ax2.set(title='Apical Mucus', xlabel='t [h]', ylabel=r'$[Asc_{\mathrm{muc}}]$ [$\mathrm{\mu}$M]')
                ax2.set_xticks(np.arange(0, 60, 10))
                
                ax2.grid()
                
 
            ax3.plot(t_long, res['enterocyte'][idx_pulse_start:idx_long_end] * k_scale, 'k-')
            ax3.set(title='Enterocyte', xlabel='t [h]', ylabel=r'$[Asc_{\mathrm{in}}]$ [$\mathrm{\mu}$M]')
            ax3.set_xticks(np.arange(0, 60, 10))
            ax3.grid()
            
            if uptake=='intestine':
                
                ax4.plot(t_long, res['isf'][idx_pulse_start:idx_long_end], 'k-')
                ax4.set(title='Basolateral ISF', xlabel='t [h]', ylabel=r'$[Asc_{\mathrm{isf}}]$ [$\mathrm{\mu}$M]')
                ax4.set_xticks(np.arange(0, 60, 10))
                
                ax4.grid()
            elif uptake=='blood':
              
                ax4.plot(t_short, res['isf'][idx_pulse_start:idx_short_end], 'k-')
                ax4.set(title='Basolateral ISF', xlabel='t [h]', ylabel=r'$[Asc_{\mathrm{isf}}]$ [$\mathrm{\mu}$M]')
                ax4.set_xticks(np.arange(0, 10, 2))
                
                ax4.grid()
            
           
            blood_phys_short = np.ones_like(t_short) * conc_params.get('ADM_blood', 50)
            blood_val = blood_phys_short + input_pulse_short if uptake == 'blood' else blood_phys_short
            ax5.plot(t_short, blood_val, 'k-')
            ax5.set(title='Blood', xlabel='t [h]', ylabel=r'$[Asc_{\mathrm{phys}}]$ [$\mathrm{\mu}$M]')
            ax5.set_xticks(np.arange(0, 10, 2))
            ax5.grid()
    
            # FIGURE 3/4 B: 
            ax_bar = fig.add_subplot(gs[1, 0:5])
            max_vals = [
                np.max(input_pulse_long if uptake=='intestine' else np.zeros_like(t_long)),
                np.max(res['mucus'][-1, idx_pulse_start:idx_long_end]), 
                np.max(res['enterocyte'][idx_pulse_start:idx_long_end] * k_scale), 
                np.max(res['isf'][idx_pulse_start:idx_short_end]),
                np.max(blood_val)
            ]
            print(f'Mucus concentration max = {np.max(res["mucus"][-1, idx_pulse_start:idx_long_end])}')
            bars = ax_bar.bar(['Intestine', 'Apical Mucus', 'Enterocyte', 'Basolateral ISF', 'Blood'], 
                              max_vals, color='lightgrey', edgecolor='black', linewidth=2, width=0.6)
            ax_bar.set_yscale('log')
            ax_bar.set_ylim(0.2, max(max_vals) * 5)
            ax_bar.set_title('Maximal Asc Concentrations', fontsize=fnt)
            ax_bar.set_ylabel(r'$[Asc]$ [$\mathrm{\mu}$M]', fontsize=fnt)
            ax_bar.bar_label(bars, fmt='%.1f', padding=3, fontsize=fnt-2)
            ax_bar.grid(axis='y')
            ax_bar.tick_params(axis='x', labelsize=fnt)
            plt.tight_layout()

            if uptake=='intestine':
                output_path = RESULTS_DIR / "figure_4_AB.png"
            else:
                output_path = RESULTS_DIR / "figure_5_AB.png"
              
            plt.savefig(output_path, dpi=300)
          
        #Additional result (not prsented in the manuscript)
        fig3d = plt.figure(figsize=(10, 7))
        ax_3d = fig3d.add_subplot(111, projection='3d')
        
        x_pos = np.linspace(0, conc_params['d_muc'] * 1e6, conc_params['N'])
        T_mesh, X_mesh = np.meshgrid(t_supshort, x_pos)
        Z_mesh = res['mucus'][:, idx_pulse_start:idx_supshort_end ]
        
        surf = ax_3d.plot_surface(X_mesh, T_mesh, Z_mesh, cmap='viridis', edgecolor='none', alpha=0.9)
        ax_3d.set_title(f'Spatial Dynamics in Mucus Layer ({uptake} Absorption)')
        ax_3d.set_xlabel(r'Mucus Depth [$\mathrm{\mu}$m]')
        ax_3d.set_ylabel('Time from impulse [h]')
        ax_3d.set_zlabel(r'Concentration [$\mathrm{\mu}$M]')
        fig3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10)

    
    if uptake=='intestine':
        avg_flux = res['Q_capillary_out'][-1] / conc_params['impuls_time']
    else:
        avg_flux = res['Q_lumen_out'][-1] / conc_params['impuls_time']
        
    return avg_flux, res

def michaelis_menten(S, V_max, K_m):
    '''
    Parameters
    ----------
    S : float
        concentration.
    V_max : float
        maximal transport rate.
    K_m : float
        Michaelis-Menten constant.

    Returns
    -------
    float
        flux.

    '''
    return (V_max * S) / (K_m + S)

def run_dose_response(stepping_values, mucus_thicknesses, base_params):
    """
    Figure 4C and Figure 5C. Dose response and Michaelis-Menten fitting for bidirectional transport.
    """
    fluxes_AB = np.zeros((len(mucus_thicknesses), len(stepping_values)))
    fluxes_BA = np.zeros((len(mucus_thicknesses), len(stepping_values)))
    
    for m_idx, muc in enumerate(mucus_thicknesses):
        for c_idx, conc_val in enumerate(stepping_values):
            # A->B tranport
            c_int = base_params.copy()
            c_int.update({'ADM_int': conc_val, 'd_muc': muc, 'ADM_blood_impulse': 0, 'ADM_blood': 0})
            flux_ab, _ = kinetics(c_int, 260, 'intestine', plots=False)
            fluxes_AB[m_idx, c_idx] = flux_ab
            
            # B->A transport
            c_blo = base_params.copy()
            c_blo.update({'ADM_int': 0, 'd_muc': muc, 'ADM_blood_impulse': conc_val, 'ADM_blood': 0})
            flux_ba, _ = kinetics(c_blo, 260, 'blood', plots=False)
            fluxes_BA[m_idx, c_idx] = flux_ba

    
    popt_AB_list = []
    popt_BA_list = []

    print("\n=========  A----->B =======")
    print(f"{'Mucus thickness':<18} | {'V_max [umol/h]':<15} | {'K_m [uM]':<10}")
    print("-" * 48)
    for i, muc in enumerate(mucus_thicknesses):
        popt, _ = curve_fit(michaelis_menten, stepping_values, fluxes_AB[i, :], p0=[max(fluxes_AB[i,:]), 500])
        popt_AB_list.append(popt)
        print(f"{muc*1e6:>5.1f} um (Index {i}) | {popt[0]} | {popt[1]}")

    print("\n=========  B----->A =======")
    print(f"{'Mucus thickness':<18} | {'V_max [umol/h]':<15} | {'K_m [uM]':<10}")
    print("-" * 48)
    for i, muc in enumerate(mucus_thicknesses):
        popt, _ = curve_fit(michaelis_menten, stepping_values, fluxes_BA[i, :], p0=[max(fluxes_BA[i,:]), 500])
        popt_BA_list.append(popt)
        print(f"{muc*1e6:>5.1f} um (Index {i}) | {popt[0]:} | {popt[1]}")

    
    S_smooth = np.linspace(min(stepping_values), max(stepping_values), 300)

    # A -> B (Figure 4C)
    plt.figure(figsize=(7, 5))
    fnt = 14
    
    plt.fill_between(
        stepping_values, 
        fluxes_AB[2, :], 
        fluxes_AB[0, :], 
        color='royalblue', alpha=0.15, label='Mucus Envelope', linewidth=0
    )
    
    
    plt.plot(stepping_values, fluxes_AB[1, :], color='black', label=f'Simulation {mucus_thicknesses[1]*1e6:.0f} $\mathrm{{\mu}}$m')
    
    #Fitting
    plt.plot(S_smooth, michaelis_menten(S_smooth, *popt_AB_list[0]), color='#bd282f', linestyle='--', linewidth=2, 
             label=fr'M-M fit {mucus_thicknesses[0]*1e6:.0f} $\mathrm{{\mu}}$m')
    plt.plot(S_smooth, michaelis_menten(S_smooth, *popt_AB_list[1]), color='#bd282f', linestyle='-', linewidth=2, 
             label=fr'M-M fit {mucus_thicknesses[1]*1e6:.0f} $\mathrm{{\mu}}$m')
    plt.plot(S_smooth, michaelis_menten(S_smooth, *popt_AB_list[2]), color='#bd282f', linestyle='-.', linewidth=2, 
             label=fr'M-M fit {mucus_thicknesses[2]*1e6:.0f} $\mathrm{{\mu}}$m')
    
    plt.title('Dose Response Analysis (A --> B)', fontsize=fnt)
    plt.xlabel(r'Intestinal vitamin C concentration [$\mathrm{\mu}$M]', fontsize=fnt)
    plt.ylabel(r'Vitamin C transport rate [$\mathrm{\mu}$mol/h]', fontsize=fnt)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    output_path = RESULTS_DIR / "figure_4_C.png"
    plt.savefig(output_path, dpi=300)

    # B -> A (Figure 5C)
    plt.figure(figsize=(7, 5))
    
    plt.fill_between(
        stepping_values, 
        fluxes_BA[2, :], 
        fluxes_BA[0, :], 
        color='royalblue', alpha=0.15, label='Mucus Envelope', linewidth=0
    )
    
    plt.plot(stepping_values, fluxes_BA[1, :], color='black', label=f'Simulation {mucus_thicknesses[1]*1e6:.0f} $\mathrm{{\mu}}$m')
    
    plt.plot(S_smooth, michaelis_menten(S_smooth, *popt_BA_list[0]), color='#bd282f', linestyle='--', linewidth=2, 
             label=fr'M-M fit {mucus_thicknesses[0]*1e6:.0f} $\mathrm{{\mu}}$m')
    plt.plot(S_smooth, michaelis_menten(S_smooth, *popt_BA_list[1]), color='#bd282f', linestyle='-', linewidth=2, 
             label=fr'M-M fit {mucus_thicknesses[1]*1e6:.0f} $\mathrm{{\mu}}$m')
    plt.plot(S_smooth, michaelis_menten(S_smooth, *popt_BA_list[2]), color='#bd282f', linestyle='-.', linewidth=2, 
             label=fr'M-M fit {mucus_thicknesses[2]*1e6:.0f} $\mathrm{{\mu}}$m')

    plt.title('Dose Response Analysis (B ---> A)', fontsize=fnt)
    plt.xlabel(r'Blood vitamin C concentration [$\mathrm{\mu}$M]', fontsize=fnt)
    plt.ylabel(r'Vitamin C efflux rate [$\mathrm{\mu}$mol/h]', fontsize=fnt)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
  
    output_path = RESULTS_DIR / "figure_5_C.png"
    plt.savefig(output_path, dpi=300)


def main():
    N_nodes = 30 #Number of divisions for mucus layer implementation

    conc_base = {
        'N': N_nodes,
        'pH_a': 7.4, 'pH_c': 7.0, 'pH_ecf': 7.4, # Compartment pH
        'psi_a': -0.036, 'psi_bl': -0.0538, # V, membrane potentials
        'ADM_int': 2500, # uM, Impulse magnitude (intestine)
        'ADM_blood_impulse': 2500, # uM, Impulse magnitude (blood)
        'ADM_blood': 50, # uM, physiological blood concentration
        'd_muc': 123e-6, #um, mucus thickness
        'impuls_time': 0.5, # h, impulse duration
        'D_w': 0.67 * 1.055e-9, # m2/s, Asc- diffusion coefficient in water
        'initial_cond': np.concatenate((np.zeros(N_nodes + 2), [50, 0,0])),
        'pKa': 4.2,
        'K_m1': 233.8,  # uM
        'V_max1_um': (6.85e-7) * (1e-6) * 60,  # umol/h/um2, the original value of 13.7 pmol/min/oocyte was recalculated according to methodology presented in the SI 
        'K_m2': 22.6,  # uM
        'V_max2_um': (2e-9) * (1e-6) * 60,  # umol/h/um2, the original value of 0.2 pmol/min/5oocyte was recalculated according to methodology presented in the SI
        'P': (1.1e-8) * 0.1 * 3600 * 5.2,  # dm/h, scaling by 5.2 represents recalculation presented in the SI
        'P_Asc': (3e-12) * 0.1 * 3600 * 11.9,  # dm/h, scaling by 11.9 represents recalculation presented in the SI
        'P_spec': (1.2e-5) * 0.1 * 3600, #dm/h, (permeability of endothelial barrier)
        'k_d': 0.0514, #Scaling constant for capillary area
        'V_water': 0.65, # Water occupation of cell volume
        'V_water_bulk': 0.2, #Non-bulk water percentage
        'A_s': 42e-10,    # dm2 cross section area of enterocyte
        'A_a': 840e-10,   # dm2 absorptive apical area of enterocyte
        'A_bl': 500e-10,  # dm2 basolateral area of enterocyte
        'k_scale': 0.32, #SVCT scaling constant
        'F': 96485,   # C/mol
        'T': 310.15,  # K
        'R': 8.314    # J/K/mol
    }
    
    conc_base['V_c'] = (900e-15) * conc_base['V_water'] * conc_base['V_water_bulk']  # dm3
    conc_base['V_ecf'] = (900e-15) * 0.39  # dm3
    conc_base['V_max1_cell'] = conc_base['k_scale'] * conc_base['V_max1_um'] * 840
    conc_base['V_max2_cell'] = conc_base['k_scale'] * conc_base['V_max2_um'] * 500
    conc_base['D_muc'] = 0.45 *conc_base['D_w'] *100 * 60 
    
    time = 300 #h
    #Simulation time is usually longer, as intrecellular concnetrations are equilibrated with respect to initial condiditions
    #The concnetration impulses appear only after plateau region of concentration in each compartment is reached.
    
    kinetics(conc_base, time, uptake='intestine', plots=True)
    kinetics(conc_base, time, uptake='blood', plots=True)
    
    mucus_thicknesses = [0.4*123e-6, 123e-6, 450e-6] 
    stepping_values = np.concatenate([
        np.arange(0, 100, 10),
        np.arange(100, 1000, 100),
        np.arange(1000, 16000, 1000)
    ])
    
    run_dose_response(stepping_values, mucus_thicknesses, conc_base)

if __name__ == "__main__":
    main()
