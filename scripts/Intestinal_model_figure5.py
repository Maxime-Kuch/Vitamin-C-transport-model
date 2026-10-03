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

from intestinal_wall_model import simulate_transport

def combined_mucus_effect(base_params, thicknesses):
    '''
    Figure 5 A,B,C,D,E. Impact of mucus on vitamin C transport
    '''
    colors = ['#bd282f', '#eac511', '#4faea2']
    colors_2 = ['#7F0000', '#D73027', '#F4A3A3']
    
    labels = ['49.2 $\mathrm{\mu}$m', '123 $\mathrm{\mu}$m', '450 $\mathrm{\mu}$m']
    
    t_sim = 265
    impulse_duration = 0.5
    
    results = []
    
    
    for thic in thicknesses:
        p = base_params.copy()
        p.update({
            'd_muc': thic,
            'ADM_int': 5000, 
            'impuls_time': impulse_duration
        })
        
        res = simulate_transport(p['initial_cond'], (0, t_sim), p, uptake_side='intestine')
        results.append(res)
    
    with plt.rc_context({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 14,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'legend.fontsize': 11
        }):
        
        fig, axes = plt.subplots(2, 3, figsize=(16,8))
        
        
        t_raw = results[1]['t']
        idx_start = np.argmax(t_raw >= 250.4)
        t_plot = t_raw[idx_start:] - 250.4
        
        # Figure 5 A plot
        for i, res in enumerate(results):
            C_membrane = res['mucus'][-1, idx_start:] 
            axes[0,0].plot(t_plot, C_membrane, color=colors[i], linewidth=1.5, label=labels[i])
            
        
        axes[0,0].set_xlim(-0.1, 5) 
        
        axes[0,0].set_xlabel('Time post-impulse [h]')
        axes[0,0].set_ylabel('Asc concentration [$\mathrm{\mu}$M]')
        axes[0,0].legend(frameon=False)
        axes[0,0].grid(True, alpha=0.3)
    
        #Figure 5B plot
        time_snapshot = 250.5 + impulse_duration + 1.0 
        
        for i, res in enumerate(results):
            idx_snap = np.argmin(np.abs(res['t'] - time_snapshot))
            C_spatial = res['mucus'][:, idx_snap]
            
            x_norm = np.linspace(0, 1, base_params['N'])
            
            
            axes[0,1].fill_between(x_norm, C_spatial, color=colors[i], alpha=0.15)
            axes[0,1].plot(x_norm, C_spatial, color=colors[i], linewidth=1.5, label=labels[i])
    
        
        axes[0,1].set_xlabel('Normalized mucus depth')
        axes[0,1].set_ylabel('Asc concentration [$\mathrm{\mu}$M]')
        axes[0,1].grid(True, alpha=0.3)
    
        #Figure 5C plot
        
        thic_range = np.linspace(10e-6, 500e-6, 30)
        conc_range = [5000, 2500, 500]
        total_uptake = []
        list_of_conc_res_upt = []
        
        for conc in conc_range:
            for thic in thic_range:
                p = base_params.copy()
                p.update({'d_muc': thic, 'ADM_int': conc, 'impuls_time': impulse_duration})
                res = simulate_transport(p['initial_cond'], (0, t_sim), p, uptake_side='intestine')
                total_uptake.append(res['Q_capillary_out'][-1]) 
                
            list_of_conc_res_upt.append(total_uptake.copy())
            total_uptake.clear()
            
        for i in range(0,len(list_of_conc_res_upt)):   
             axes[1,0].plot(thic_range * 1e6, list_of_conc_res_upt[i], linewidth=1.5, color=colors_2[i], label=fr'{conc_range[i]} $\mathrm{{\mu}}$M')
             print(f'Mucus thickness for maximal uptake (C={conc_range[i]}): {thic_range[np.argmax(list_of_conc_res_upt[i])]}')
        print(f'Observation time (uptake): {t_sim-250.5}')
        
        
        axes[1,0].set_xlabel('Mucus thickness [$\mathrm{\mu}$m]')
        axes[1,0].set_ylabel('Total Asc amount transported [$\mathrm{\mu}$mol]')
        axes[1,0].legend(frameon=False)
        axes[1,0].grid(True, alpha=0.3)
        
        # Figure 5D plot
        thic_range = np.linspace(1e-6, 500e-6, 30)
        conc_range = [5000, 250, 60]
        total_efflux = []
        list_of_conc_res = []
        print(f'Observation time (efflux): {t_sim-250.5}')
        
        for conc in conc_range:
            for thic in thic_range:
                p = base_params.copy()
                p.update({'d_muc': thic, 'ADM_int': 0, 'ADM_blood_impulse': conc, 'impuls_time': impulse_duration})
                res = simulate_transport(p['initial_cond'], (0, t_sim), p, uptake_side='blood')
                
                total_efflux.append(np.abs(res['Q_lumen_out'][-1])) 
            list_of_conc_res.append(total_efflux.copy())
            total_efflux.clear()
            
        
        for i in range(0,len(list_of_conc_res)):   
            axes[1,1].plot(thic_range * 1e6, list_of_conc_res[i], linewidth=1.5, color=colors_2[i], label=fr'{conc_range[i]} $\mathrm{{\mu}}$M')
        
        axes[1,1].set_xlabel('Mucus thickness [$\mathrm{\mu}$m]')
        axes[1,1].set_ylabel('Total Asc amount transported [$\mathrm{\mu}$mol]')
        axes[1,1].legend(frameon=False)
        axes[1,1].grid(True, alpha=0.3)
        
        # Figure 5E plot
        thic_range = np.linspace(1e-6, 500e-6, 70)
        conc_range = np.linspace(1, 8000, 70)
        opt_thic=[]
        total_uptake=[]
        
        
        for conc in conc_range:
            for thic in thic_range:
                p = base_params.copy()
                p.update({'d_muc': thic, 'ADM_int': conc, 'impuls_time': impulse_duration})
                res = simulate_transport(p['initial_cond'], (0, t_sim), p, uptake_side='intestine')
                total_uptake.append(res['Q_capillary_out'][-1])
            
            max_upt = thic_range[np.argmax(total_uptake)] 
            total_uptake.clear()
            opt_thic.append(max_upt*1e6)
           
            
        axes[1,2].plot(conc_range, opt_thic, linewidth=1.5, color='black')
        
        axes[1,2].set_xlabel('Intestinal Asc concentration [$\mathrm{\mu}$M]')
        axes[1,2].set_ylabel('Optimal mucus thickness [$\mathrm{\mu}$m]')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
          
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
  
    #Simulation time is usually longer, as intrecellular concnetrations are equilibrated with respect to initial condiditions
    #The concnetration impulses appear only after plateau region of concentration in each compartment is reached.
    
    kinetics(conc_base, time, uptake='intestine', plots=True)
    
    mucus_thicknesses = [0.4*123e-6, 123e-6, 450e-6] 
    combined_mucus_effect(conc_base, mucus_thicknesses)
  
if __name__ == "__main__":
    main()
