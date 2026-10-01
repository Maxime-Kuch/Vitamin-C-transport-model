import numpy as np
from scipy.integrate import solve_ivp


# Flux functions and ODE solving

def J_SVCT1(conc, C_Asc_out):
    '''

    Parameters
    ----------
    conc : dict
        Dictionary of parameters.
    C_Asc_out : float
        instanteneous extracellular concentration.

    Returns
    -------
    TYPE float
        instanteneous flux through SVCT1 transporter.

    '''
    #Extraction of parameters and recalculations
    pH_a = conc['pH_out']
    R_normal = conc['D'] / 2
    A_cancer = conc['cell_vol_fac'] * conc['k_AV'] * 4 * np.pi * (R_normal * 1e-5)**2 #Scaled area of the cell membrane
    V_max1_cell = conc['k_fac_svct'] * conc['V_max1_um'] * A_cancer * 1e10 / 10 #Area-scaled Vmax of transporter
    
    C_a_i = C_Asc_out * (10**(pH_a - conc['pKa'])) / (1 + (10**(pH_a - conc['pKa'])))
    return (V_max1_cell * C_a_i) / (conc['K_m1'] + C_a_i)

def J_SVCT2(conc, C_Asc_out):
    '''

    Parameters
    ----------
    conc : dict
        Dictionary of parameters.
    C_Asc_out : float
        instanteneous extracellular concentration.

    Returns
    -------
    TYPE float
        instanteneous flux through SVCT2 transporter.

    '''
    #Extraction of parameters and recalculations
    pH_ecf = conc['pH_out']
    R_normal = conc['D'] / 2
    A_cancer = conc['cell_vol_fac'] * conc['k_AV'] * 4 * np.pi * (R_normal * 1e-5)**2 #Scaled area of the cell membrane
    V_max2_cell = conc['k_fac_svct'] * conc['V_max2_um'] * A_cancer * 1e10 #Area-scaled Vmax of transporter
    
    C_b_i = C_Asc_out * (10**(pH_ecf - conc['pKa'])) / (1 + (10**(pH_ecf - conc['pKa'])))
    return (V_max2_cell * C_b_i) / (conc['K_m2'] + C_b_i)

def J_diff(conc, C_Asc_c, C_Asc_out):
    '''

    Parameters
    ----------
    conc : dict
        Dictionary of parameters.
    C_Asc_c : float
        instanteneous intracellular concentration.
    C_Asc_out : float
        instanteneous extracellular concentration.

    Returns
    -------
    TYPE float
        instanteneous total diffusive flux.

    '''
    #Extraction of parameters and recalculations
    pH_a, pH_c, psi = conc['pH_out'], conc['pH_in'], conc['psi']
    R_normal = conc['D'] / 2
    A_cancer = conc['cell_vol_fac'] * conc['k_AV'] * 4 * np.pi * (R_normal * 1e-5)**2 #Scaled area of the cell membrane
    
    #Calculation of ionised and neutral vitamin C forms concentrations
    C_c_H = C_Asc_c / (10**(pH_c - conc['pKa']))
    C_a_H = C_Asc_out / (10**(pH_a - conc['pKa']))
    C_c_i = C_Asc_c * (10**(pH_c - conc['pKa'])) / (1 + (10**(pH_c - conc['pKa'])))
    C_a_i = C_Asc_out * (10**(pH_a - conc['pKa'])) / (1 + (10**(pH_a - conc['pKa'])))
    
    J_AscH = conc['P'] * (C_a_H - C_c_H) * A_cancer #Neutral form diffusion
    value = -1 * conc['F'] * psi / (conc['R_const'] * conc['T'])
    J_Asc = -conc['P_Asc'] * value * ((C_a_i - C_c_i * np.exp(value)) / (1 - np.exp(value))) * A_cancer #Ionised form diffusion
    return J_Asc + J_AscH

def J_met(C_in, conc):
    '''
    Parameters
    ----------
    C_in : float 
        instanteneous intracellular concentration.
    conc : dict
        dictionary of parameters.

    Returns
    -------
    TYPE float
        Instanteneous metabolic flux.

    '''
    R_normal = conc['D'] / 2 
    V_cancer = ((4/3) * np.pi * (R_normal * 1e-5)**3) * conc['V_water'] * conc['Vexcl']
    k_metabolism = np.log(2)/(15*24) # 1/h, metabolism rate constant 
    
    return -k_metabolism * C_in * conc['k_fac_met']*V_cancer

def extracellular_forcing(t):
    '''
    Parameters
    ----------
    t : float
        time

    Returns
    -------
    TYPE float
        instanteneous extracellular concentration derivative.
        
    The corresponding constants are the result of exponential function fitting to extracellular conenteration data. 
    
    '''
    return -0.06 * 1100 * np.exp(-t * 0.06)

def solve_cell_model(t_span, conc):
    '''
    Parameters
    ----------
    t_span : list
        simulation duration.
    
    conc : dict
        dictionary of parameters.

    Returns
    -------
    TYPE numpy arrays
        solutions of differential equations
        
    The function generates the final function used for ODE solving
    '''
    R_normal = conc['D'] / 2 
    V_cancer = ((4/3) * np.pi * (R_normal * 1e-5)**3) * conc['V_water']
    
    def ODE_system(t, y):
        '''
        Parameters
        ----------
        t : list
            simulation time steps.
        
        y : list
            initial conditions.

        Returns
        -------
        TYPE list
            concnetration result at each compartment, at a given timepoint
        '''
        C_Asc_c, C_Asc_out = y
        dC_Asc_c = (J_SVCT1(conc, C_Asc_out) + J_SVCT2(conc, C_Asc_out) + J_diff(conc, C_Asc_c, C_Asc_out) + J_met(C_Asc_c, conc)) / (V_cancer * conc['Vexcl']) 
        dC_Asc_out = extracellular_forcing(t)
        return [dC_Asc_c, dC_Asc_out]
        
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(ODE_system, t_span, conc['initial_cond'], method='RK45', t_eval=t_eval)
    
    # Intracellular scaling, the constant (3.26e-12) represents the cell volume used in the original article from which intracellular data is extracted
    k_scale = V_cancer * conc['Vexcl'] / (3.26e-12)
    
    return sol.t, sol.y[0] * k_scale, sol.y[1]

