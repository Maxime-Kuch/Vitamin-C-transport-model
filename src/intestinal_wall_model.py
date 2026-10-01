import numpy as np
from scipy.integrate import solve_ivp


##################  FLUX FUNCTIONS  ##################

def J_SVCT1(conc, C_Asc_a):
    '''

    Parameters
    ----------
    conc : dict
        Dictionary of parameters.
    C_Asc_a : float
        instanteneous apical concentration (Concentration in mucus at x=d).

    Returns
    -------
    TYPE float
        instanteneous flux through SVCT1 transporter.

    '''
    pH_a = conc['pH_a']
    C_a_i = C_Asc_a * (10**(pH_a - conc['pKa'])) / (1 + (10**(pH_a - conc['pKa'])))
    return (conc['V_max1_cell'] * C_a_i / (conc['K_m1'] + C_a_i))

def J_SVCT2(conc, C_Asc_b):
    '''

    Parameters
    ----------
    conc : dict
        Dictionary of parameters.
    C_Asc_b : float
        instanteneous concentration in interstitial fluid.

    Returns
    -------
    TYPE float
        instanteneous flux through SVCT2 transporter.

    '''
    pH_ecf = conc['pH_ecf']
    C_b_i = C_Asc_b * (10**(pH_ecf - conc['pKa'])) / (1 + (10**(pH_ecf - conc['pKa'])))
    return (conc['V_max2_cell'] * C_b_i / (conc['K_m2'] + C_b_i))

def J_diff_a(conc, C_Asc_c, C_Asc_a):
    '''

    Parameters
    ----------
    conc : dict
        Dictionary of parameters.
    C_Asc_c : float
        instanteneous intracellular concentration.
    C_Asc_a : float
        instanteneous apical concentration (Concentration in mucus at x=d).

    Returns
    -------
    TYPE float
        instanteneous total diffusive flux through apical membrane.

    '''
    
    pH_a, pH_c, psi = conc['pH_a'], conc['pH_c'], conc['psi_a']
    z = -1
    C_c_H = C_Asc_c / (10**(pH_c - conc['pKa']))
    C_a_H = C_Asc_a / (10**(pH_a - conc['pKa']))
    C_c_i = C_Asc_c * (10**(pH_c - conc['pKa'])) / (1 + (10**(pH_c - conc['pKa'])))
    C_a_i = C_Asc_a * (10**(pH_a - conc['pKa'])) / (1 + (10**(pH_a - conc['pKa'])))
    J_AscH = conc['P'] * (C_a_H - C_c_H) * conc['A_a']
    value = z * conc['F'] * psi / (conc['R'] * conc['T'])
    J_Asc = -conc['P_Asc'] * value * ((C_a_i - C_c_i * np.exp(value)) / (1 - np.exp(value))) * conc['A_a']
    return J_Asc + J_AscH

def J_diff_b(conc, C_Asc_c, C_Asc_b):
    '''

    Parameters
    ----------
    conc : dict
        Dictionary of parameters.
    C_Asc_c : float
        instanteneous intracellular concentration.
    C_Asc_b : float
        instanteneous concentration in interstitial fluid.

    Returns
    -------
    TYPE float
        instanteneous total diffusive flux through basolateral membrane.

    '''
    
    pH_ecf, pH_c, psi = conc['pH_ecf'], conc['pH_c'], conc['psi_bl']
    z = -1
    C_c_H = C_Asc_c / (10**(pH_c - conc['pKa']))
    C_b_H = C_Asc_b / (10**(pH_ecf - conc['pKa']))
    C_c_i = C_Asc_c * (10**(pH_c - conc['pKa'])) / (1 + (10**(pH_c - conc['pKa'])))
    C_b_i = C_Asc_b * (10**(pH_ecf - conc['pKa'])) / (1 + (10**(pH_ecf - conc['pKa'])))
    J_AscH = conc['P'] * (C_b_H - C_c_H) * conc['A_bl']
    value = z * conc['F'] * psi / (conc['R'] * conc['T'])
    J_Asc = -conc['P_Asc'] * value * ((C_b_i - C_c_i * np.exp(value)) / (1 - np.exp(value))) * conc['A_bl']
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
    k_metabolism = np.log(2)/(15*24) # 1/h, metabolism rate constant 
    
    return -k_metabolism * C_in * conc['V_c']

def J_cap(C_Asc_b, C_phys, conc):
    '''
    Parameters
    ----------
    C_Asc_b : float
       instanteneous concentration in interstitial fluid.
    C_phys : float
        blood concentration.
    conc : dict
        dictionary of parameters.

    Returns
    -------
    TYPE float
        Flux through endothelial cell barier.

    '''
    return conc['P_spec'] * (C_phys - C_Asc_b) * conc['A_bl'] * conc['k_d']

def J_impulse(t, type_func, Cmax, time_span):
    '''
    Parameters
    ----------
    t : float
        timepoint.
    type_func : string
        type of function used for concentration impulse.
    Cmax : float
        maximal concentration.
    time_span : float
        impulse duration.

    Returns
    -------
    TYPE float
        instanteneous concentration at given timepoint.
    
    The function is used for generation of concentration impulse (C_blood(t) and C_int(t) functions).
    '''
    b = 250.5
    if type_func == 'gauss':
        return Cmax * np.exp((-(t - b)**2) / (2 * (time_span**2)))
    elif type_func == 'square':
        k = 200 
        return Cmax * (1/(1 + np.exp(-k*(t - b))) - 1/(1 + np.exp(-k*(t - (b + time_span)))))
    return 0.0


def simulate_transport(y0, t_span, conc, uptake_side='intestine'):
    '''
    Parameters
    ----------
    y0 : list
        initial conditions.
    t_span : list
        timespan of simulation.
    conc : dict
        dictionary of parameters.
    uptake_side : TYPE string, optional
        determines where impuls generation occurs ('intestine' or 'blood'). The default is 'intestine'.

    Returns
    -------
    TYPE dict
        Dictionary of simulation results.
        
    The function generates function used for solving of model equations.

    '''
    N = conc['N']
    dx = (conc['d_muc'] / N) * 10 

    def Model(t, y):
        '''
        Parameters
        ----------
        t : numpy array
            timesteps of simulation.
        y : list
            initial conditions.

        Returns
        -------
        dydt : list
            list of solver results.
        
        The function solves model equations, the partial differential equtions are transformed into set of ODEs
        that are solved together with ODEs for other compartments (See "Implementation of mucus layer with method 
        of lines" in the SI)
        '''
        C_muc = y[:N]
        C_Asc_c = y[N]
        C_Asc_b = y[N+1]
        
        if uptake_side == 'intestine':
            C_lumen = J_impulse(t, 'square', conc['ADM_int'], conc['impuls_time'])
            C_blood_phys = conc.get('ADM_blood', 50) 
        else: 
            C_lumen = 0 
            C_blood_phys = conc.get('ADM_blood', 50) + J_impulse(t, 'square', conc['ADM_blood_impulse'], conc['impuls_time'])

        dydt = np.zeros_like(y)
        dC_muc = np.zeros(N)
        D_m = conc['D_muc']
        
        dC_muc[0] = (D_m / dx**2) * (C_lumen - 2 * C_muc[0] + C_muc[1])
        dC_muc[1:-1] = (D_m / dx**2) * (C_muc[:-2] - 2 * C_muc[1:-1] + C_muc[2:])
        
        C_Asc_a = C_muc[-1] 
        
        J_svct1 = J_SVCT1(conc, C_Asc_a)
        J_dif_a = J_diff_a(conc, C_Asc_c, C_Asc_a)
        J_svct2 = J_SVCT2(conc, C_Asc_b)
        J_dif_b = J_diff_b(conc, C_Asc_c, C_Asc_b)
        J_cap_flux = J_cap(C_Asc_b, C_blood_phys, conc)
        J_lumen_out = (D_m / dx) * (C_muc[0] - C_lumen)
        J_metab = J_met(C_Asc_c, conc)
        
        dC_muc[-1] = ((D_m / dx) * (C_muc[-2] - C_muc[-1]) - J_svct1 - J_dif_a) / dx
        
        dC_Asc_c = (J_svct1 + J_svct2 + J_dif_a + J_dif_b+ J_metab) / conc['V_c'] 
        dC_Asc_b = (-J_svct2 - J_dif_b + J_cap_flux) / conc['V_ecf']
        dC_Asc_blo = 0 
        
        if t >= 250.5:
            dQ_capillary_out = -J_cap_flux if uptake_side == 'intestine' else J_cap_flux 
            dQ_lumen_out = J_lumen_out if uptake_side == 'blood' else -J_lumen_out
        else:
            dQ_capillary_out, dQ_lumen_out = 0,0

        dydt[:N] = dC_muc
        dydt[N] = dC_Asc_c
        dydt[N+1] = dC_Asc_b
        dydt[N+2] = dC_Asc_blo
        dydt[N+3] = dQ_capillary_out
        dydt[N+4] = dQ_lumen_out
        
        return dydt

    t_eval = np.linspace(t_span[0], t_span[1], 7000)
    sol = solve_ivp(Model, t_span, y0, method='BDF', t_eval=t_eval, max_step=0.2)

    return {
        't': sol.t,
        'mucus': sol.y[:N],          
        'enterocyte': sol.y[N],      
        'isf': sol.y[N+1],           
        'blood': sol.y[N+2],         
        'Q_capillary_out': sol.y[N+3],
        'Q_lumen_out': sol.y[N+4]          
    }

