import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from cardiovascular_model_2 import CardiovascularModel

import multiprocessing as mp
from functools import partial
from scipy.interpolate import interp1d

# Initialize the model to access its parameters

def define_parameters(CM):
    parameters = {
        'C_ao': CM.elastance[0, 0],           # Intra-thoracic arteries
        'C_ea': CM.elastance[0, 1],           # Extra-thoracic arteries
        'C_ev': CM.elastance[0, 2],           # Extra-thoracic veins
        'C_iv': CM.elastance[0, 3],           # Intra-thoracic veins
        'C_ra_min': CM.elastance[0, 4],       # Right atrium (min)
        'C_rv_min': CM.elastance[0, 5],       # Right ventricle (min)
        'C_pa': CM.elastance[0, 6],           # Pulmonary arteries
        'C_pv': CM.elastance[0, 7],           # Pulmonary veins
        'C_la_min': CM.elastance[0, 8],       # Left atrium (min)
        'C_lv_min': CM.elastance[0, 9],       # Left ventricle (min)
        
        # 'C_ra_max': CM.elastance[1, 4],       # Right atrium (max)
        # 'C_rv_max': CM.elastance[1, 5],       # Right ventricle (max)
        # 'C_la_max': CM.elastance[1, 8],       # Left atrium (max)
        # 'C_lv_max': CM.elastance[1, 9],       # Left ventricle (max)
        
        'R_ao': CM.resistance[0],             # Intra-thoracic arteries
        'R_ea': CM.resistance[1],             # Extra-thoracic arteries
        'R_ev': CM.resistance[2],             # Extra-thoracic veins
        'R_iv': CM.resistance[3],             # Intra-thoracic veins
        'R_ra': CM.resistance[4],             # Right atrium
        'R_rv': CM.resistance[5],             # Right ventricle
        'R_pa': CM.resistance[6],             # Pulmonary arteries
        'R_pv': CM.resistance[7],             # Pulmonary veins
        'R_la': CM.resistance[8],             # Left atrium
        'R_lv': CM.resistance[9],             # Left ventricle
        
        'V0_ao': CM.uvolume[0],              # Intra-thoracic arteries
        'V0_ea': CM.uvolume[1],              # Extra-thoracic arteries
        'V0_ev': CM.uvolume[2],              # Extra-thoracic veins
        'V0_iv': CM.uvolume[3],              # Intra-thoracic veins
        'V0_ra': CM.uvolume[4],              # Right atrium
        'V0_rv': CM.uvolume[5],              # Right ventricle
        'V0_pa': CM.uvolume[6],              # Pulmonary arteries
        'V0_pv': CM.uvolume[7],              # Pulmonary veins
        'V0_la': CM.uvolume[8],              # Left atrium
        'V0_lv': CM.uvolume[9],              # Left ventricle

        'G_baro': CM.G,                     # Baroreceptor gain
        'TBV': 5700,                      # Total blood volume
    }
    return parameters

def solve_ode(parameters, key, base_val, scale, user_dict, dt):
    
    CM_solve = CardiovascularModel(real_time=False)
    TBV = 5700

    t_span=(0.0, 20.0)
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt) 


    if key.startswith('C_'):
        indx = list(parameters.keys()).index(key)
        CM_solve.elastance[0, indx] = base_val*(1/scale)
    if key.startswith('R_'):
        indx = list(parameters.keys()).index(key) - 10
        CM_solve.resistance[indx] = base_val*scale
    if key.startswith('V0_'):
        indx = list(parameters.keys()).index(key) - 20
        TBV = TBV - (CM_solve.uvolume[indx] - base_val*scale)
        CM_solve.uvolume[indx] = base_val*scale
    if key.startswith('G_'): 
        CM_solve.G = np.array(base_val)*scale
        user_dict['baroreceptor'] = True
        user_dict['contractility'] = 0.5
        t_span = (0.0, 40)
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    if key.startswith('TBV'):
        TBV = base_val*scale 

    X = np.zeros(14)
    X[:10] = TBV * (CM_solve.uvolume / np.sum(CM_solve.uvolume))

    sol = solve_ivp(lambda t, x: CM_solve.ext_st_sp_eq(t, x, **user_dict),
                    t_span, X, t_eval=t_eval, method='LSODA',
                    rtol=1e-8, atol=1e-8)

    return define_output_parameters(user_dict, sol, dt, CM_solve)


def define_output_parameters(dict, sol, dt, CM_solve):
    beats = int(5*(CM_solve.HP)/dt)
    volume = sol.y[0, -beats:]
    pressure = CM_solve.elastance[0,0] * (volume - CM_solve.uvolume[0])

    sbp, dbp, map, pp = calc_pressures(pressure)
    max_lv_volume, min_lv_volume, max_lv_pressure, min_lv_pressure, max_pa_pressure, min_pa_pressure, max_cv_pressure, min_cv_pressure = calc_output(sol, beats, CM_solve)
    CO = calc_co(volume, dict['HR'])

    return max_lv_volume, min_lv_volume, max_lv_pressure, sbp, dbp, map, pp, max_cv_pressure, min_cv_pressure, CO


def simulate_apw(user_dict, dt=0.005):

    CM = CardiovascularModel(real_time=False)
    parameters = define_parameters(CM)
    scales = np.array([0.5, 0.75, 1, 1.5, 2])

    dict_out = {}
    
    try:
        
        with mp.Pool(processes=mp.cpu_count()) as pool:

            for key, base_val in parameters.items():
                print(f'Analyzing {key}...')

                partial_func = partial(solve_ode, parameters, key, base_val,
                                    user_dict=user_dict, dt=dt)

                results = pool.map(partial_func, scales)

                max_lv_volume, min_lv_volume, max_lv_pressure, sbp, dbp, map, pp, max_cv_pressure, min_cv_pressure, CO = zip(*results)

                dict_out[key] = {
                    'max_lv_volume': max_lv_volume,
                    'min_lv_volume': min_lv_volume,
                    'max_lv_pressure': max_lv_pressure,
                    'sbp': sbp,
                    'dbp': dbp,
                    'map': map,
                    'pp': pp,
                    'max_cv_pressure': max_cv_pressure,
                    'min_cv_pressure': min_cv_pressure,
                    'CO': CO}

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user. Cleaning up...")
        pool.terminate()
        pool.join()

    return dict_out

def calc_output(sol, beats, CM_solve):

    t_model, P = zip(*CM_solve.P)
    P_model = np.array(P).T

    P = np.zeros((len(P_model), len(sol.t)))
    for i,x in enumerate(P_model):
        interp_func = interp1d(t_model, x, kind='linear', fill_value='extrapolate')
        P[i] = interp_func(sol.t)

    max_lv_volume = max(sol.y[9, -beats:])
    min_lv_volume = min(sol.y[9, -beats:])
    max_lv_pressure = max(P[9, -beats:])
    min_lv_pressure = min(P[9, -beats:])
    max_pa_pressure = max(P[6, -beats:])
    min_pa_pressure = min(P[6, -beats:])
    max_cv_pressure = max(P[3, -beats:])
    min_cv_pressure = min(P[3, -beats:])

    return max_lv_volume, min_lv_volume, max_lv_pressure, min_lv_pressure, max_pa_pressure, min_pa_pressure, max_cv_pressure, min_cv_pressure


def calc_pressures(pressure):
    
    pressure_array = np.array(pressure)
    dPP = np.diff(pressure_array)
    sign_change = np.where(np.diff(np.sign(dPP)) != 0)[0]

    if len(sign_change) < 2:
        return 0,0,0,0
    
    sbp = np.max(pressure_array[sign_change])
    dbp = np.min(pressure_array[sign_change])
    pp = sbp - dbp
    map = dbp + pp/3

    return sbp, dbp, map, pp


def calc_co(volumes, HR):
    
    if len(volumes) < 2:
        return 0

    dV = np.diff(volumes)
    sign_change = np.where(np.diff(np.sign(dV)) != 0)[0]

    if len(sign_change) < 2:
        return 0
    
    volumes = np.array(volumes)
    SV = np.max(volumes[sign_change]) - np.min(volumes[sign_change]) 
    CO = SV/1000 * HR

    return CO


if __name__ == "__main__":
    input_dict = {
        'RPM': 0,
        'contractility': 1,
        'SVR': 1,
        'compliance': 1,
        'fluids': 0,
        'HR': 70,
        'P_set': 85,
        'baroreceptor': False
    }

    results = simulate_apw(input_dict)

    flattened_results = []

    for key, metrics in results.items():
        for scale_idx, scale_value in enumerate([0.5, 0.75, 1, 1.5, 2]):
            flattened_results.append({
                'Parameter': key,
                'Scale': scale_value,
                'max V_lv': metrics['max_lv_volume'][scale_idx],
                'min V_lv': metrics['min_lv_volume'][scale_idx],
                'max P_lv': metrics['max_lv_pressure'][scale_idx],
                'SBP': metrics['sbp'][scale_idx],
                'DBP': metrics['dbp'][scale_idx],
                'MAP': metrics['map'][scale_idx],
                'PP': metrics['pp'][scale_idx],
                'max P_cv': metrics['max_cv_pressure'][scale_idx],
                'min P_cv': metrics['min_cv_pressure'][scale_idx],
                'CO': metrics['CO'][scale_idx]
            })

    # Convert to DataFrame
    df = pd.DataFrame(flattened_results)

    root = r""
    df.to_excel(fr"{root}\sensitivity_analysis_final.xlsx", sheet_name='Sensitivity_analysis', index=False)
    df.to_csv(fr"{root}\sensitivity_analysis_final.csv", index=False)

