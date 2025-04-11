import numpy as np
#from baroreceptor_sim import baroreceptor_control

class CardiovascularModel:
    def __init__(self, dt = 0.01):

        self.dt = dt
        self.baro_recept = False

        # Initialize arrays to store model parameters
        self.elastance = np.zeros((2, 10))
        self.resistance = np.zeros(10)
        self.uvolume = np.zeros(10)

        # Initialize model parameters
        self.elastance[:, 0] = [1.43, np.nan]      # Intra-thoracic arteries
        self.elastance[:, 1] = [0.8, np.nan]       # Extra-thoracic arteries
        self.elastance[:, 2] = [0.0169, np.nan]    # Extra-thoracic veins
        self.elastance[:, 3] = [0.0182, np.nan]    # Intra-thoracic veins
        self.elastance[:, 4] = [0.05, 0.12]        # Right atrium (min, max)
        self.elastance[:, 5] = [0.057, 0.40]       # Right ventricle (min, max)
        self.elastance[:, 6] = [0.233, np.nan]     # Pulmonary arteries
        self.elastance[:, 7] = [0.0455, np.nan]    # Pulmonary veins
        self.elastance[:, 8] = [0.12, 0.4]        # Left atrium (min, max)
        self.elastance[:, 9] = [0.09, 3]           # Left ventricle (min, max)

        self.resistance = np.array([
            0.2,   # Intra-thoracic arteries
            0.5,   # Extra-thoracic arteries
            0.09,   # Extra-thoracic veins
            0.003,  # Intra-thoracic veins
            0.003,  # Right atrium
            0.003,  # Right ventricle
            0.11,   # Pulmonary arteries
            0.003,  # Pulmonary veins
            0.003,  # Left atrium
            0.008,  # Left ventricle
        ])

        self.uvolume = np.array([
            140,   # Intra-thoracic arteries
            370,   # Extra-thoracic arteries
            1000,  # Extra-thoracic veins
            1190,  # Intra-thoracic veins
            14,    # Right atrium
            26,    # Right ventricle
            50,    # Pulmonary arteries
            350,   # Pulmonary veins
            11,    # Left atrium
            20,    # Left ventricle
        ])

        self.t_stop = 0
        self.dose = 0

        self.sn_dot = np.zeros(7)
        self.HR = 70

        self.P_set = 85
        self.control_params = {
            "Gc_hr": 0.9,
            "tau_hr": 105,
            "Gc_r": 0.05,
            "tau_r": 205, 
        }

        self.Fes_delayed = np.full(int(2/self.dt), 2.66).tolist()
        self.Fev_delayed = np.zeros(int(0.2/self.dt)).tolist() #np.full(int(0.2/self.dt), 4.66).tolist()

        self.ncc = 0
        self.HR_c = 70

    def _apply_fluids(self, t, fluids):
        
        if fluids != 0:
            self.t_stop = t + 5
            self.dose = (fluids/5)*self.dt

    def adjust_elastance(self, contractility, fcompl):

        adj_elastance = self.elastance.copy()
        adj_elastance[1, 4] *= contractility
        adj_elastance[1, 5] *= contractility
        adj_elastance[1, 8] *= contractility
        adj_elastance[1, 9] *= contractility

        # adj_elastance[0, 5] *= diastolic_dysfunction
        # adj_elastance[0, 9] *= diastolic_dysfunction

        adj_elastance[0, 0] *= 1/fcompl   
        adj_elastance[0, 1] *= 1/fcompl
        adj_elastance[0, 6] *= 1/fcompl
    
        return adj_elastance
    
    def cardiac_contraction(self, t, HR, adj_elastance):
        HP = 60/HR
        Tas = 0.03 + 0.09 * HP
        Tav = 0.01
        Tvs = 0.16 + 0.2 * HP
        Tvs1 = 0.75*Tvs
        Tvs2 = 0.25*Tvs
        T = self.dt 
    
        self.ncc = (t % HP) / T
        
        if self.ncc <= round(Tas / T):
            aaf = np.sin(np.pi * self.ncc / (Tas / T))
        else:
            aaf = 0

        ela = adj_elastance[0, 8] + (adj_elastance[1, 8] - adj_elastance[0, 8]) * aaf
        era = adj_elastance[0, 4] + (adj_elastance[1, 4] - adj_elastance[0, 4]) * aaf

        if self.ncc <= round((Tas + Tav) / T):
            vaf = 0
        elif self.ncc <= round((Tas + Tav + Tvs1) / T):
            vaf = 1 - np.cos(np.pi * (self.ncc-(Tas + Tav) / T) / (Tvs1 / T))
        elif self.ncc <= round((Tas + Tav + Tvs) / T):
            vaf = 1 + np.cos(np.pi * (self.ncc-(Tas + Tav + Tvs1) / T) / ((Tvs2) / T))
        else:  
            vaf = 0

        elv = adj_elastance[0, 9] + (adj_elastance[1, 9] - adj_elastance[0, 9]) * vaf
        erv = adj_elastance[0, 5] + (adj_elastance[1, 5] - adj_elastance[0, 5]) * vaf

        return ela, elv, era, erv
    
    def baroreceptor_control(self, P, dVdt, elastance, P_set, Pbaro, dHRv, dHRh):
        # Baroreceptor control
        tz = 6.37
        tp = 20.76 #2.076
        
        Fas_min = 2.52
        Fas_max = 47.87
        Ka = 11.758

        # Measured pressure and afferent pathway
        dPbarodt = (P + tz*(dVdt*elastance) - Pbaro) / tp
        Fas = (Fas_min + Fas_max*np.exp((Pbaro - P_set)/Ka)) / (1 + np.exp((Pbaro - P_set)/Ka))

        Fes_inf = 2.10
        Fes_0 = 16.11
        Kes = 0.0675
        Fev_0 = 3.2
        Fev_inf = 6.3
        Kev = 7.06 
        Fas_0 = 25

        # Efferent pathway
        Fes = Fes_inf + (Fes_0 - Fes_inf) * np.exp(-Kes*Fas)
        Fev = (Fev_0 + Fev_inf*np.exp((Fas-Fas_0)/Kev)) / (1 + np.exp((Fas-Fas_0)/Kev))

        self.Fes_delayed.append(max(Fes, 2.66))
        self.Fev_delayed.append(Fev)

        Gh = -0.13     # Heart rate
        Ths = 20        #2.0

        Gv = 0.09
        Thv = 15        #1.5

        sFh = Gh * (np.log(self.Fes_delayed[-int(2/self.dt)]-2.65+1)-1.1)
        sFv = Gv * (self.Fev_delayed[-int(0.2/self.dt)]-4.66)

        ddHRv = (sFv - dHRv)/Thv
        ddHRh = (sFh - dHRh)/Ths

        return dPbarodt, ddHRv, ddHRh

    def baroreceptor_control_old(self, P, P_set, Delta_HR_c, Delta_R_c):

        dDelta_HR_c = (-Delta_HR_c + self.control_params['Gc_hr'] * (P_set-P)) / self.control_params['tau_hr']  # Heart rate
        dDelta_R_c = (-Delta_R_c + self.control_params['Gc_r'] * (P_set-P)) / self.control_params['tau_r']  # Resistance

        return dDelta_HR_c, dDelta_R_c
    
    def calc_ecmo_flow(self, RPM, P_pre, P_after):
        PFc = 300*np.exp(0.002*RPM)/(300+np.exp(0.002*RPM))
        R_ecmo = 2                            
        F_ecmo = (P_pre + PFc - P_after) / R_ecmo if P_pre + PFc - P_after > 0 else 0

        return F_ecmo

    
    # Return parameters for GUI
    def return_values(self):
        return self.HR_c, self.ncc


    def ext_st_sp_eq(self, t, x, **kwargs):

        # Split kwargs
        RPM             =       kwargs.get('F_ecmo', 0)               #* 1000/60
        contractility   =       kwargs.get('contractility', 1)
        fSVR            =       kwargs.get('SVR', 1)
        fcompl          =       kwargs.get('compliance', 1)
        fluids          =       kwargs.get('fluids', 0)
        HR              =       kwargs.get('HR', 70) 
        P_set           =       kwargs.get('P_set', 85)


        # Split state vector
        V = x[:10]
        Pbaro = x[10] 
        DHRv = x[11] 
        DHRh = x[12]

        # Only update during diastole to prevent corrupt contractions
        HP_c = 60/HR + DHRh + DHRv
        if self.ncc > 0.9*(HP_c/self.dt):
            self.HR_c = 60/HP_c
        else: self.HR_c
        R_c = fSVR #+ DR

        # Calculate variables
        adj_elastance = self.adjust_elastance(contractility, fcompl)
        ela, elv, era, erv = self.cardiac_contraction(t, self.HR_c, adj_elastance)
        self._apply_fluids(t, fluids)

        if t<self.t_stop:
            V[1] += self.dose

        # Calculate pressures
        P = np.zeros(10)
        P[0] = adj_elastance[0, 0] * (V[0] - self.uvolume[0])
        P[1] = adj_elastance[0, 1] * (V[1] - self.uvolume[1])
        P[2] = adj_elastance[0, 2] * (V[2] - self.uvolume[2])
        P[3] = adj_elastance[0, 3] * (V[3] - self.uvolume[3]) 
        P[4] = era * (V[4] - self.uvolume[4]) 
        P[5] = erv * (V[5] - self.uvolume[5]) 
        P[6] = adj_elastance[0, 6] * (V[6] - self.uvolume[6]) 
        P[7] = adj_elastance[0, 7] * (V[7] - self.uvolume[7]) 
        P[8] = ela * (V[8] - self.uvolume[8]) 
        P[9] = elv * (V[9] - self.uvolume[9]) 

        # Calculate flows
        F = np.zeros(10)
        F[0] = (P[0] - P[1]) / (self.resistance[0] * R_c)          
        F[1] = (P[1] - P[2]) / (self.resistance[1] * R_c)
        F[2] = (P[2] - P[3]) / (self.resistance[2] * R_c)
        F[3] = (P[3] - P[4]) / self.resistance[3] if P[3] - P[4] > 0 else (P[3] - P[4]) / (10 * self.resistance[3])
        F[4] = max((P[4] - P[5]) / self.resistance[4], 0)
        F[5] = max((P[5] - P[6]) / self.resistance[5], 0)
        F[6] = (P[6] - P[7]) / self.resistance[6]
        F[7] = (P[7] - P[8]) / self.resistance[7] if P[7] - P[8] > 0 else (P[7] - P[8]) / (10 * self.resistance[7])
        F[8] = max((P[8] - P[9]) / self.resistance[8], 0)
        F[9] = max((P[9] - P[0]) / self.resistance[9], 0)   

        F_ecmo = self.calc_ecmo_flow(RPM, P[3], P[1])

        # Derivatives of volumes
        dVdt = np.zeros(10)
        dVdt[0] = F[9] - F[0]
        dVdt[1] = F[0] - F[1] + F_ecmo
        dVdt[2] = F[1] - F[2]
        dVdt[3] = F[2] - F[3] - F_ecmo
        dVdt[4] = F[3] - F[4]
        dVdt[5] = F[4] - F[5]
        dVdt[6] = F[5] - F[6]
        dVdt[7] = F[6] - F[7]
        dVdt[8] = F[7] - F[8]
        dVdt[9] = F[8] - F[9]

        if self.baro_recept == True:
            dBarodt = self.baroreceptor_control(P[0], dVdt[0], adj_elastance[0,0], P_set, Pbaro, DHRv, DHRh)
        else:
            dBarodt = 0, 0, 0
            self.HR_c = HR

        # Combine all derivatives
        dxdt = np.zeros(len(dVdt) + len(dBarodt))
        dxdt[:len(dVdt)] = dVdt
        dxdt[len(dVdt):] = dBarodt

        return dxdt







