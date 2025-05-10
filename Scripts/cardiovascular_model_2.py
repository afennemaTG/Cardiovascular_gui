import numpy as np

class CardiovascularModel:
    def __init__(self, dt = 0.01, real_time = True):

        self.dt = dt

        # Initialize arrays to store model parameters
        self.elastance = np.zeros((2, 10))
        self.resistance = np.zeros(10)
        self.uvolume = np.zeros(10)

        self.elastance[:, 0] = [2.3, np.nan]      # Intra-thoracic arteries
        self.elastance[:, 1] = [1.1, np.nan]       # Extra-thoracic arteries
        self.elastance[:, 2] = [0.01, np.nan]    # Extra-thoracic veins
        self.elastance[:, 3] = [0.03, np.nan]    # Intra-thoracic veins
        self.elastance[:, 4] = [0.04, 0.15]        # Right atrium (min, max)
        self.elastance[:, 5] = [0.04, 0.60]       # Right ventricle (min, max)
        self.elastance[:, 6] = [0.23, np.nan]     # Pulmonary arteries
        self.elastance[:, 7] = [0.12, np.nan]    # Pulmonary veins
        self.elastance[:, 8] = [0.08, 0.17]        # Left atrium (min, max)
        self.elastance[:, 9] = [0.08, 3]           # Left ventricle (min, max)

        self.resistance = np.array([
            0.18,   # Intra-thoracic arteries
            0.8,   # Extra-thoracic arteries
            0.06,   # Extra-thoracic veins
            0.012,  # Intra-thoracic veins
            0.003,  # Right atrium
            0.003,  # Right ventricles
            0.08,   # Pulmonary arteries
            0.01,  # Pulmonary veins
            0.003,  # Left atrium
            0.006,  # Left ventricle
        ])

        self.uvolume = np.array([
            250,   # Intra-thoracic arteries
            500,   # Extra-thoracic arteries
            1450,  # Extra-thoracic veins
            1335,  # Intra-thoracic veins
            10,    # Right atrium
            50,    # Right ventricle
            90,    # Pulmonary arteries
            490,   # Pulmonary veins
            10,    # Left atrium
            50,    # Left ventricle
        ])

        self.R_ecmo = 3.2 # Resistance of the ECMO circuit
        
        self.dose = 0
        self.dose_adm = 0
        self.fluids = 0

        self.HR_c = 70
        self.P_set = 85
        self.G = [-0.13, 0.09, 0.45]

        # Export variables
        if real_time:
            self.P_intra = 0
            self.elv = 0
            self.P = np.zeros(10)
            self.real_time = True
        else:
            self.P_intra = []
            self.elv = []
            self.P = []
            self.real_time = False
        
        self.Fes_delayed = np.full(int(2/self.dt), 2.66).tolist()
        self.Fev_delayed = np.zeros(int(0.2/self.dt)).tolist()

        self.t0 = 0
        self.t0_resp = 0

    def _apply_fluids(self, t, fluids):
        
        if fluids != 0 and self.fluids == 0:
            self.fluids = fluids
            self.dose_adm = 0
            self.dose = (fluids/50)*self.dt             # Administration speed has to be updated sometime to match timestep of the simulation
            
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
    
    def cardiac_phase(self, t, HR):
            
        t_end = self.t0 + 60/self.HR_c
        phi = t - self.t0

        if t >= t_end:
            phi = 0
            self.HR_c = 60 / self.HP
            self.t0 = t
        
        return phi
    
    def resp_phase(self, t, RR):
        t_end = self.t0_resp + 60/RR
        tau = t - self.t0_resp

        if t >= t_end:
            tau = 0
            self.t0_resp = t

        return tau

    def cardiac_contraction(self, t, HR, adj_elastance):
        
        phi = self.cardiac_phase(t, HR)

        HP = 60/self.HR_c
        Tas = 0.03 + 0.09 * HP
        Tav = 0.01
        Tvs = 0.16 + 0.2 * HP
        Tvs1 = 0.75*Tvs
        Tvs2 = 0.25*Tvs
        T = self.dt 
        
        ncc = (phi / HP) / T

        #ncc = (t % HP) / T
        
        if ncc <= round(Tas / T):
            aaf = np.sin(np.pi * ncc / (Tas / T))
        else:
            aaf = 0

        ela = adj_elastance[0, 8] + (adj_elastance[1, 8] - adj_elastance[0, 8]) * aaf
        era = adj_elastance[0, 4] + (adj_elastance[1, 4] - adj_elastance[0, 4]) * aaf

        if ncc <= round((Tas + Tav) / T):
            vaf = 0
        elif ncc <= round((Tas + Tav + Tvs1) / T):
            vaf = 1 - np.cos(np.pi * (ncc-(Tas + Tav) / T) / (Tvs1 / T))
        elif ncc <= round((Tas + Tav + Tvs) / T):
            vaf = 1 + np.cos(np.pi * (ncc-(Tas + Tav + Tvs1) / T) / ((Tvs2) / T))
        else:  
            vaf = 0

        elv = adj_elastance[0, 9] + (adj_elastance[1, 9] - adj_elastance[0, 9]) * vaf
        erv = adj_elastance[0, 5] + (adj_elastance[1, 5] - adj_elastance[0, 5]) * vaf

        return ela, elv, era, erv

    def cardiac_contraction_DH(self, t, HR, adj_elastance):
        # Cardiac contraction using the double hill model. Parameter values based on DOI: 10.1002/cnm.1466
        
        phi = self.cardiac_phase(t, HR)

        HP = 60/self.HR_c
        m1 = [1.32, 1.32, 1.32, 1.32]
        m2 = [13.1, 27.4, 13.1, 27.4]
        tau1 = [0.11*HP, 0.269*HP, 0.11*HP, 0.269*HP]
        tau2 = [0.18*HP, 0.452*HP, 0.18*HP, 0.452*HP]
        onset = [0, 0.15*HP, 0, 0.15*HP]

        E = np.zeros(4)
        for i, c in enumerate([8, 9, 4, 5]):
            Emin, Emax = adj_elastance[0, c], adj_elastance[1, c]
            t = phi - onset[i] if phi - onset[i] > 0 else 0
            g1, g2 = ((t)/tau1[i])**m1[i], ((t)/tau2[i])**m2[i]
            
            a = 1.516       # Normalization constant derived from the double hill curve to ensure E goes to Emax
            E[i] = Emax*a*(g1/(1+g1) * 1/(1+g2)) + Emin
        
        return E

    def baroreceptor_control(self, P, dVdt, elastance, P_set, X):
        # Baroreceptor control. Values are taken from the paper of Ursino (1998) DOI: 10.1152/ajpheart.1998.275.5.H1733

        Pbaro, dHRv, dHRs, dRs = X[0], X[1], X[2], X[3]

        tz = 6.37                   # Time constant
        tp = 20.76 #2.076           # Time constant
        
        Fas_min = 2.52              # Minimum firing rate afferent pathway
        Fas_max = 47.87             # Maximum firing rate afferent pathway
        Ka = 11.758                 # Slope of the sigmoid function

        Fes_inf = 2.10              # Firing rate efferent pathway (inf)
        Fes_0 = 16.11               # Firing rate efferent pathway (0)
        Kes = 0.0675                # Slope of the sigmoid function
        Fev_0 = 3.2                 # Firing rate efferent vagal pathway (0)
        Fev_inf = 6.3               # Firing rate efferent vagal pathway (inf)
        Kev = 7.06                  # Slope of the sigmoid function
        Fas_0 = 25  

        Ghs = self.G[0]              # Baroreceptor gain heart rate
        Ths = 2.0                   # Time constant for the sympathetic heart rate response to baroreceptor stimulation
        Gv = self.G[1]              # Baroreceptor gain heart rate vagal
        Thv = 1.5                   # Time constant for the vagal heart rate response to baroreceptor stimulation
        Grs = self.G[2]              # Baroreceptor gain resistance
        Trs = 6                     # Time constant for the resistance response to baroreceptor stimulation

        # Measured pressure and afferent pathway
        dPbarodt = (P + tz*(dVdt*elastance) - Pbaro) / tp
        Fas = (Fas_min + Fas_max*np.exp((Pbaro - P_set)/Ka)) / (1 + np.exp((Pbaro - P_set)/Ka))

        # Efferent pathway
        Fes = Fes_inf + (Fes_0 - Fes_inf) * np.exp(-Kes*Fas)
        Fev = (Fev_0 + Fev_inf*np.exp((Fas-Fas_0)/Kev)) / (1 + np.exp((Fas-Fas_0)/Kev))

        self.Fes_delayed.append(max(Fes, 2.66))         # Append to list in order to provide time delay
        self.Fev_delayed.append(Fev)

        sFh = Ghs * (np.log(self.Fes_delayed[-int(2/self.dt)]-2.65+1)-1.1)
        sFv = Gv * (self.Fev_delayed[-int(0.2/self.dt)]-4.66)
        sFr = Grs * (np.log(self.Fes_delayed[-int(2/self.dt)]-2.65+1)-1.1)

        # Change in heart rate
        ddHRv = (sFv - dHRv)/Thv
        ddHRs = (sFh - dHRs)/Ths
        ddRs =  (sFr - dRs)/Trs

        return dPbarodt, ddHRv, ddHRs, ddRs
    
    def calc_ecmo_flow(self, RPM, P_pre, P_after):
        # Parameter values based on clinical experience
        P_max = 600
        dp = 0.0008

        PFc = (P_max / (1 + np.exp(-dp * RPM))) - P_max / 2
        F_ecmo = (P_pre + PFc - P_after) / self.R_ecmo if P_pre + PFc - P_after > 0 else 0

        return F_ecmo
    
    def mechanical_ventilation(self, t, RR = 20, PEEP = 5, P_vent = 6, I_E_ratio = 2):
        PEEP, P_vent = PEEP*0.73556, P_vent*0.73556         # Convert to mmHg

        tau = self.resp_phase(t, RR)
        resp_cycle = 60/RR
        RC = 0.3

        t_insp = resp_cycle/(1+I_E_ratio) # Inspiratory time

        if tau < t_insp:
            P_intra = PEEP + P_vent/(t_insp)*tau
        else: 
            P_intra = PEEP + P_vent*np.exp(-((tau-t_insp)/RC))

        return P_intra

    def export_function(self):
        # Return parameters for GUI
        export_dict = {
            'HR': self.HR_c,
            'P_intra': self.P_intra,
            'elv': self.elv,
            'P': self.P,}
        
        return export_dict

    def ext_st_sp_eq(self, t, x, **kwargs):

        # Split kwargs
        RPM             =       kwargs.get('RPM', 0)  
        contractility   =       kwargs.get('contractility', 1)
        fSVR            =       kwargs.get('SVR', 1)
        fcompl          =       kwargs.get('compliance', 1)
        fluids          =       kwargs.get('fluids', 0)
        HR              =       kwargs.get('HR', 70) 
        P_set           =       kwargs.get('P_set', 85)
        baro_recept     =       kwargs.get('baroreceptor', False)  
        ventilation     =       kwargs.get('ventilation', False)


        # Split state vector
        V = x[:10]
        X_baro = x[10:] 

        self.P_intra = self.mechanical_ventilation(t) if ventilation == True else 0

        self.HP = 60 / HR + X_baro[1] + X_baro[2] if baro_recept == True else 60 / HR
        self.dR = X_baro[3] if baro_recept == True else 0

        # Calculate variables
        adj_elastance = self.adjust_elastance(contractility, fcompl)
        ela, elv, era, erv = self.cardiac_contraction_DH(t, HR, adj_elastance)
        self._apply_fluids(t, fluids)

        if self.dose_adm < self.fluids:
            V[1] += self.dose
            self.dose_adm += self.dose
        else:
            self.dose_adm = 0
            self.fluids = 0

        # Calculate pressures
        P = np.zeros(10)
        P[0] = adj_elastance[0, 0] * (V[0] - self.uvolume[0]) + self.P_intra
        P[1] = adj_elastance[0, 1] * (V[1] - self.uvolume[1])
        P[2] = adj_elastance[0, 2] * (V[2] - self.uvolume[2])
        P[3] = adj_elastance[0, 3] * (V[3] - self.uvolume[3]) + self.P_intra
        P[4] = era * (V[4] - self.uvolume[4])
        P[5] = erv * (V[5] - self.uvolume[5]) 
        P[6] = adj_elastance[0, 6] * (V[6] - self.uvolume[6]) + self.P_intra
        P[7] = adj_elastance[0, 7] * (V[7] - self.uvolume[7]) + self.P_intra
        P[8] = ela * (V[8] - self.uvolume[8]) 
        P[9] = elv * (V[9] - self.uvolume[9]) 

        # Calculate flows
        F = np.zeros(10)
        F[0] = (P[0] - P[1]) / (self.resistance[0] * fSVR)          
        F[1] = (P[1] - P[2]) / (self.resistance[1] * fSVR + self.dR)
        F[2] = (P[2] - P[3]) / (self.resistance[2] * fSVR) #if P[2] - P[3] > 0 else (P[2] - P[3]) / (10 * self.resistance[2])
        F[3] = (P[3] - P[4]) / self.resistance[3] #if P[3] - P[4] > 0 else (P[3] - P[4]) / (10 * self.resistance[3])
        F[4] = max((P[4] - P[5]) / self.resistance[4], 0)
        F[5] = max((P[5] - P[6]) / self.resistance[5], 0)
        F[6] = (P[6] - P[7]) / (self.resistance[6] * fSVR)
        F[7] = (P[7] - P[8]) / self.resistance[7] #if P[7] - P[8] > 0 else (P[7] - P[8]) / (10 * self.resistance[7])
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

        if baro_recept == True:
            dBarodt = self.baroreceptor_control(P[0], dVdt[0], adj_elastance[0,0], P_set, X_baro)
        else:
            dBarodt = 0, 0, 0, 0

        # Combine all derivatives
        dxdt = np.zeros(len(dVdt) + len(dBarodt))
        dxdt[:len(dVdt)] = dVdt
        dxdt[len(dVdt):] = dBarodt

        # Export variables
        if self.real_time:
            self.elv = elv
            self.P = P
        else:
            self.elv.append(elv)
            self.P.append((t, P))

        return dxdt







