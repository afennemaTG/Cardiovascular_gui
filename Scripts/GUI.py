import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.integrate import solve_ivp, RK45
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from cardiovascular_model_2 import CardiovascularModel
from presets import pre_sets


def solve_ode(t_span, x0, dict):

    # solver = RK45(
    #     fun = lambda t,x: CM.ext_st_sp_eq(t,x, **dict),
    #     t0 = t_span[0],
    #     y0 = x0,
    #     t_bound = t_span[1],
    #     max_step = 0.005,
    # )

    # solver.step()
    # state = solver.y
    # t = solver.t

    solution = solve_ivp(lambda t,x: CM.ext_st_sp_eq(t,x, **dict), t_span, x0, method='RK45', rtol=1e-8, atol=1e-8)

    dict['fluids'] = 0 

    return solution      #solver

def calc_pressures(pressure):

    if len(pressure) < 2:
        return 0,0,0,0
    
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


class ODEGuiApp:

# MODEL INITIALIZATION
    def __init__(self, root, **kwargs):
        root.title(kwargs.get('title', 'Cardiovascular Model GUI'))
        self.dark_mode = kwargs.get('dark_mode', False)
        self.running = False
        self.time_elapsed = 0

        self.t = 0
        self.dt = kwargs.get('dt', 0.01)    
        self.save = False
        self.par_adjusted = True
        self.stab_thres = 5

        self.buffer_counter = 0
        self.buffer_interval = int(0.05/self.dt)

        if self.dark_mode:
            self.clr, self.clr_text = 'black', 'white'
            root.configure(bg='black')
        else:
            self.clr, self.clr_text = 'white', 'black'
        
        self.init_model()
        self.init_plot()
        self.init_controls()

        root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_controls(self):    
        self.interaction_frame = tk.Frame(root)
        self.interaction_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.interaction_frame.rowconfigure(0, weight=1)
        self.interaction_frame.rowconfigure(1, weight=1)
        self.interaction_frame.rowconfigure(2, weight=1)
        self.interaction_frame.columnconfigure(0, weight=1)
        self.interaction_frame.columnconfigure(1, weight=1)
        self.interaction_frame.columnconfigure(2, weight=1)

        self.text_display = self.ax_pressure_time.text(1.02, 0.6, "", transform=self.ax_pressure_time.transAxes, fontsize=14, color=self.clr_text, verticalalignment="top")
        self.text_display2 = self.ax_pressure_time.text(1.02, 0.4, "", transform=self.ax_pressure_time.transAxes, fontsize=10, color=self.clr_text, verticalalignment="top")
        
        controls_frame = tk.Frame(self.interaction_frame)
        controls_frame.grid(row=0, column=1)
        ttk.Button(controls_frame, text="Start", command=self.start).pack(side=tk.LEFT, padx=10)
        ttk.Button(controls_frame, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=10)
        ttk.Button(controls_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=10)

        save_frame = ttk.Frame(self.interaction_frame) 
        save_frame.grid(row=1, column=1, sticky='N')
        self.save_button = ttk.Button(save_frame, text="Save plot", command=self.saver)
        self.save_button.pack(side=tk.LEFT, padx=10)
        self.clear_button = ttk.Button(save_frame, text="Clear plot", command=self.clear_plot)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        self.rescale_button = ttk.Button(save_frame, text="Rescale", command=self.rescale)
        self.rescale_button.pack(side=tk.LEFT, padx=10)

        param_frame = ttk.Frame(self.interaction_frame)
        param_frame.grid(row=2, column=0, sticky='N')
        self.sliders = {}
        self.sliders['SVR'] = self.add_slider(param_frame, "SVR", 0.5, 2.0, self.dict['SVR'], self.update_svr)
        self.sliders['F_ecmo'] = self.add_slider(param_frame, "ECMO flow", 0, 5000, self.dict['F_ecmo'], self.update_flow)
        self.sliders['contractility'] = self.add_slider(param_frame, "Contractility", 0.1, 2.0, self.dict['contractility'], self.update_contractility)
        self.sliders['compliance'] = self.add_slider(param_frame, "Compliance", 0.5, 2.0, self.dict['compliance'], self.update_compliance)
        self.sliders['HR'] = self.add_slider(param_frame, "Heart rate", 40, 120, self.dict['HR'], self.update_hr)
        
        self.fluid_button_500 =ttk.Button(param_frame, text="Give 500ml fluids", command=lambda: self.update_fluid(500))
        self.fluid_button_500.pack(side=tk.LEFT, padx=5)
        self.fluid_button_1000 = ttk.Button(param_frame, text="Give 1000ml fluids", command=lambda: self.update_fluid(1000))
        self.fluid_button_1000.pack(side=tk.LEFT, padx=5)

        buttons_frame = ttk.Frame(self.interaction_frame)
        buttons_frame.grid(row=2, column=1)
        self.baro_button = tk.Button(buttons_frame, text="Baroreceptor OFF", bg='Tomato', command=self.toggle_baroreceptor)
        self.baro_button.pack(side=tk.TOP, pady=10)
        self.vent_button = tk.Button(buttons_frame, text="Ventilation OFF", bg='Tomato', command=self.toggle_ventilation)
        self.vent_button.pack(side=tk.TOP, pady=10)
        preset_names = ["normal", "cardiogenic shock", "septic shock", 'case 1', 'case 2']
        self.preset_menu = ttk.OptionMenu(buttons_frame, tk.StringVar(), "Presets", *preset_names, command=self.pre_set)
        self.preset_menu.pack(side=tk.TOP, pady=10)

        empty_frame = ttk.Frame(self.interaction_frame)
        empty_frame.grid(row=2, column=2)
        self.empty_label = ttk.Label(empty_frame, 
                                     text="Disclaimer: \nThis simulation is for educational purposes only \n and should not be used for medical purposes.",
                                     justify='center',
                                     font=("Arial Black", 10))
        self.empty_label.pack(side=tk.TOP, padx=10)

    def init_model(self):
        self.dict = {'F_ecmo': 0, 
                     'contractility': 1, 
                     'SVR': 1, 
                     'compliance': 1, 
                     'fluids': 0, 
                     'HR': 70, 
                     'P_set': 85, 
                     'baroreceptor': False,
                     'ventilation': False}

        global CM
        CM = CardiovascularModel(self.dt)

        self.TBV = 5000
        self.current_state = np.zeros(13)
        self.current_state[:10] = self.TBV * (CM.uvolume / np.sum(CM.uvolume))
        self.current_state[10] = 85
        self.HR = self.dict['HR']

        self.ao_pressures, self.time_values = [], []
        self.lv_pressures, self.lv_volumes =  [], []
        self.old_time_values, self.old_pressure_values = [], []
        self.adj_elastance = CM.adjust_elastance(self.dict['contractility'], self.dict['compliance'])

        self.dbp_arr = [0,0]
        self.sbp_arr = [0,0]

    def init_plot(self):
        self.plot_frame = tk.Frame(root, bg=self.clr)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Get screen resolution
        dpi = int(root.winfo_fpixels('1i'))  
        scr_width = root.winfo_screenwidth() / dpi
        scr_height = root.winfo_screenheight() / dpi

        self.fig = plt.figure(figsize=(scr_width, scr_height*0.6), dpi=dpi, facecolor=self.clr)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.4, left=0.05, right=0.95)
        self.ax_pressure_time = self.fig.add_subplot(gs[0])
        self.ax_pressure_volume = self.fig.add_subplot(gs[1])
        
        self.min_val, self.max_val = 80, 100
        self.x_lim, self.y_lim = 100, 100

        self.ax_pressure_time.set_facecolor(self.clr)
        self.ax_pressure_time.set_title("Pressure over Time", color=self.clr_text)
        self.ax_pressure_time.set_xlabel("Time (s)", color=self.clr_text)
        self.ax_pressure_time.set_ylabel("Pressure (mmHg)", color=self.clr_text)
        self.ax_pressure_time.tick_params(axis='both', colors=self.clr_text)
        self.ax_pressure_time.set_xlim(0, 5)

        self.ax_pressure_volume.set_facecolor(self.clr)
        self.ax_pressure_volume.set_title("LV pressure-volume loop", color=self.clr_text)
        self.ax_pressure_volume.set_xlabel("Volume (mL)", color=self.clr_text)
        self.ax_pressure_volume.set_ylabel("Pressure (mmHg)", color=self.clr_text)
        self.ax_pressure_volume.tick_params(axis='both', colors=self.clr_text)
        self.ax_pressure_volume.set(xlim=(0,self.x_lim), ylim=(0,self.y_lim))

        self.ax_pressure_time.tick_params(length=0) if self.dark_mode else None
        self.ax_pressure_volume.tick_params(length=0) if self.dark_mode else None

        self.line_pt, = self.ax_pressure_time.plot([], [], lw=2, color='tab:blue')
        self.old_line, = self.ax_pressure_time.plot([], [], lw=2, color='tab:blue')
        self.saved_line_pt, = self.ax_pressure_time.plot([], [], lw=2, color='tab:blue', alpha=0.2)

        self.line_pv, = self.ax_pressure_volume.plot([], [], lw=2, color='tab:red')
        self.saved_line_pv, = self.ax_pressure_volume.plot([], [], lw=2, color='tab:red', alpha=0.2)	

        # Stabalizing overlay
        self.overlay = self.ax_pressure_time.axhspan(0, 300, color='gray', alpha=0.4, zorder=5)
        self.text = self.ax_pressure_time.text(0.5, 0.5, "Stabilizing...", color='white', fontsize=16, ha='center', va='center', 
                                          transform=self.ax_pressure_time.transAxes, zorder=6)
        self.overlay.set_visible(False)
        self.text.set_visible(False)  
        self.stabalizing = False
                
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        plt.tight_layout()
        self.clear_plot()
        
    def add_slider(self, parent, label, min_val, max_val, var, command):
        frame = ttk.Frame(parent)
        frame.pack(anchor=tk.SW, fill=tk.X, pady=5)

        ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)
        
        var = tk.DoubleVar(value=var)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, command=command, orient="horizontal", length=200)
        slider.pack(side=tk.LEFT)

        # Store label for updating
        if label == "SVR":
            self.svr_label = ttk.Label(frame, text=f"{var.get():.1f}")
            self.svr_label.pack(side=tk.LEFT, padx=5)
        elif label == "ECMO flow":
            self.flow_label = ttk.Label(frame, text=f"{var.get():.0f} RPM")        #L/min
            self.flow_label.pack(side=tk.LEFT, padx=5)
        elif label == "Contractility":
            self.contractility_label = ttk.Label(frame, text=f"{100*var.get():.0f}%")
            self.contractility_label.pack(side=tk.LEFT, padx=5)
        elif label == "Compliance":
            self.compliance_label = ttk.Label(frame, text=f"{var.get():.1f}")
            self.compliance_label.pack(side=tk.LEFT, padx=5)
        elif label == "Heart rate":
            self.hr_label = ttk.Label(frame, text=f"{var.get():.0f} bpm")
            self.hr_label.pack(side=tk.LEFT, padx=5)

        return slider

# MODEL INPUT CONTROLS    
    def update_svr(self, value):
        self.dict['SVR'] = float(value)
        self.svr_label.config(text=f"{float(value):.1f}")

        self.par_adjusted = True 
    
    def update_flow(self, value):
        self.dict['F_ecmo'] = float(value)
        self.flow_label.config(text=f"{float(value):.0f} RPM")     #L/min

        self.par_adjusted = True 
    
    def update_contractility(self, value):
        self.dict['contractility'] = float(value)
        self.adj_elastance = CM.adjust_elastance(self.dict['contractility'], self.dict['compliance'])
        self.contractility_label.config(text=f"{100*float(value):.0f} %")

        self.par_adjusted = True   
    
    def update_compliance(self, value):
        self.dict['compliance'] = float(value)
        self.adj_elastance = CM.adjust_elastance(self.dict['contractility'], self.dict['compliance'])
        self.compliance_label.config(text=f"{float(value):.1f}")

        self.par_adjusted = True 

    def update_hr(self, value):
        self.dict['HR'] = float(value)
        self.hr_label.config(text=f"{float(value):.0f} bpm")   

        self.par_adjusted = True 

    def update_fluid(self, ml):
        self.dict['fluids'] = ml
        self.fluid_button_500.config(state=tk.DISABLED)
        self.fluid_button_1000.config(state=tk.DISABLED)
        root.after(10000, self.fluid_button_500.config, {'state': tk.NORMAL})
        root.after(10000, self.fluid_button_1000.config, {'state': tk.NORMAL})
        self.TBV = self.TBV + ml

        self.par_adjusted = True 
    
    def toggle_baroreceptor(self):
        if self.dict['baroreceptor']:
            self.dict['baroreceptor'] = False
            self.baro_button.config(text="Baroreceptor OFF", bg='tomato')
        else:
            self.dict['baroreceptor'] = True
            self.baro_button.config(text="Baroreceptor ON", bg='chartreuse3')

        self.par_adjusted = True

    def toggle_ventilation(self):
        if self.dict['ventilation']:
            self.dict['ventilation'] = False
            self.vent_button.config(text="Ventilation OFF", bg='tomato')
            self.stab_thres = 5
        else:
            self.dict['ventilation'] = True
            self.vent_button.config(text="Ventilation ON", bg='chartreuse3')
            self.stab_thres = 8

        self.par_adjusted = True
        
    def pre_set(self, name):

        self.par_adjusted = True
        pre_set_dict = pre_sets(name)
        self.dict.update({k: pre_set_dict[k] for k in pre_set_dict if k in self.dict})

        for key, slider in self.sliders.items():
            slider.set(self.dict[key]) 

# PLOT CONTROLS
    def start(self):
        if not self.running:
            self.running = True
            self.run_simulation()
    
    def stop(self):
        self.running = False
    
    def reset(self):
        self.running = False
        self.time_elapsed = 0
        self.t = 0

        self.clear_plot()
        self.init_model()

        self.line_pt.set_data([], [])
        self.line_pv.set_data([], [])
        self.saved_line_pt.set_data([], [])
        self.saved_line_pv.set_data([], [])
        self.old_line.set_data([], [])
        self.text_display.set_text("")
        self.text_display2.set_text("")
        self.overlay.set_visible(False)
        self.text.set_visible(False)
        self.canvas.draw()

        for key, slider in self.sliders.items():
            slider.set(self.dict[key])
        self.baro_button.config(text="Baroreceptor OFF", bg='tomato') 

    def saver(self):
        self.save = not self.save

        if self.save:
            self.save_button.config(text="Stop saving")
            self.saved_time_values.append(self.time_elapsed)
            self.saved_ao_pressure.append(np.nan)
            self.saved_lv_volume.append(np.nan)
            self.saved_lv_pressure.append(np.nan)  
        else:
            self.save_button.config(text="Save plot")
        
    def clear_plot(self):
        self.saved_lv_volume = []
        self.saved_lv_pressure = []
        self.saved_ao_pressure = []
        self.saved_time_values = []

        if hasattr(self, 'saved_line_pt') and hasattr(self, 'saved_line_pv'):
            self.saved_line_pt.set_data([], [])
            self.saved_line_pv.set_data([], [])
            self.canvas.draw()
    
    def rescale(self):
        self.min_val = min(self.ao_pressures[-int(5/self.dt):])-20
        self.max_val = max(self.ao_pressures[-int(5/self.dt):])+20
        self.ax_pressure_time.set_ylim(self.min_val, self.max_val)

        self.x_lim = max(self.lv_volumes[-int(1/self.dt):])+50
        self.y_lim = max(self.lv_pressures[-int(1/self.dt):])+20
        self.ax_pressure_volume.set_xlim(0, self.x_lim)
        self.ax_pressure_volume.set_ylim(0, self.y_lim)

    def update_plot(self):

        self.buffer_counter = 0 
        skip = int(0.15/self.dt)
        beat = int(60/(self.HR*self.dt))

        # Update the plot lines
        if hasattr(self, 'old_time_values') and hasattr(self, 'old_pressure_values'):
            self.old_line.set_data(self.old_time_values[skip:], self.old_pressure_values[skip:])

        # Update current line
        self.line_pt.set_data(self.time_values, self.ao_pressures[-len(self.time_values):])
        self.line_pv.set_data(self.lv_volumes[-(beat-skip):], self.lv_pressures[-(beat-skip):])

        if self.save:
            self.saved_line_pt.set_data(self.saved_time_values, self.saved_ao_pressure)
            self.saved_line_pv.set_data(self.saved_lv_volume, self.saved_lv_pressure)

        sbp, dbp, map, pp = calc_pressures(self.ao_pressures[-int(beat*1.5):])
        self.cardiac_output = calc_co(self.lv_volumes[-int(beat*1.5):], self.HR)
        
        self.text_display.set_text(f"{sbp:.0f}/{dbp:.0f} \n({map:.0f})")
        self.text_display2.set_text(
            f"PP: {pp:.0f} mmHg\n"
            f"CO: {self.cardiac_output:.1f} L/min\n"
            f"TBV: {self.TBV:.0f} ml\n"
            f"HR: {self.HR:.0f} bpm\n"
            f"ECMO flow: {self.F_ecmo:.1f} L/min")

        self.min_val = min(self.ao_pressures[-1]-10, self.min_val)
        self.max_val = max(self.ao_pressures[-1]+10, self.max_val)
        self.ax_pressure_time.set_ylim(self.min_val, self.max_val)

        self.x_lim = max(self.lv_volumes[-1]+50, self.x_lim)
        self.y_lim = max(self.lv_pressures[-1]+10, self.y_lim)
        self.ax_pressure_volume.set_xlim(0, self.x_lim)
        self.ax_pressure_volume.set_ylim(0, self.y_lim)
        
        self.stabilize(sbp, dbp)

        self.canvas.draw_idle()

    def stabilize(self, dbp, sbp):
        # Compare with old values
        dbp_change = dbp != self.dbp_arr[-1]
        sbp_change = sbp != self.sbp_arr[-1]

        if dbp_change:
            self.dbp_arr.append(dbp)
        elif sbp_change:
            self.sbp_arr.append(sbp)

        if dbp_change or sbp_change:
            dbp_dif = abs(self.dbp_arr[-1] - self.dbp_arr[-2])
            sbp_dif = abs(self.sbp_arr[-1] - self.sbp_arr[-2])

            if dbp_dif < self.stab_thres and sbp_dif < self.stab_thres:
                self.overlay.set_visible(False)
                self.text.set_visible(False)
                self.par_adjusted = False
            else: 
                self.overlay.set_visible(True)
                self.text.set_visible(True)

        elif self.par_adjusted == True:
            self.overlay.set_visible(True)
            self.text.set_visible(True)
    
# MAIN FUNCTION
    def run_simulation(self):
        if self.running:

            self.t += self.dt
            t_span = (self.t, self.t + self.dt)

            self.HR = CM.return_values()
            
            solution = solve_ode(t_span, self.current_state, self.dict)
            ela = CM.cardiac_contraction(self.t, self.HR, self.adj_elastance)[1]

            ao_pressure = self.adj_elastance[0,0] * (solution.y[0,-1] - CM.uvolume[0])
            lv_volume = solution.y[9,-1]
            lv_pressure = ela*(lv_volume - CM.uvolume[9])
            
            P_pa = self.adj_elastance[0,1]*(solution.y[1,-1] - CM.uvolume[1])
            P_cv = self.adj_elastance[0,3]*(solution.y[3,-1] - CM.uvolume[3])
            self.F_ecmo = CM.calc_ecmo_flow(self.dict['F_ecmo'], P_cv, P_pa)
            self.F_ecmo = self.F_ecmo*60/1000  # Convert to L/min
                
            # Store values
            self.ao_pressures.append(ao_pressure)
            self.lv_volumes.append(lv_volume)
            self.lv_pressures.append(lv_pressure)

            self.time_values.append(self.time_elapsed*self.dt)   
            self.current_state = solution.y[:,-1]

            # Store plot vlaues
            if self.time_values[-1] > 5: 

                self.old_time_values = self.time_values[-int(5/self.dt):].copy()
                self.old_pressure_values = self.ao_pressures[-int(5/self.dt):].copy()
                
                self.time_values = [0]
                self.time_elapsed = 0
                self.ao_pressures = self.old_pressure_values        # Store maximum 5 seconds of data

            # Remove old values 
            self.old_time_values = self.old_time_values[1:]
            self.old_pressure_values = self.old_pressure_values[1:]    
            
            # Save plot values
            if self.save == True:
                if self.time_elapsed == 0:     
                    self.saved_ao_pressure.append(np.nan)
                    self.saved_time_values.append(0)
                else:
                    self.saved_ao_pressure.append(ao_pressure)
                    self.saved_time_values.append(self.time_elapsed*self.dt)
                
                self.saved_lv_volume.append(self.lv_volumes[-1])
                self.saved_lv_pressure.append(self.lv_pressures[-1])
            

            self.time_elapsed += 1
            self.buffer_counter += 1
            
            if self.buffer_counter >= self.buffer_interval:
                self.update_plot()
            
            root.after(1, self.run_simulation)
    
    def on_closing(self):
        self.running = False  # Stop the simulation loop
        root.quit()
        root.destroy()

if __name__ == "__main__":
    time_step = 0.005
    root = tk.Tk()
    root.state('zoomed')
    app = ODEGuiApp(root, dt = time_step, dark_mode=False)
    root.mainloop()
