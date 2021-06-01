import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search
from copy import deepcopy
from scipy.ndimage.filters import gaussian_filter1d
from pyrates.utility.visualization import plot_timeseries

"""
Allows to perform numerical simulations of a two population GPe model (arkypallidal and prototypical) 
with gamma-distributed axonal delays and bi-exponential synapses. Creates a time series of the GPe firing rates under 
different conditions (see Fig. 1 in Gast et al. (2021) JNS)

In the first section, the GPe model parameters can be customized to simulate the model behavior for different 
dynamic regimes.
For replication of the timeseries in Fig. 2, 3 and 4 of Gast et al. (2021) JNS, adjustments need to be made regarding
the input parameters and the gamma-distributed axonal delays (the latter can be done in the gpe.yaml file found in 
config_files).

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) installed.
"""

#########################
# parameter definitions #
#########################

# simulation parameters
#######################

dt = 1e-3    # initial integration step-size in ms
dts = 1e-1   # sampling step-size in ms
T = 5000.0   # length of time integral in ms

# input parameters
##################

sim_steps = int(np.round(T/dt))
offset1 = int(np.round(600.0/dt))      # input start in ms
dur = int(np.round(50.0/dt))           # input duration in ms
offset2 = int(np.round((1200.0)/dt))   # second input start in ms
dur2 = dur                             # second input duration in ms
amp1 = 500.0                           # input strength
amp2 = -amp1                           # second input strength
gaussian_var = 10.0                    # variance of Gaussian kernel used to smooth the input

ctx = np.zeros((sim_steps, 1))
ctx[offset1:offset1+dur, 0] = amp1
ctx[offset2:offset2+dur2, 0] = amp2
ctx = gaussian_filter1d(ctx, gaussian_var, axis=0)

str_period = 77                        # period of periodic striatal stimulation in ms
str_amp = 15.0                         # amplitude of periodic striatal stimulation in ms

# GPe parameters
################

k_gp = 1.0   # base scaling of gpe-instrinsic connections
k = 10.0     # base coupling strength
param_grid = {
        'k_ae': [k*1.5],
        'k_pe': [k*5.0],
        'k_pp': [1.5*k*k_gp],
        'k_ap': [2.0*k*k_gp],
        'k_aa': [0.1*k*k_gp],
        'k_pa': [0.5*k*k_gp],
        'k_ps': [10.0*k*k_gp],
        'k_as': [1.0*k*k_gp],
        'eta_e': [0.02],
        'eta_p': [12.0],
        'eta_a': [26.0],
        'eta_s': [0.002],
        'delta_p': [9.0],
        'delta_a': [3.0],
        'tau_p': [18],
        'tau_a': [32],
        #'omega': [str_period],
        #'alpha': [str_amp]
    }
param_grid = pd.DataFrame.from_dict(param_grid)

param_map = {
    'k_ae': {'vars': ['weight'], 'edges': [('stn', 'gpe_a')]},
    'k_pe': {'vars': ['weight'], 'edges': [('stn', 'gpe_p')]},
    'k_pp': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_p')]},
    'k_ap': {'vars': ['weight'], 'edges': [('gpe_p', 'gpe_a')]},
    'k_aa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_a')]},
    'k_pa': {'vars': ['weight'], 'edges': [('gpe_a', 'gpe_p')]},
    'k_ps': {'vars': ['weight'], 'edges': [('str', 'gpe_p')]},
    'k_as': {'vars': ['weight'], 'edges': [('str', 'gpe_a')]},
    'eta_p': {'vars': ['gpe_proto_op/eta_i'], 'nodes': ['gpe_p']},
    'eta_a': {'vars': ['gpe_arky_op/eta_a'], 'nodes': ['gpe_a']},
    'eta_e': {'vars': ['stn_dummy_op/eta_e'], 'nodes': ['stn']},
    'eta_s': {'vars': ['str_dummy_op/eta_s'], 'nodes': ['str']},
    'delta_p': {'vars': ['gpe_proto_op/delta_i'], 'nodes': ['gpe_p']},
    'delta_a': {'vars': ['gpe_arky_op/delta_a'], 'nodes': ['gpe_a']},
    'tau_p': {'vars': ['gpe_proto_op/tau_i'], 'nodes': ['gpe_p']},
    'tau_a': {'vars': ['gpe_arky_op/tau_a'], 'nodes': ['gpe_a']},
    #'omega': {'vars': ['sl_op/t_off'], 'nodes': ['driver']},
    #'alpha': {'vars': ['sl_op/alpha'], 'nodes': ['driver']}
}

conditions = [{},  # healthy control -> GPe-p: 60 Hz, GPe-a: 10 Hz
              {'eta_s': 20.0},  # STR excitation -> GPe-p: 10 Hz, GPe-a: 40 Hz
              {'eta_e': 0.1},  # STN inhibition -> GPe-p: 30 Hz, GPe_a: 20 Hz
              {'k_pp': 0.1, 'k_pa': 0.1, 'k_aa': 0.1, 'k_ap': 0.1, 'k_ps': 0.1,
               'k_as': 0.1},  # GABAA blockade in GPe -> GPe_p: 100 Hz
              {'k_pe': 0.1, 'k_pp': 0.1, 'k_pa': 0.1, 'k_ae': 0.1, 'k_aa': 0.1, 'k_ap': 0.1,
               'k_ps': 0.1, 'k_as': 0.1},  # AMPA blockade and GABAA blockade in GPe -> GPe_p: 70 Hz
              ]

# simulations
#############

for c_dict in deepcopy(conditions):
    for key in param_grid:
        if key in c_dict:
            c_dict[key] = np.asarray(param_grid[key]) * c_dict[key]
        elif key in param_grid:
            c_dict[key] = np.asarray(param_grid[key])
    param_grid_tmp = pd.DataFrame.from_dict(c_dict)
    results, result_map = grid_search(
        circuit_template="config_files/gpe/gpe_2pop",
        param_grid=param_grid_tmp,
        param_map=param_map,
        simulation_time=T,
        step_size=dt,
        permute=True,
        sampling_step_size=dts,
        inputs={
            #'gpe_p/gpe_proto_syns_op/I_ext': ctx,
            #'gpe_a/gpe_arky_syns_op/I_ext': ctx
            },
        outputs={
            'r_i': 'gpe_p/gpe_proto_op/R_i',
            'r_a': 'gpe_a/gpe_arky_op/R_a',
        },
        init_kwargs={
            'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
        method='RK45',
    )

    fig2, ax = plt.subplots(figsize=(6, 2.0))
    results = results * 1e3
    plot_timeseries(results, ax=ax)
    plt.legend(['GPe-p', 'GPe-a'])
    ax.set_ylabel('Firing rate')
    ax.set_xlabel('time (ms)')
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    plt.show()
