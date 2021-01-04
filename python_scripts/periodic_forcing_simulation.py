import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyrates.utility.grid_search import grid_search
from pyrates.utility.visualization import plot_timeseries
from pyrates.utility.data_analysis import welch

"""
Allows to perform numerical simulations of a two population GPe model (arkypallidal and prototypical) 
with periodic forcing and gamma-dstributed axonal delays and bi-exponential synapses. 
Creates a time series plot, a plot of the GPe-intrinsic coupling strengths, and a plot with the power spectral density
of the GPe dynamics.

In the first section, the GPe and stimulation parameters can be customized to simulate the model behavior for different 
dynamic regimes.

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) installed.
"""

#########################
# parameter definitions #
#########################

# simulation parameters
#######################

dt = 1e-3    # initial integration step-size in ms
dts = 0.1    # sampling step-size in ms
T = 2050.0   # length of time integral in ms

# periodic stimulation parameters
#################################

sim_steps = int(np.round(T/dt))
omega = [82.0]                     # stimulation period in ms
alpha = [40.0]                     # stimulation amplitude

# GPe parameters
################

k_gp = 3.0   # base scaling of gpe-instrinsic connections
k_p = 1.5    # relative strength of GPe-p over GPe-a projections
k_i = 0.9    # relative strength of between vs. within population coupling
k = 100.0    # base coupling strength
eta = 100.0  # base background input scaling
param_grid = {
        'k_ae': [10.0*k],
        'k_pe': [10.0*k],
        'k_pp': [k*k_gp*k_p/k_i],
        'k_ap': [k*k_gp*k_p*k_i],
        'k_aa': [k*k_gp/(k_p*k_i)],
        'k_pa': [k*k_gp*k_i/k_p],
        'k_ps': [20.0*k],
        'k_as': [20.0*k],
        'eta_e': [0.02],
        'eta_p': [4.8*eta],
        'eta_a': [-6.5*eta],
        'eta_s': [0.002],
        'delta_p': [90.0],
        'delta_a': [120.0],
        'tau_p': [25],
        'tau_a': [20],
        'omega': omega,
        'alpha': alpha
    }

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
    'omega': {'vars': ['sl_op/t_off'], 'nodes': ['driver']},
    'alpha': {'vars': ['sl_op/alpha'], 'nodes': ['driver']}
}

##############################
# Simulation of GPe dynamics #
##############################

results, result_map = grid_search(
    circuit_template="config_files/gpe/gpe_2pop_driver",
    param_grid=param_grid,
    param_map=param_map,
    simulation_time=T,
    step_size=dt,
    permute_grid=True,
    sampling_step_size=dts,
    inputs={},
    outputs={'r_i': 'gpe_p/gpe_proto_op/R_i'},
    init_kwargs={'backend': 'numpy', 'solver': 'scipy', 'step_size': dt},
    method='RK45'
)

############
# plotting #
############

# Gpe-p firing rate timeseries
fig, ax = plt.subplots(figsize=(8, 4))
plot_timeseries(results, ax=ax)
ax.set_ylabel('GPe-p firing rate per ms')
ax.set_xlabel('time (ms)')
ax.set_xlim([1000.0, T-50.0])
ax.set_ylim([0.01, 0.1])
ax.tick_params(axis='both', which='major')
plt.tight_layout()

# GPe connection strengths
conns = ['k_pp', 'k_ap', 'k_pa', 'k_aa']
connections = pd.DataFrame.from_dict({'value': [param_grid[k][0] for k in conns],
                                      'connection': [r'$J_{pp}$', r'$J_{ap}$', r'$J_{pa}$', r'$J_{aa}$']})
fig2, ax2 = plt.subplots(figsize=(4, 2))
sns.set_color_codes("muted")
sns.barplot(x="value", y="connection", data=connections, color="b")
ax2.set(ylabel="", xlabel="")
ax2.tick_params(axis='x', which='major', labelsize=9)
sns.despine(left=True, bottom=True)
ax2.set_title('GPe Coupling')
plt.tight_layout()

# power-spectral density profile of the simulated time series
fig3, ax3 = plt.subplots(figsize=(6, 3))
results = results * 1e3
results.index = results.index * 1e-3
psds, freqs = welch(results, fmin=1.0, fmax=100.0, tmin=1.0, n_fft=2048, n_overlap=1024)
freq_results = pd.DataFrame(data=np.log(psds.T), index=freqs, columns=['r_i'])
plot_timeseries(freq_results, ax=ax3)
ax3.set_ylabel('log PSD')
ax3.set_xlabel('frequency (Hz)')
ax3.set_ylim([-15.0, 5.0])
plt.tight_layout()

plt.show()
