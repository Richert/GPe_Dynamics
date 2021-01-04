from pyrates.utility.pyauto import PyAuto, get_from_solutions
import numpy as np
import matplotlib.pyplot as plt

"""
Bifurcation analysis of the effect of periodic forcing on a two population GPe model (arkypallidal and prototypical) 
with gamma-dstributed axonal delays and bi-exponential synapses. Creates the bifurcation diagram in Fig. 3 of
(citation).

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
You can pass your auto-07p installation path to the python command ('python gpe_bifurcation_analysis.py custom_path') 
or change the default value of `auto_dir` below.
"""

# config
n_dim = 20
n_params = 25
a = PyAuto("config_files", auto_dir='~/PycharmProjects/auto-07p')
store_params = ['PAR(23)', 'PAR(25)', 'PAR(14)']
store_vars = ['U(2)', 'U(4)', 'U(19)', 'U(20)']

############################
# create initial condition #
############################

# set base strength of GPe coupling (k_gp)
s0_sols, s0_cont = a.run(e='gpe_2pop_forced', c='qif_lc', ICP=[19, 11], NPAR=n_params, name='k_gp',
                         NDIM=n_dim, RL0=0.0, RL1=100.0, NMX=6000, DSMAX=0.5, UZR={19: [30.0]},
                         STOP={'UZ1'}, variables=store_vars, params=store_params)

# set GPe-p projection strength (k_p)
s1_sols, s1_cont = a.run(starting_point='UZ1', ICP=[20, 11], NPAR=n_params, name='k_p',
                         NDIM=n_dim, RL0=0.0, RL1=100.0, NMX=6000, DSMAX=0.5, UZR={20: [1.5]},
                         STOP={'UZ1'}, variables=store_vars, params=store_params, origin=s0_cont)

# set between vs. within population coupling strengths (k_i)
s2_sols, s2_cont = a.run(starting_point='UZ1', c='qif_lc', ICP=[21, 11], NPAR=n_params, name='k_i',
                         NDIM=n_dim, RL0=0.1, RL1=10.0, origin=s1_cont, NMX=6000, DSMAX=0.1, DS='-',
                         UZR={21: [0.9]}, STOP={'UZ1'}, variables=store_vars, params=store_params)

# continuation of eta_p
s3_sols, s3_cont = a.run(starting_point='UZ1', c='qif_lc', ICP=[2, 11], NPAR=n_params,
                         name='c1:eta_p', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=s2_cont,
                         NMX=6000, DSMAX=0.1, UZR={2: [4.8]}, STOP={'UZ1'}, variables=store_vars,
                         params=store_params)

# continuation of eta_a
s4_sols, s4_cont = a.run(starting_point='UZ1', c='qif_lc', ICP=[3, 11], NPAR=n_params,
                         name='c1:eta_a', NDIM=n_dim, RL0=-20.0, RL1=20.0, origin=s3_cont,
                         NMX=6000, DSMAX=0.1, UZR={3: [-6.5]}, STOP={'UZ1'}, DS='-', variables=store_vars,
                         params=store_params)

starting_point = 'UZ1'
starting_cont = s4_cont

###########################################################
# analyze response of oscillating GPe to periodic forcing #
###########################################################

# driver parameter boundaries
alpha_min = 0.0
alpha_max = 100.0
omega_min = 25.0
omega_max = 100.0

# perform 1D parameter continuation of driver strength
alpha_sols, alpha_cont = a.run(starting_point=starting_point, origin=starting_cont, c='qif_lc', ICP=[23, 11],
                               NPAR=n_params, name='alpha', NDIM=n_dim, NMX=2000, DSMAX=0.05, RL0=alpha_min,
                               RL1=alpha_max, STOP={}, UZR={23: [30.0]}, variables=store_vars, params=store_params)

# perform 1D parameter continuation of driver period
omega_sols, omega_cont = a.run(starting_point='UZ1', origin=alpha_cont, c='qif_lc', ICP=[25, 11],
                               NPAR=n_params, name='omega', NDIM=n_dim, NMX=8000, DSMAX=0.05, RL0=omega_min,
                               RL1=omega_max, STOP={}, UZR={25: [77.3]}, bidirectional=True, variables=store_vars,
                               params=store_params)

# perform 2D investigation of torus bifurcations found in 1D continuations
i, j = 0, 0
for s in omega_sols.values():
    if 'TR' in s['bifurcation']:
        i += 1
        p_tmp = f'TR{i}'
        sols_tmp, cont_tmp = a.run(starting_point=p_tmp, origin=omega_cont, c='qif3', ICP=[25, 23, 11],
                                   NPAR=n_params, name=f'{p_tmp}:omega/alpha', NDIM=n_dim, NMX=2000,
                                   DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R21', 'R11'}, UZR={},
                                   bidirectional=True, variables=store_vars, params=store_params)
        bfs = get_from_solutions(['bifurcation'], sols_tmp)
        if "R2" in bfs:
            s_tmp, c_tmp = a.run(starting_point='R21', origin=cont_tmp, c='qif_lc', ICP=[25, 11],
                                 NPAR=n_params, name='c1:omega/R21', NDIM=n_dim, NMX=500, DSMAX=0.001,
                                 RL0=omega_min, RL1=omega_max, STOP={'PD1', 'TR1'}, UZR={},
                                 variables=store_vars, params=store_params, DS='-')
            pds = get_from_solutions(['bifurcation'], s_tmp)
            if "PD" in pds:
                j += 1
                p2_tmp = f'PD{j}'
                s2_tmp, c2_tmp = a.run(starting_point=p2_tmp, origin=c_tmp, c='qif3', ICP=[25, 23, 11],
                                       NPAR=n_params, name=f'{p_tmp}:omega/alpha:{p2_tmp}', NDIM=n_dim, NMX=2000,
                                       DSMAX=0.05, RL0=omega_min, RL1=omega_max, STOP={'BP1', 'R22'}, UZR={},
                                       bidirectional=True, variables=store_vars, params=store_params)

############
# plotting #
############

fig, ax = plt.subplots(figsize=(8.0, 4.0))

# continuation of the torus bifurcation in alpha and omega
i = 1
while i < 11:
    try:
        ax = a.plot_continuation('PAR(25)', 'PAR(23)', cont=f'TR{i}:omega/alpha', ax=ax, ignore=['UZ', 'BP'],
                                 line_color_stable='#148F77', line_color_unstable='#148F77',
                                 custom_bf_styles={'R1': {'marker': 'h', 'color': 'k'},
                                                   'R2': {'marker': 'h', 'color': 'g'},
                                                   'R3': {'marker': 'h', 'color': 'r'},
                                                   'R4': {'marker': 'h', 'color': 'b'}},
                                 line_style_unstable='solid')
        i += 1
    except KeyError:
        i += 1
    j = 1
    while j < 11:
        try:
            ax = a.plot_continuation('PAR(25)', 'PAR(23)', cont=f'TR{i}:omega/alpha:PD{j}', ax=ax, ignore=['UZ', 'BP'],
                                     line_color_stable='#3689c9', line_color_unstable='#3689c9',
                                     line_style_unstable='solid')
            j += 1
        except KeyError:
            j += 1

ax.set_ylabel(r'$\alpha$')
ax.set_xlabel(r'$\omega$')
ax.set_ylim([0.0, 60.0])
ax.set_xlim([28.0, 91.5])

plt.tight_layout()
plt.show()
