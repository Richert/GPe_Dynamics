from pyrates.utility.pyauto import PyAuto
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import sys

"""
Bifurcation analysis of GPe model with two populations (arkypallidal and prototypical) and 
gamma-dstributed axonal delays and bi-exponential synapses. Creates the bifurcation diagrams of Fig. 1 and 2 of
(citation).

To run this code, you need Python >= 3.6 with PyRates (https://github.com/pyrates-neuroscience/PyRates) and 
auto-07p (https://github.com/auto-07p/auto-07p) installed.
You can pass your auto-07p installation path to the python command ('python gpe_bifurcation_analysis.py custom_path') 
or change the default value of `auto_dir` below.
"""

path = sys.argv[-1]
auto_dir = path if type(path) is str and ".py" not in path else "~/PycharmProjects/auto-07p"

# config
n_dim = 18
n_params = 23
a = PyAuto("auto_files", auto_dir=auto_dir)

# choose for which GPe coupling patterns the analysis should be run
###################################################################

c1 = False  # strong bidirectional coupling between GPe-p and GPe-a
c2 = False  # weak bidirectional coupling between GPe-p and GPe-a
c3 = False  # c2 plus strong GPe-p projections
c4 = True  # c1 plus strong GPe-p projections

############################
# create initial condition #
############################

# initial numerical integration over time
t_sols, t_cont = a.run(e='gpe_2pop', c='ivp', ICP=14, DS=5e-3, DSMIN=1e-4, DSMAX=1.0, NMX=100000, name='t',
                       UZR={14: 10000.0}, STOP=['UZ1'], NDIM=n_dim, NPAR=n_params)


# set base strength of GPe coupling (k_gp)
s0_sols, s0_cont = a.run(starting_point='UZ1', c='qif', ICP=19, NPAR=n_params, name='k_gp', NDIM=n_dim,
                         RL0=0.99, RL1=100.0, origin=t_cont, NMX=6000, DSMAX=0.1,
                         UZR={19: [30.0]}, STOP=['UZ1'])

# set relative strength of between vs. within population coupling (k_i)
s1_sols, s1_cont = a.run(starting_point='UZ1', c='qif', ICP=21, NPAR=n_params, name='k_i', NDIM=n_dim,
                         RL0=0.1, RL1=10.0, origin=s0_cont, NMX=6000, DSMAX=0.1, bidirectional=True,
                         UZR={21: [0.7, 0.9, 1.8, 2.2]}, STOP={})

#########################################################################
# c1: investigation of GPe behavior for strong GPe-a <-> GPe-p coupling #
#########################################################################

if c1:

    starting_point = 'UZ4'
    starting_cont = s1_cont

    # set condition-specific non-free model parameters
    ##################################################

    # set background input of GPe-p (eta_p)
    c1_b1_sols, c1_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c1:eta_p',
                             NDIM=n_dim, RL0=-30.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.05,
                             UZR={2: [4.0]}, STOP=['UZ1'])

    starting_point = 'UZ1'
    starting_cont = c1_b1_cont

    # main bifurcation analysis
    ###########################

    # perform 1D parameter continuation of GPe-a background input (eta_a)
    c1_b2_sols, c1_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c1:eta_a',
                                   NDIM=n_dim, RL0=-40.0, RL1=20.0, origin=starting_cont, NMX=8000, DSMAX=0.01,
                                   bidirectional=True)

    # plotting
    ##########

    fig1 = plt.figure(tight_layout=True, figsize=(6, 2))
    grid1 = gs.GridSpec(1, 1)

    # 1D parameter continuation of GPe-a background input (eta_a)
    ax = fig1.add_subplot(grid1[0, 0])
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c1:eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', ignore=['UZ'])
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel('GPe-p firing rate per ms')
    ax.set_title('1D Bifurcation diagram')
    ax.set_xlim([0.0, 0.5])
    ax.set_ylim([0.02, 0.085])

#######################################################################
# c2: investigation of GPe behavior for weak GPe-p <-> GPe-a coupling #
#######################################################################

if c2:

    starting_point = 'UZ1'
    starting_cont = s1_cont

    # set condition-specific non-free model parameters
    ##################################################

    # set background input of GPe-p (eta_p)
    c2_b1_sols, c2_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params, name='c2:eta_p',
                             NDIM=n_dim, RL0=-5.0, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                             bidirectional=True, UZR={2: [4.5]}, STOP=['UZ1'])

    starting_point = 'UZ1'
    starting_cont = c2_b1_cont

    # main bifurcation analysis
    ###########################

    # perform 1D parameter continuation of GPe-a background input (eta_a)
    c2_b2_sols, c2_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params, name='c2:eta_a',
                                   NDIM=n_dim, RL0=-25.0, RL1=15.0, origin=starting_cont, NMX=8000, DSMAX=0.01,
                                   bidirectional=True, UZR={})

    # continuation of periodic orbit of hopf bifurcations found in previous step
    c2_b2_lc1_sols, c2_b2_lc1_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], NPAR=n_params,
                                           name='c2:eta_a:lc', NDIM=n_dim, RL0=-12.0, RL1=0.0, origin=c2_b2_cont,
                                           NMX=6000, DSMAX=0.02, STOP={'BP1', 'PD1'})
    c2_b2_lc2_sols, c2_b2_lc2_cont = a.run(starting_point='HB2', c='qif2b', ICP=[3, 11], NPAR=n_params,
                                           name='c2:eta_a:lc2', NDIM=n_dim, RL0=0.0, RL1=12.0, origin=c2_b2_cont,
                                           NMX=6000, DSMAX=0.02, STOP={'BP1', 'PD1', 'LP3'})

    # plotting
    ##########

    fig2 = plt.figure(tight_layout=True, figsize=(6, 2))
    grid2 = gs.GridSpec(1, 1)

    # 1D parameter continuation of GPe-a background input (eta_a) with minima and maxima of periodic orbits
    ax = fig2.add_subplot(grid2[0, 0])
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c2:eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', ignore=['UZ'])
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c2:eta_a:lc', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             custom_bf_styles={'LP': {'marker': 'p'}})
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont='c2:eta_a:lc2', ax=ax, ignore=['BP'], line_color_stable='#148F77',
                             custom_bf_styles={'LP': {'marker': 'p'}})
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel('GPe-p firing rate per ms')
    ax.set_title('1D Bifurcation diagram')
    ax.set_xlim([-10.0, 10.0])
    ax.set_ylim([0.02, 0.1])

##########################################################################
# c3: investigation of GPe behavior for c2 with strong GPe-p projections #
##########################################################################

if c3:

    starting_point = 'UZ2'
    starting_cont = s1_cont

    # set condition-specific non-free model parameters
    ##################################################

    # set GPe-p projection strength (k_p)
    c3_b1_sols, c3_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='c3:k_p',
                                   NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   UZR={20: [1.5]}, STOP=['UZ1'])

    starting_point = 'UZ1'
    starting_cont = c3_b1_cont

    # set background input of GPe-p (eta_p)
    c3_b2_sols, c3_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                   name=f'c3:eta_p', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=starting_cont,
                                   NMX=6000, DSMAX=0.1, UZR={2: [4.8]}, STOP=['UZ1'])

    starting_point = 'UZ1'
    starting_cont = c3_b2_cont

    # main bifurcation analysis
    ###########################

    # perform 1D parameter continuation of GPe-a background input (eta_a)
    c3_b3_sols, c3_b3_cont = a.run(starting_point=starting_point, c='qif', ICP=3, NPAR=n_params,
                                   name=f'c3:eta_a', NDIM=n_dim, RL0=-12.0, RL1=2.0,
                                   origin=starting_cont, NMX=8000, DSMAX=0.01, bidirectional=True, UZR={3: [-2.0]})

    # continue limit cycles of hopf bifurcation in eta_a
    c3_b3_lc_sols, c1_b3_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[3, 11], NPAR=n_params,
                                         name=f'c3:eta_a:lc', NDIM=n_dim, origin=c3_b3_cont, NMX=2000, DSMAX=0.02,
                                         RL0=-12.0)

    # perform 1D parameter continuation of GPe-a to GPe-p projection (k_pa) for eta_a = -2.0
    c3_b4_sols, c3_b4_cont = a.run(starting_point='UZ1', origin=c3_b3_cont, ICP=8, NPAR=n_params,
                                   NDIM=n_dim, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.01, bidirectional=True,
                                   name='c3:k_pa')

    # continue limit cycles of hopf bifurcation in k_pa
    c3_b4_lc_sols, c3_b4_lc_cont = a.run(starting_point='HB1', c='qif2b', ICP=[8, 11], NPAR=n_params,
                                         name=f'c3:k_pa:lc', NDIM=n_dim, origin=c3_b4_cont, NMX=2000, DSMAX=0.02,
                                         RL0=0.0)

    # perform 2D parameter continuation for hopf bifurcation found in k_pa
    c3_b4_2d1_sols, c1_b4_2d1_cont = a.run(starting_point='HB1', origin=c3_b4_cont, c='qif2', ICP=[8, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c3:k_pa/k_pp', bidirectional=True)
    c3_b4_2d2_sols, c1_b4_2d2_cont = a.run(starting_point='HB1', origin=c3_b4_cont, c='qif2', ICP=[8, 7], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c3:k_pa/k_ap', bidirectional=True)

    # plotting
    ##########

    fig3 = plt.figure(tight_layout=True, figsize=(6.0, 6.0))
    grid3 = gs.GridSpec(3, 2)

    # 1D parameter continuation of GPe-a background input (eta_a) with minima and maxima of periodic orbits
    ax = fig3.add_subplot(grid3[0, :])
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont=f'c3:eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', custom_bf_styles={'HB': {'color': 'k'}})
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont=f'c3:eta_a:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', custom_bf_styles={'HB': {'color': 'k'}})
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel('GPe-p firing rate per ms')
    ax.set_title('1D bifurcation diagram')
    ax.set_xlim([-10.0, 0.0])
    ax.set_ylim([0.0, 0.1])

    # 1D parameter continuation of GPe-a to GPe-p projection (k_pa) with minima and maxima of periodic orbits
    ax = fig3.add_subplot(grid3[1, :])
    ax = a.plot_continuation('PAR(8)', 'U(2)', cont=f'c3:k_pa', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', custom_bf_styles={'HB': {'color': 'k'}})
    ax = a.plot_continuation('PAR(8)', 'U(2)', cont=f'c3:k_pa:lc', ax=ax, line_color_stable='#148F77',
                             line_color_unstable='#148F77', custom_bf_styles={'HB': {'color': 'k'}})
    ax.set_xlabel(r'$k_{pa}$')
    ax.set_ylabel('GPe-p firing rate per ms')
    ax.set_title('1D bifurcation diagram')
    ax.set_xlim([0.7, 1.4])
    ax.set_ylim([0.0, 0.1])

    # 2D parameter continuation in k_pa and k_pp
    ax = fig3.add_subplot(grid3[2, 0])
    ax = a.plot_continuation('PAR(8)', 'PAR(6)', cont='c3:k_pa/k_pp', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#8299b0', line_style_unstable='solid', ignore=['LP'])
    ax.set_xlabel(r'$k_{pa}$')
    ax.set_ylabel(r'$k_{pp}$')
    ax.set_title('2D bifurcation diagram')
    ax.set_xlim([0.0, 3.0])
    ax.set_ylim([0.0, 1.5])

    # 2D parameter continuation in k_pa and k_ap
    ax = fig3.add_subplot(grid3[2, 1])
    ax = a.plot_continuation('PAR(8)', 'PAR(7)', cont='c3:k_pa/k_ap', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#8299b0', line_style_unstable='solid', ignore=['LP'])
    ax.set_xlabel(r'$k_{pa}$')
    ax.set_ylabel(r'$k_{ap}$')
    ax.set_title('2D bifurcation diagram')
    ax.set_xlim([0.0, 3.0])
    ax.set_ylim([0.0, 10.0])

##########################################################################
# c4: investigation of GPe behavior for c1 with strong GPe-p projections #
##########################################################################

if c4:

    starting_point = 'UZ3'
    starting_cont = s1_cont

    # set condition-specific non-free model parameters
    ##################################################

    # set GPe-p projection strength (k_p)
    c4_b1_sols, c4_b1_cont = a.run(starting_point=starting_point, c='qif', ICP=20, NPAR=n_params, name='c4:k_p',
                                   NDIM=n_dim, RL0=0.1, RL1=10.0, origin=starting_cont, NMX=6000, DSMAX=0.1,
                                   UZR={20: [1.5]}, STOP=['UZ1'])

    starting_point = 'UZ1'
    starting_cont = c4_b1_cont

    # set background input of GPe-p (eta_p)
    c4_b2_sols, c4_b2_cont = a.run(starting_point=starting_point, c='qif', ICP=2, NPAR=n_params,
                                   name=f'c4:eta_p', NDIM=n_dim, RL0=0.0, RL1=10.0, origin=starting_cont,
                                   NMX=6000, DSMAX=0.1, UZR={2: [3.2]})

    starting_point = 'UZ1'
    starting_cont = c4_b2_cont

    # main bifurcation analysis
    ###########################

    # perform 1D parameter continuation of GPe-a background input (eta_a)
    c4_b3_sols, c4_b3_cont = a.run(starting_point='UZ1', c='qif', ICP=3, NPAR=n_params,
                                   name=f'c4:eta_a', NDIM=n_dim, RL0=-2.0, RL1=6.0,
                                   origin=starting_cont, NMX=8000, DSMAX=0.01, bidirectional=True, UZR={3: [3.0]})

    # perform 1D parameter continuation of GPe-a to GPe-p projection (k_pa) for eta_a = 3.0
    c4_b4_sols, c4_b4_cont = a.run(starting_point='UZ1', origin=c4_b3_cont, ICP=8, NPAR=n_params, NDIM=n_dim, RL0=0.0,
                                   RL1=10.0, NMX=8000, DSMAX=0.01, bidirectional=True, name='c4:k_pa')

    # perform 2D parameter continuation for hopf and fold bifurcations found in k_pa
    c4_b4_2d1_sols, c4_b4_2d1_cont = a.run(starting_point='LP1', origin=c4_b4_cont, c='qif2', ICP=[8, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c4:k_pa/k_pp:lp1', bidirectional=True)
    c4_b4_2d2_sols, c4_b4_2d2_cont = a.run(starting_point='LP1', origin=c4_b4_cont, c='qif2', ICP=[8, 7], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c4:k_pa/k_ap:lp1', bidirectional=True)

    c4_b4_2d3_sols, c4_b4_2d3_cont = a.run(starting_point='HB1', origin=c4_b4_cont, c='qif2', ICP=[8, 6], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c4:k_pa/k_pp:hb1', bidirectional=True)
    c4_b4_2d4_sols, c4_b4_2d4_cont = a.run(starting_point='HB1', origin=c4_b4_cont, c='qif2', ICP=[8, 7], NDIM=n_dim,
                                           NPAR=n_params, RL0=0.0, RL1=10.0, NMX=8000, DSMAX=0.1,
                                           name='c4:k_pa/k_ap:hb1', bidirectional=True)

    # plotting
    ##########

    fig4 = plt.figure(tight_layout=True, figsize=(6.0, 6.0))
    grid4 = gs.GridSpec(3, 2)

    # 1D parameter continuation of GPe-a background input (eta_a)
    ax = fig4.add_subplot(grid4[0, :])
    ax = a.plot_continuation('PAR(3)', 'U(2)', cont=f'c4:eta_a', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', custom_bf_styles={'HB': {'color': 'k'}})
    ax.set_xlabel(r'$\eta_a$')
    ax.set_ylabel('GPe-p firing rate per ms')
    ax.set_title('1D bifurcation diagram')
    ax.set_xlim([2.0, 4.0])
    ax.set_ylim([0.0, 0.1])

    # 1D parameter continuation of GPe-a to GPe-p projection (k_pa)
    ax = fig4.add_subplot(grid4[1, :])
    ax = a.plot_continuation('PAR(8)', 'U(2)', cont=f'c4:k_pa', ax=ax, line_color_stable='#76448A',
                             line_color_unstable='#5D6D7E', custom_bf_styles={'HB': {'color': 'k'}})
    ax.set_xlabel(r'$k_{pa}$')
    ax.set_ylabel('GPe-p firing rate per ms')
    ax.set_title('1D bifurcation diagram')
    ax.set_xlim([0.7, 1.4])
    ax.set_ylim([0.0, 0.1])

    # 2D parameter continuation in k_pa and k_pp
    ax = fig4.add_subplot(grid4[2, 0])
    ax = a.plot_continuation('PAR(8)', 'PAR(6)', cont='c4:k_pa/k_pp:hb1', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#8299b0', line_style_unstable='solid', ignore=['LP'])
    ax = a.plot_continuation('PAR(8)', 'PAR(6)', cont='c4:k_pa/k_pp:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', line_style_unstable='solid', ignore=['LP'])
    ax.set_xlabel(r'$k_{pa}$')
    ax.set_ylabel(r'$k_{pp}$')
    ax.set_title('2D bifurcation diagram')
    ax.set_xlim([0.0, 3.0])
    ax.set_ylim([0.0, 1.5])

    # 2D parameter continuation in k_pa and k_ap
    ax = fig4.add_subplot(grid4[2, 1])
    ax = a.plot_continuation('PAR(8)', 'PAR(7)', cont='c4:k_pa/k_ap:hb1', ax=ax, line_color_stable='#8299b0',
                             line_color_unstable='#8299b0', line_style_unstable='solid', ignore=['LP'])
    ax = a.plot_continuation('PAR(8)', 'PAR(7)', cont='c4:k_pa/k_ap:lp1', ax=ax, line_color_stable='#3689c9',
                             line_color_unstable='#3689c9', line_style_unstable='solid', ignore=['LP'])
    ax.set_xlabel(r'$k_{pa}$')
    ax.set_ylabel(r'$k_{ap}$')
    ax.set_title('2D bifurcation diagram')
    ax.set_xlim([0.0, 3.0])
    ax.set_ylim([0.0, 2.0])

plt.show()
