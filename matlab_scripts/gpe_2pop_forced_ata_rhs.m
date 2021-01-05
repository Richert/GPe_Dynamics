function [du, I_p, I_a] = gpe_2pop_forced_ata_rhs(u,s_pp,s_ap,s_pa,s_aa,I_ext_p,I_ext_a,...
    eta_p,eta_a,N_p,N_a,k_pp,k_pa,k_ap,k_aa,alpha,t_on,t_off,tau_p,tau_a)
%GPE_2POP_FORCED_ATA_RHS Dynamic equations of a two-population gpe qif network with periodic forcing and all-to-all coupling.
%       Right-hand side of the set of differential equations for the QIF network of arkypallidal and prototypical neurons
%       driven by a sigmoidal transformation of a stuart-landau oscillator.

%% parameters and state variables

% declare constants
tau_gabaa_r = 0.5;
tau_gabaa_d = 5.0;
k_d = 1.33;
m_p = N_p + 6;
m_a = m_p + N_a;
slope = 100.0;
omega = 2.0*pi/(t_on+t_off);

% extract state variables at gpe-p
v_p = u(1:N_p);
I_p = u(N_p+1);
y_p = u(N_p+2);
r_pp_1 = u(N_p+3);
r_pp_2 = u(N_p+4);
r_pa_1 = u(N_p+5);
r_pa_2 = u(m_p);

% extract state variables at gpe-a
v_a = u(m_p+1:m_a);
I_a = u(m_a+1);
y_a = u(m_a+2);
r_ap_1 = u(m_a+3);
r_ap_2 = u(m_a+4);
r_aa_1 = u(m_a+5);
r_aa_2 = u(m_a+6);

% extract state variables of periodic input
x1 = u(m_a+7);
x2 = u(m_a+8);

%% calculate right-hand side update of equation system

% prepare population input
s1 = alpha/(1.0 + exp(-slope*(x1-cos(omega*t_on/2.0))));
s2 = alpha/(1.0 + exp(-slope*(-x1-cos(omega*t_on/2.0))));

% 1. population updates
% ---------------------

% GPe-p
d_vp = (v_p.^2 + eta_p + I_ext_p - I_p*tau_p) ./ tau_p;

% GPe-a
d_va = (v_a.^2 + eta_a + I_ext_a - I_a*tau_a) ./ tau_a;

% 2. synapse dynamics
% -------------------

% at GPe-p
d_Ip = y_p;
d_yp = (k_pp*r_pp_2 + k_pa*r_pa_2 - y_p*(tau_gabaa_r+tau_gabaa_d) - I_p...
    )/(tau_gabaa_r*tau_gabaa_d);

% at GPe-a
d_Ia = y_a;
d_ya = (k_ap*r_ap_2 + k_aa*r_aa_2 + s1 - s2 - y_a*(tau_gabaa_r+tau_gabaa_d) - I_a...
    )/(tau_gabaa_r*tau_gabaa_d);

% Gpe-p to GPe-p
d_pp_1 = k_d * (s_pp - r_pp_1);
d_pp_2 = k_d * (r_pp_1 - r_pp_2);

% Gpe-p to GPe-a
d_ap_1 = k_d * (s_ap - r_ap_1);
d_ap_2 = k_d * (r_ap_1 - r_ap_2);

% Gpe-a to GPe-p
d_pa_1 = k_d * (s_pa - r_pa_1);
d_pa_2 = k_d * (r_pa_1 - r_pa_2);

% Gpe-a to GPe-a
d_aa_1 = k_d * (s_aa - r_aa_1);
d_aa_2 = k_d * (r_aa_1 - r_aa_2);

% input dynamics
% --------------

d_x1 = -omega*x2 + x1*(1-x1^2-x2^2);
d_x2 = omega*x1 + x2*(1-x1^2-x2^2);

%% return rhs update vector

du = [d_vp,d_Ip,d_yp,d_pp_1,d_pp_2,d_pa_1,d_pa_2,d_va,d_Ia,d_ya,d_ap_1,...
    d_ap_2,d_aa_1,d_aa_2,d_x1,d_x2];

end
