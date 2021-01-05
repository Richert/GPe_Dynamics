%% Description
% This script simulates the behavior of a spiking neural network of recurrently coupled prototypical and arkypallidal
% cells (both QIF populations) with all-to-all coupling.
%
% You need a Matlab installation to run this script. Simulation and model parameters can be adjusted in the first part
% of the script.

close all; clear all;

% parameter settings
%%%%%%%%%%%%%%%%%%%%

% general parameters
dt = 1e-3;
dt2 = 1.0;
T_init = 100.0;
T_sim = T_init + 200.0;
T = round(T_sim/dt);
T_rec = round((T_sim-T_init)/dt2);
T_start = round(T_init/dt);
t_rec = round(dt2/dt);
v_mac_init = -2;
n_plot = 100;

% scaling and input parameters
r_s = 0.002;
r_e = 0.02;
k = 100.0;
eta = 100.0;
alpha = 40.0;
t_off = 82.0;
t_on = 5.0;

% GPe-p parameters
N_p = 40000;
tau_p = 25;
v_th_p = 1e4;
Delta_p = 90.0;
eta_p = 3.2*eta;
eta_p = eta_p+Delta_p.*tan((pi/2).*(2.*[1:N_p]-N_p-1)./(N_p+1));

% GPe-a parameters
N_a = 20000;
tau_a = 20;
v_th_a = 1e4;
Delta_a = 120.0;
eta_a = 3.0*eta;
eta_a = eta_a+Delta_a.*tan((pi/2).*(2.*[1:N_a]-N_a-1)./(N_a+1));

% coupling parameters
k_gp = 3.0;
k_i = 1.8;
k_p = 1.5;
k_pp = k*k_gp*k_p/k_i;
k_ap = k*k_gp*k_p*k_i;
k_pa = k*k_gp*k_i/k_p;
k_aa = k*k_gp/(k_i*k_p);

% initializations
%%%%%%%%%%%%%%%%%

% state vector
m_p = N_p + 6;
m_a = m_p + N_a;
N = m_a + 8;
u = zeros(1,N);
u(1:N_p) = v_mac_init;
u(m_p+1:m_a) = v_mac_init;
u(N-1) = 1.0;

% GPe-p
wait_p = zeros(1,N_p);
spike_n_p = zeros(1,N_p);
spike_t_p = wait_p;

rp_rec = zeros(1,t_rec);
vp_rec = zeros(size(rp_rec));
rp_rec_av = zeros(1,T_rec);
vp_rec_av = zeros(size(rp_rec_av));

for i = 1:N_p; raster_p{i} = []; end

% GPe-a
wait_a = zeros(1,N_a);
spike_n_a = zeros(1,N_a);
spike_t_a = wait_a;

ra_rec = zeros(1,t_rec);
va_rec = zeros(size(ra_rec));
ra_rec_av = zeros(1,T_rec);
va_rec_av = zeros(size(ra_rec_av));

for i = 1:N_a; raster_a{i} = []; end

% inputs
It = dt.*[1:T];
I_p = 0.*((It>0)&(It<T_sim)) + k*tau_p*(10*r_e - 20*r_s);
I_a = 0.*((It>0)&(It<T_sim)) + k*tau_a*(10*r_e - 20*r_s);

% simulation
%%%%%%%%%%%%

for t = 1:T

    % calculate population inputs
    [rp, spike_t_p, spike_n_p, mask_p, wait_p] = get_network_input(spike_t_p, wait_p, dt);
    [ra, spike_t_a, spike_n_a, mask_a, wait_a] = get_network_input(spike_t_a, wait_a, dt);
    s_pp = mean(rp);
    s_ap = s_pp;
    s_pa = mean(ra);
    s_aa = s_pa;

    % calculate rhs update
    [du, Ip, Ia] = gpe_2pop_forced_ata_rhs(u,s_pp,s_ap,s_pa,s_aa,I_p(1,t),I_a(1,t),...
        eta_p,eta_a,N_p,N_a,k_pp,k_pa,k_ap,k_aa,alpha,t_on,t_off,tau_p,tau_a);
    u(1:N_p) = u(1:N_p) + dt .* mask_p .* du(1:N_p);
    u(N_p+1:m_p) = u(N_p+1:m_p) + dt .* du(N_p+1:m_p);
    u(m_p+1:m_a) = u(m_p+1:m_a) + dt .* mask_a .* du(m_p+1:m_a);
    u(m_a+1:end) = u(m_a+1:end) + dt .* du(m_a+1:end);

    % spiking mechanism
    [vp, spike_t_p, wait_p] = spiking_mechanism(u(1:N_p), v_th_p, dt, wait_p, ...
        spike_t_p, tau_p, eta_p - tau_p*Ip + I_p(1, t));
    [va, spike_t_a, wait_a] = spiking_mechanism(u(m_p+1:m_a), v_th_a, dt, ...
        wait_a, spike_t_a, tau_a, eta_a - tau_a*Ia + I_a(1, t));

    % state variable updates
    u(1:N_p) = vp;
    u(m_p+1:m_a) = va;

    % variable recordings
    t_buffer = mod(t,t_rec);
    rp_rec(t_buffer+1) = s_pp;
    vp_rec(t_buffer+1) = mean(vp(wait_p==0));
    ra_rec(t_buffer+1) = s_aa;
    va_rec(t_buffer+1) = mean(va(wait_a==0));

    % coarse grained observations
    if t_buffer==0 && t > T_start
        t_tmp = (t-T_start).*(dt/dt2);
        vp_rec_av(t_tmp) = mean(vp_rec);
        rp_rec_av(t_tmp) = mean(rp_rec);
        va_rec_av(t_tmp) = mean(va_rec);
        ra_rec_av(t_tmp) = mean(ra_rec);
        if mod(t,1/dt)==0
            sprintf('Time: %d of %d done.',t.*dt,T.*dt)
        end
    end

    % spike recordings
    if sum(spike_n_p)>0
        rid = find(spike_n_p==1);
        for i = 1:numel(rid)
            raster_p{rid(i)} = cat(1,raster_p{rid(i)},t);
        end
    end
    if sum(spike_n_a)>0
        rid = find(spike_n_a==1);
        for i = 1:numel(rid)
            raster_a{rid(i)} = cat(1,raster_a{rid(i)},t);
        end
    end
end

% plotting
%%%%%%%%%%

% membrane potentials
figure();
hold on
plot(vp_rec_av);  plot(va_rec_av);
ylabel('v')
xlabel('time')
legend('GPe-p', 'GPe-a')
hold off

% firing rates
figure();
hold on
plot(rp_rec_av*1e3);  plot(ra_rec_av*1e3);
ylabel('r')
xlabel('time')
legend('GPe-p', 'GPe-a')
hold off

% GPe-p spikes
figure();
hold on;
[~,randidx] = sort(randn(N_p,1));
for i = 1:n_plot
 id = randidx(i);
 if ~isempty(raster_p{id})
     plot(raster_p{id},i.*ones(numel(raster_p{id},1)),'k.', 'MarkerSize', 4)
 end
end
ylim([0, n_plot+1]);
xlabel('time')
ylabel('neuron #')
title('GPe-p')
hold off

% GPe-a spikes
figure();
hold on;
[~,randidx] = sort(randn(N_a,1));
for i = 1:n_plot
 id = randidx(i);
 if ~isempty(raster_a{id})
     plot(raster_a{id},i.*ones(numel(raster_a{id},1)),'k.', 'MarkerSize', 4)
 end
end
ylim([0, n_plot+1]);
xlabel('time')
ylabel('neuron #')
title('GPe-a')
hold off

% END OF FILE
