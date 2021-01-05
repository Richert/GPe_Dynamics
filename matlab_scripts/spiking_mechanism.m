function [v, spike_t, wait] = spiking_mechanism(v, v_th, dt, wait, spike_t, tau, I)
%SPIKING_MECHANISM Generates spikes based on membrane potential of the neurons.
%    Cuts of membrane potential dynamics at threshold v_th and approximates rest of the waiting time of the neuron
%    according to QIF equations.
spike = (v>v_th);
wait_tmp = (2*tau./v(spike))./dt - (6*I(spike)./v(spike).^3)./dt;
spike_t_tmp = (tau./v(spike))./dt;
wait(spike) = max(wait_tmp, 1);
spike_t(spike) = max(spike_t_tmp, 1);
v(spike) = -v(spike);
end