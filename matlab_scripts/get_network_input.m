function [fr, spike_t, spike_n, mask, wait] = get_network_input(spike_t, wait, dt)
%GET_NETWORK_INPUT Calculate current firing activity of each neuron.
%   Based on the approximated spike timings and waiting periods, calculates an instantaneous firing rate of each neuron.
wait = max(wait-1,0);
spike_t = max(spike_t-1,0);
spike_n = 1.*((spike_t>0)&(spike_t<=1));
mask = 1.*(round(wait)==0);
fr = spike_n ./ dt;
end