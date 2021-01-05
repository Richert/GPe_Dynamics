function MI = get_modulation_index(raw_phase,raw_amp,n_bins)
%% GET_MODULATION_INDEX function for calculating the PAC modulation index from the phase of a low-frequency and the
%   amplitude of a high-frequency signal.

% extract information
n_phabins = size(raw_phase,1);
n_ampbins = size(raw_amp{1},1);

% calculate PAC (modulation index) for each pair of low- and high frequency
MI = zeros(n_phabins,n_ampbins);

for i = 1:n_phabins
    for j = 1:n_ampbins

        % get low freq phase and high freq amp
        low_phase = raw_phase(i,:);
        high_amp = raw_amp{i}(j,:);
        MI(i,j) = get_pac(low_phase, high_amp, n_bins);
    end
end

end
