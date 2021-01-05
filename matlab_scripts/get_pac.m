function pac = get_pac(low_phase,high_amp,n_bins)
%GET_PAC Calculates the modulation index measuring phase-amplitude coupling.
%   Uses the instantaneous phase of a low-frequency signal and the instantaneous amplitude of a high frequency signal.
n_timepoints = size(low_phase, 2);
lowphase_bin = ceil( tiedrank( low_phase ) / (n_timepoints / n_bins) ); % -- bin the low phase angles into nbins -- NOTE: tiedrank also exists in eeglab toolbox; when added to path, may cause conflict
highamp_bin = zeros(1,n_bins);
for k=1:n_bins
    highamp_bin(k) = squeeze(mean(high_amp(lowphase_bin==k))); % -- compute mean high power in each bin
end
highamp_bin = highamp_bin ./ sum(highamp_bin); % -- normalize
pac = 1 + sum(highamp_bin.*log(highamp_bin)) / log(n_bins);
end
