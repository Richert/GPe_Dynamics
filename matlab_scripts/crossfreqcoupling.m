function [MI, PPC, fphaseCenters, fampCenters] = crossfreqcoupling(data,srate,phase_band,amp_band,n_bins,n_low,n_high)
%CROSSFREQCOUPLING calculates PAC profile of a 1d timeseries (data) with a
%   sampling rate (srate) for all combinations of phase frequencies and amplitude frequencies. phase_band and amp_band
%   contain 2 scalars each indicating the min and max of the respective frequency band. n_low and n_high indicate the
%   number of discrete frequency bands to filter the signal around within these limits.

% frequency binning
%%%%%%%%%%%%%%%%%%%

pha_locut = phase_band(1);
pha_hicut = phase_band(2);
num_phabins = round((pha_hicut-pha_locut)/n_low);
cutoff = 4*srate/pha_locut;
pnts = length(data(:, cutoff:end-cutoff));

amp_locut = amp_band(1);
amp_hicut = amp_band(2);
num_ampbins = round((amp_hicut-amp_locut)/n_high);

fphase = linspace(pha_locut,pha_hicut,num_phabins+1);
famp = linspace(amp_locut,amp_hicut,num_ampbins+1);

fphaseCenters = fphase(2:end)-diff(fphase)/2;
fampCenters = famp(2:end)-diff(famp)/2;

MI = zeros(num_phabins,num_ampbins);
PPC = zeros(size(MI));

data4phase= zeros(num_phabins,pnts);
data4amp = cell(num_phabins,1);
Hdata4phase = zeros(size(data4phase));
Hdata4amp = cell(num_phabins,1);
getamp = cell(num_phabins,1);

% filtering of the phase-containing frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for idx = 1:num_phabins
    lowband = [fphase(idx),fphase(idx+1)];
    low_filt_order = round(3*(srate/lowband(1)));
    low_filterweights = fir1(low_filt_order,[mean(lowband)-(lowband(2)-lowband(1)) mean(lowband)+(lowband(2)-lowband(1))]/(srate/2)); % -- NOTE: if eeglab is added to path, remove eeglab/plugins/biosig from path
    lowfilt = filtfilt(low_filterweights,1,data);
    data4phase(idx,:)= lowfilt(cutoff:end-cutoff);
    Hdata4phase(idx,:) = hilbert(data4phase(idx,:));
end
getphase = angle(Hdata4phase);

% filtering of the amplitude-containing frequencies
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for indpha = 1:num_phabins
    data4amp{indpha} = zeros(num_ampbins,pnts);
    Hdata4amp{indpha} = zeros(num_ampbins,pnts);
    start_amp = min(find(fampCenters > fphaseCenters(indpha)*3/2));
    for idx = start_amp:num_ampbins
        highband = [(fampCenters(idx)-fphaseCenters(indpha)/2),(fampCenters(idx)+fphaseCenters(indpha)/2)];
        high_filt_order = round(3*(srate/highband(1)));
        high_filterweights = fir1(high_filt_order,[mean(highband)-(highband(2)-highband(1)) mean(highband)+(highband(2)-highband(1))]/(srate/2)); % -- NOTE: if eeglab is added to path, remove eeglab/plugins/biosig from path
        highfilt = filtfilt(high_filterweights,1,data);
        data4amp{indpha}(idx,:) = highfilt(cutoff:end-cutoff);
        Hdata4amp{indpha}(idx,:) = hilbert(data4amp{indpha}(idx,:));
        getamp{indpha} = abs(Hdata4amp{indpha});
    end
end

% modulation index (PAC) calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MI(:,:) = get_modulation_index(getphase,getamp,n_bins);

% phase-phase coupling (PPC) calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:num_phabins
    lf_signal = data4phase(i,:);
    hf_signals = data4amp{i};
    low_band = [fphase(i),fphase(i+1)];
    for j=1:num_ampbins
        [~, ~, ppc_osc, ppc_env] = PhaseLockAmp(hf_signals(j,:)',lf_signal',low_band,srate,2);
        PPC(i,j) = ppc_osc/ppc_env;
    end
end

end