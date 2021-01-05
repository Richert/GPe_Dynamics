function [HFosciLockedLFpeaks,HFampLockedLFpeaks,PLAosci,PLAamp]= PhaseLockAmp(HFsignal,LFsignal,lowFreq,fs,NT)
%PHASELOCKAMP function for calculating phase-locked amplitudes of a high-frequency signal. Serves to calculate the
% phase-phase coupling with a lower frequency signal.

%Default values of the outputs --------------------------------------------
HFosciLockedLFpeaks = [];
HFampLockedLFpeaks = [];
PLAosci = [];
PLAamp = [];

%Parameters ---------------------------------------------------------------

sampleT = round( NT * fs/lowFreq(1));
halfSampleT = round(sampleT/2);
window = round(fs/lowFreq(1)/2);

%Z-score normalization ----------------------------------------------------
%in order to have zero mean and unit variance (unit amplitude) in HF and LF signals.
HFsignal_osci = zscore(HFsignal);
HFsignal_amp = abs(hilbert(HFsignal_osci));
LFsignal = zscore(LFsignal);
Nsamples = size(LFsignal,1);

% get maximum amplitude of high-frequency signal (for normalization)
max_amp = max(abs(HFsignal_osci));
LFphase = angle(hilbert(LFsignal)); %[rad] range: [-pi,pi]

%Find zero-crossings in LF-phase time series.
s1 = LFphase(1:Nsamples-1);
s2 = LFphase(2:Nsamples);

pulseZC_peak = (s1.*s2)<0 & abs(s1-s2)<pi;

%Re-arrange the pulses in order to emulate a causal zero-crossing detection.
dim = 1;
pulseZC_peak = cat(dim, 0, pulseZC_peak);

%Compute the indices for the peaks of the LF signal (zeroes of the LF-phase signal).
indLF_peak= find(pulseZC_peak > 0);

%Compute the number of LF peaks.
Npeak = length(indLF_peak);

%Compute the valid indices for the HF and LF peaks.
JJstart = NaN;
JJend = NaN;

for jj=1:+2:Npeak-1
    if indLF_peak(jj)-halfSampleT > 0
        JJstart = jj;
        break,
    end
end
for jj=Npeak-1:-2:1
    if indLF_peak(jj)+halfSampleT < Nsamples
        JJend = jj;
        break,
    end
end

HFosciLockedLFpeaks = zeros(2*halfSampleT+1,JJend-JJstart+1);
HFampLockedLFpeaks = zeros(2*halfSampleT+1,JJend-JJstart+1);
for jj=JJstart:JJend %Loop over the peak indices.
    HFosciLockedLFpeaks(:,jj)= HFsignal_osci(indLF_peak(jj)-halfSampleT:indLF_peak(jj)+halfSampleT);
    HFampLockedLFpeaks(:,jj)= HFsignal_amp(indLF_peak(jj)-halfSampleT:indLF_peak(jj)+halfSampleT);
end

mean_HFosciLockedLFpeaks = mean(HFosciLockedLFpeaks,2);
mean_HFampLockedLFpeaks = mean(HFampLockedLFpeaks,2);

%% compute the PLA_osci and PLA_env
PLAosci = max(abs(mean_HFosciLockedLFpeaks((halfSampleT+1-window):(halfSampleT+1+window))))/max_amp;
PLAamp = max(mean_HFampLockedLFpeaks((halfSampleT+1-window):(halfSampleT+1+window)))/max_amp;

end
