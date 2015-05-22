function [mPSD, PAC] = sliding_window_mPSD_PAC(data, srate, mf, lof, hif, sw_size)

% function [mPSD, PAC] = sliding_window_mPSD_PAC(raw_data, srate, mf, lof, hif, sw_size)
% Usage: [mPSD, PAC] = sliding_window_mPSD_PAC(raw_data, 1024, [40 250], [4 8], [80 150], 512)

% SLIDING_WINDOW_mPSD_PAC takes in electrocoriticography or local field potential
% data and calculates the slope of the PSD across the frequency range mf. It 
% also calcualtes phase / amplitude coupling (using the phase locking value method)
% between the phase-giving frequency range lof and amplitude-giving range
% hif. These are calculated across a sliding window of size sw_size.
%
% This function requires 6 parameters:
% data: A 1, 2, or 3D matrix where the final dimension is samples, the second-to-last
% dimension is channels, and the third-to-last is subjcts/trials/bins/etc.
% srate: sampling rate of the input data
% mf: the upper and lower limits of the slope estimation frequency range
% lof: the upper and lower limits of the phase passband
% hif: the upper and lower limits of the amplitude passband
% sw_size: the size of the sliding window

% Bradley Voytek & Torben Noto
% CC BY 2015
% University of California, San Diego
% Department of Cognitive Science and Neurosciences

%checking and reshaping data
s = size(data);

squeeze_final = 0;
if length(s) > 3
    error('data must be 3 dimentions or less with the final dimention being samples')
end

if length(s) == 2
    data = reshape(data,[1,s(1),s(2)]);
    squeeze_final = 1;
end

if length(s) == 1
    data = reshape(data,[1,1,s(1)]);
    squeeze_final = 1;
end

sw_size = floor(sw_size);

mPSD = zeros(size(data));
PAC = zeros(size(data));

hamming_window = hamming(sw_size); %sliding hamming window

%initialize PSD slope frequency bins
f = (0:(sw_size / 2) - 1) * (srate / sw_size);
[~, freqs(1)] = min(abs(f-mf(1)));
[~, freqs(2)] = min(abs(f-mf(2)));
freqs = freqs(1):freqs(2);

%loop through data
for subj= 1:size(data,1)
    for chan= 1:size(data,2)
        
        %get hif analytic amplitude
        hi_band = eegfilt(data(subj,chan, :), srate, hif(1), []);
        hi_band = eegfilt(hi_band, srate, [], hif(2));
        hi_band = abs(hilbert(hi_band));

        %get lo_band phase of hi_band AA
        lo_hi_band = eegfilt(hi_band(1, :), srate, lof(1), []);
        lo_hi_band = eegfilt(lo_hi_band, srate, [], lof(2));
        lo_hi_band = angle(hilbert(lo_hi_band));

        % get lo_band phase
        lo_band = eegfilt(data(subj,chan, :), srate, lof(1), []);
        lo_band = eegfilt(lo_band, srate, [], lof(2));
        lo_band = angle(hilbert(lo_band));

        %pad data to remove edge artifacts
        newdat = [squeeze(fliplr(data(subj,chan,1:sw_size)))' squeeze(data(subj,chan, :))' squeeze(fliplr(data(subj, chan, end-sw_size+1:end)))'];
        new_lo_hi = [squeeze(fliplr(lo_hi_band(1,1:sw_size))) squeeze(lo_hi_band(1, :)) squeeze(fliplr(lo_hi_band(1, end-sw_size+1:end)))];
        newlo_band = [squeeze(fliplr(lo_band(1,1:sw_size))) squeeze(lo_band(1, :)) squeeze(fliplr(lo_band(1, end-sw_size+1:end)))];
        
        %calculate envelope to signal measure
        new_lo_hi = new_lo_hi - newlo_band;
        
        %initialize output variables
        mPSDdat = zeros(1,length(newdat));
        pacdat = zeros(1,length(newdat));
        
        fprintf('Calculating mPSD between %d and %d Hz with sliding window of %d samples\n',(freqs(1)-1)*(f(2)-f(1)),(freqs(end)-1)*(f(2)-f(1)),sw_size)
        
        for ti = 1:(length(newdat) - sw_size)
            %calculate slope of high gamma power at time ti
            temp = fft(hamming_window' .* newdat(ti:(sw_size+ti-1)), sw_size);
            temp = log10(((abs(temp(1:(sw_size / 2))) / sw_size) .^ 2) .* 2);
            temp = ([f(freqs)' ones(size(f(freqs)'))]) \ temp(freqs)'; %solving system of equations to get slope of PSD
            mPSDdat(ti) = temp(1);
            
            %calculate PAC at time ti
            temp = new_lo_hi(ti:(sw_size+ti-1));
            pacdat(ti) = abs(sum(exp(1i * temp), 'double')) / length(temp); %circular mean of PAC in sliding window
        end

        %remove padding
        mPSD(subj,chan,:) = mPSDdat((sw_size+1):(end-sw_size));
        PAC(subj,chan,:) = pacdat((sw_size+1):(end-sw_size));
    end
end

%clean up output
if squeeze_final == 1
    mPSD = squeeze(mPSD);
    PAC = squeeze(PAC);
end


