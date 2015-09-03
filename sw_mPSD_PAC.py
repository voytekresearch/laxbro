"""

Code to calculate the spectral slope and phase amplitude coupling
in one channel of electrophysiological data


Authors: Torben Noto & Bradley Voytek
"""

import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, hilbert
from itertools import chain


def sw_mpsd_pac(data, srate, sw_size=250, lo_band=[6, 10], hi_band=[80, 150], m_range=[80, 150], pac_method='PLV'):

    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs #nyquist frequency
        low = float(lowcut) / nyq
        high = float(highcut) / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    # This function applies Butterworth filter to data
    def butter_bandpass_filter(mydata, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, mydata)
        return y

    hi_band = abs(hilbert(butter_bandpass_filter(data, hi_band[0], hi_band[1], round(srate))))
    lo_hi_band = np.angle(hilbert(butter_bandpass_filter(hi_band, lo_band[0], lo_band[1], round(srate))))
    lo_band = np.angle(hilbert(butter_bandpass_filter(data, lo_band[0], lo_band[1], round(srate))))

    #  padding data
    newdat = list(chain(data[sw_size-1::-1], data, data[len(data):len(data)-sw_size-1:-1]))
    new_lo_hi = list(chain(lo_hi_band[sw_size-1::-1], lo_hi_band, lo_hi_band[len(lo_hi_band):len(lo_hi_band)-sw_size-1:-1]))
    #newlo = list(chain(lo_band[sw_size-1::-1], lo_band, lo_band[len(lo_band):len(lo_band)-sw_size-1:-1]))

    #  initializing output
    mPSD = np.zeros((1, len(newdat)))
    PAC = np.zeros((1, len(newdat)))

    #  frequency bins
    xf = np.linspace(0.0, sw_size, sw_size*(srate/1000))

    #  finding closest fx bin to lower fx boundary
    bin0ind = min(enumerate(xf), key=lambda i: abs(i[1]-m_range[0]))[0]
    bin1ind = min(enumerate(xf), key=lambda i: abs(i[1]-m_range[1]))[0]
    print bin0ind
    print bin1ind

    for s in xrange(len(newdat)-sw_size):
        window_fourier = np.log10(abs(fft(newdat[s:s+sw_size] * np.hanning(sw_size))))
        mpsd_fourier = window_fourier[bin0ind:bin1ind]
        X = np.vstack([np.array(xf[bin0ind:bin1ind]), np.ones((1, len(xf[bin0ind:bin1ind])))]).T
        y = np.vstack(np.array(mpsd_fourier))
        mPSD[0, s] = np.linalg.lstsq(X, y)[0][0]

        # PLV
        PAC[0, s] = abs(sum(np.exp(np.dot(1j, new_lo_hi[s:s+sw_size])))) / len(new_lo_hi[s:s+sw_size])

    PAC = np.squeeze(PAC)
    PAC = PAC[sw_size:]
    PAC = PAC[:sw_size]

    mPSD = np.squeeze(mPSD)
    mPSD = mPSD[sw_size:]
    mPSD = mPSD[:sw_size]

    return mPSD, PAC
