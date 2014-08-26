import numpy as np
import aubio

importance_rate = 0.95
window_size = 1024 # ~2MS
hop_size = window_size / 4
sample_rate = 44100
n_filters = 40
n_coeffs = 13
hanning = np.hanning(window_size)
freq_bins = np.fft.fftfreq(window_size, d=1.0/sample_rate)[0:window_size/2]

def use_centroid():
    global distance_metric
    distance_metric = distance_centroid

def use_mfccs():
    global distance_metric
    distance_metric = distance_mfccs

def use_rms():
    global distance_metric
    distance_metric = distance_rms

def use_rms_centroid():
    global distance_metric
    distance_metric = distance_rms_centroid

# Opposite of fitness = distance
# test, original are numpy arrays of values
def distance(test, original):
    # Make copies
    test = np.array(test)
    original = np.array(original)

    num_rows_t = np.ceil(len(test) * 1.0 / window_size)
    num_rows_o = np.ceil(len(original) * 1.0 / window_size)
    num_rows = int(max(num_rows_t, num_rows_o))
    
    # Reshape pads with zeros
    test.resize((num_rows, window_size))
    original.resize((num_rows, window_size))
    
    return distance_metric(test, original)  

def distance_rms_centroid(test, original):
    d_rms = distance_rms(test, original)
    d_centroid = distance_centroid(test, original)
    # RMS is on order of 10^-2, centroid is on order 10^6.
    # Multiply RMS by 10^8.
    return (d_rms * pow(10, 8)) + d_centroid

def distance_rms(test, original):
    sum = 0.0
    num_rows = len(test)
    for i in range(num_rows):
        test_rms = rms(test[i])
        original_rms = rms(original[i])
        sum += pow(importance_rate, i) * pow(test_rms - original_rms, 2)
    return sum

def rms(buf):
    return np.sqrt(np.dot(buf, buf) * 1.0 / len(buf))

def distance_centroid(test, original):
    sum = 0.0
    num_rows = len(test)
    for i in range(num_rows):
        test_centroid = spectral_centroid(test[i])
        original_centroid = spectral_centroid(original[i])
        sum += pow(importance_rate, i) * pow(test_centroid - original_centroid, 2)
    return sum

# Centroid is weighted mean of frequencies present with magnitudes as weights
def spectral_centroid(data):
    f = np.fft.fft(data * hanning)
    f = f[0:window_size/2]
    f = [abs(x.real) for x in f]
    if np.sum(f) <= 0.00000001:
        #print np.sum(f), "is zero"
        return 0
    return np.dot(freq_bins, f) / np.sum(f)
    
def distance_mfccs(test, orig):
    test_mfccs = mfccs(test)
    orig_mfccs = mfccs(orig)
    sum = 0.0
    for i in range(len(test_mfccs)):
        difference = test_mfccs[i] - orig_mfccs[i]
        sum += pow(importance_rate, i) * np.dot(difference, difference)
    return sum

def mfccs(windows):
    p = aubio.pvoc(window_size, hop_size)
    m = aubio.mfcc(window_size, n_filters, n_coeffs, sample_rate)
    windows = np.float32(windows)
    
    mfccs = np.zeros([13,])
    for i in range(len(windows)):
        samples = windows[i]
        spec = p(samples)
        mfcc_out = m(spec)
        mfccs = np.vstack((mfccs, mfcc_out))
    del p
    del m
    return mfccs

distance_metric = distance_mfccs
