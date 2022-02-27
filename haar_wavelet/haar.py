import wave

import numpy as np
from scipy import signal

import config

def compute_haar(data, n, s, w):
    h0 = [1/np.sqrt(2), 1/np.sqrt(2)]
    h1 = [1/np.sqrt(2), -1/np.sqrt(2)]
    s[n+1] = my_downsample_2x(signal.lfilter(h0, [1], data))
    w[n+1] = my_downsample_2x(signal.lfilter(h1, [1], data))

def compute_haar_wrapper(raw_data, levels=10):
    l = len(raw_data)
    max_layers = np.log2(l)
    if levels > max_layers:
        levels = int(max_layers)
    print("Length: {}. Max Layers: {}".format(l, max_layers))
    s = {}
    w = {}
    s[0] = raw_data
    for i in range(levels):
        compute_haar(s[i], i, s, w)
    return w, s

def reconstruct(j, s, w):
    g0 = [1/np.sqrt(2), 1/np.sqrt(2)]
    g1 = [1/np.sqrt(2), -1/np.sqrt(2)]

    w2 = my_average_us2x(w[j])
    s2 = my_average_us2x(s[j])
    # s_hat_2x = signal.lfilter(g0, [1], s2) + signal.lfilter(g1, [1], w2)
    s_hat_2x = (w2 + s2) / np.sqrt(2)
    s_diff = s[j-1] - s_hat_2x
    return s_hat_2x, s_diff

def save_sample(fname, data, j):
    with wave.open(fname, "w") as wf:
        wf.setframerate(config.RATE // (2**j))
        wf.setnchannels(config.CHANNELS)
        wf.setsampwidth(2)
        wf.writeframes(data.astype("h").tobytes())

def my_upsample_2x(data):
    out = np.zeros((data.shape[0] * 2,), dtype=data.dtype)
    out[0::2] = data[:]
    out[1::2] = data[:]
    return out

def my_average_us2x(data):
    out = np.zeros((data.shape[0] * 2,), dtype=data.dtype)
    out[0::2] = data[:]
    out[1:-1:2] = (data[:-1] + data[1:]) / 2
    out[-1] = data[-1]
    return out

def my_downsample_2x(data):
    if data.shape[0] % 2 == 1:
        data = data[:-1]
    out = np.zeros((data.shape[0] // 2,), dtype=data.dtype)
    out[:] = data[0::2]
    return out

def my_average_ds2x(data):
    if data.shape[0] % 2 == 1:
        data = data[:-1]
    out = np.zeros((data.shape[0] // 2,), dtype=data.dtype)
    out[:] = (data[0::2] + data[1::2]) / 2
    return out

if __name__ == "__main__":
    wf = wave.open("record.wav", "rb")
    dn = np.frombuffer(wf.readframes(wf.getnframes()), dtype="h")
    compute_haar(dn)