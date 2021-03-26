import struct
import numpy as np
from scipy.io import wavfile
import librosa


def read_audio(path):
    data, sample_rate = librosa.load(path, mono=False)
    data=data.reshape(data.shape[1],data.shape[0])
    return data, sample_rate

def normalize_audio(audio):
    norm = np.linalg.norm(audio)
    audio = audio/norm
    return audio

def audio_padding(audio,time,window,freq):
    size=int(time*freq)
    window_size=int(window*freq)
    if size%window_size!=0:
        size=0
        while size<time*freq:
            size=size+window_size
    if len(audio)<size:
        result = np.zeros((size,4))
        result[:audio.shape[0],:audio.shape[1]] = audio
        return result
    else:
        return audio

def split_audio(window,freq,audio):
    window_size=int(window*freq)
    n_groups=int(len(audio)/window_size)
    return np.split(audio,n_groups), n_groups

def fft_frame(audio):
    spectrum=np.fft.fft(audio)
    M=len(spectrum)
    M2=int(M/2)
    return spectrum[M2:M]

def fft(frames):
    fft_list=[]
    for frame in frames:
        fft_list.append(fft_frame(frame))
    return np.asarray(fft_list)

def fft_set(set_,out='s'):
    set_=np.asarray(set_)
    fft_list=[]
    for frames in set_:
        fft_list.append(fft(frames))
    fft_list=np.asarray(fft_list)
    magnitude=np.abs(fft_list)
    phase=np.angle(fft_list)
    if out=='s':
        return np.concatenate((np.asarray(magnitude), np.asarray(phase)),axis=3)
    elif out=='m':
        return np.asarray(magnitude)
    elif out=='p':
        return np.asarray(phase)
    elif out=='mp':
        return np.asarray(magnitude), np.asarray(phase)
    else:
        print('ERROR: out can be \'m\' to get the magnitude,  \'p\' to get the phase,  \'mp\' to get magnitude and phase,  \'s\' to get the spectrum concatenating magnitude and phase on the last axis')
        exit()