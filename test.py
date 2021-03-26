from L3DAS import data
from L3DAS import audio_processing as dsp
import numpy as np
dataset=data.Dataset("Task2",num_samples=1,frame_len=20, mic='AB')
audio,classes,location=dataset.get_dataset()
print(audio.shape)
print(classes.shape)
print(location.shape)
magnitude, phase=dsp.fft_set(audio,out='mp')
spec = dsp.fft_set(audio,out='s')
print(spec.shape)
print(magnitude.shape)
print(phase.shape)
print(location[0])
