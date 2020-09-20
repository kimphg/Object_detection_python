# -*- coding: utf-8 -*-
import os
import librosa   #thư viện xủ lý tín hiệu âm thanh
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #để đọc file wav 
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
train_audio_path = 'D:/DIGIREG/DATA/'
labels=os.listdir(train_audio_path)
from numpy import save
from numpy import asarray

WORD = 6000
WORD_HALF = int(WORD/2)
CHUNK = 10000 
all_wave = []
all_label = []
all_feature = []
for label in labels:
    print("Label:",label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 16000)
        
        leng = len(samples)
        chunk_count = int(leng/CHUNK)
        word_count = 0
        for i in range(0,chunk_count):
            data = samples[i*CHUNK:(i+2)*CHUNK]
            
            energy = abs(data)
            mean = np.mean(energy)
            value = 0
            max_val = 0
            max_idn = 0
            for idn in range(0,len(energy)):
                value += (energy[idn]-value)/1000
                if (idn>WORD_HALF) & (idn<(len(data)-WORD_HALF )):
                    if max_val<value:
                        max_val = value
                        max_idn = idn
            if max_val>(mean*2.5):
                word_count = word_count+1
                data = data[max_idn-WORD_HALF:max_idn+WORD_HALF]
#                 feature_vector = librosa.feature.melspectrogram(y=data, sr=16000)[:40]
#                 feature_vector = librosa.feature.chroma_stft(y=data, sr=16000)
                all_wave.append(data)
#                 all_feature.append(feature_vector)
                all_label.append(label)
        
        print("chunks:",chunk_count," samples:",word_count)
save('all_wave', all_wave)
#save('all_feature', all_feature)
save('all_label', all_label)
