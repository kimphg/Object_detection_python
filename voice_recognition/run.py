﻿print("* init")
import os
import IPython.display as ipd
# from scipy.io import wavfile #để đọc file wav 
# import warnings
import tensorflow as tf
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import socket
import pyaudio
import wave
import struct
import numpy as np
CHUNK = 16000 # read each 1000 miliseconds
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 20
WORD = 6000
WORD_HALF = int(WORD/2)
WORD_QUAD = int(WORD/2)
# circular_buf_size = 50
p = pyaudio.PyAudio()
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
model.summary()
# open the file for reading.
wf = wave.open("test.wav", 'rb')
# open stream based on the wave object which has been input.
# stream = p.open(format =
#                 p.get_format_from_width(wf.getsampwidth()),
#                 channels = wf.getnchannels(),
#                 rate = wf.getframerate(),
#                 input = True)
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
# stream = p.open("test.wav",format=FORMAT,
#                 channels=CHANNELS,)
frames = []
features = []
waves = []
words = []
circular_index = 0
os.system( 'cls' )
print("* recording")
inputDataNew = np.ndarray(shape=(CHUNK), dtype=float)
inputDataOld = np.ndarray(shape=(WORD_QUAD), dtype=float)
last_word_time = 0
i=0
while 1:
    data = stream.read(CHUNK) 
    #convert to float array
    for j in range(0, int(len(data)/4)):
        value = struct.unpack('f', data[j*4:j*4+4])
        inputDataNew[j] = value[0]
    #megre end of old and new
    inputData = np.concatenate((inputDataOld,inputDataNew))
    inputDataOld = inputDataNew[CHUNK-WORD_QUAD:].copy()
    i=i+1
    if i<2:
        continue
    else:
        
        energy = abs(inputData)
        mean = np.mean(energy)
        value = 0
        max_val = 0
        max_idn = 0
        for idn in range(0,len(energy)):
            value += (energy[idn]-value)/300
            if (idn>WORD_HALF) & (idn<(len(inputData)-WORD_HALF )):
                if max_val<value:
                    max_val = value
                    max_idn = idn
        if max_val>(mean*2):
            global_time = i*CHUNK+max_idn-WORD_QUAD
            if (global_time-last_word_time)< WORD:
                continue
            last_word_time = global_time
            data = inputData[max_idn-WORD_HALF:max_idn+WORD_HALF]
            norm = np.mean(abs(data))*250
            data = data/norm
            feature_vector_1 = librosa.feature.chroma_stft(y=data, sr=16000)
            feature_vector_2 = librosa.feature.melspectrogram(y=data, sr=16000)[:40]
            feature_vector = np.concatenate((feature_vector_1,feature_vector_2))
            modelInput = np.array(feature_vector).reshape(1,feature_vector.shape[0],feature_vector.shape[1],1)
            output = model.predict(modelInput)
            indexMax = np.argmax(output[0], axis=0) 
            confident = output[0][indexMax]
            if ((indexMax!=10)&(confident>0.8)):
#             os.system( 'cls' )
                print ("ID:",len(waves)," prediction:",indexMax,"confident:",output[0][indexMax]," time:",global_time/RATE)
                waves.append(inputData)
                words.append(data)
                features.append(feature_vector)
                byte_message = (indexMax)
                opened_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                opened_socket.sendto(byte_message, ("127.0.0.1", 5005))
#             plt.figure(figsize=(10, 4))
#             librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
#             plt.colorbar()
#             plt.title('Chromagram')
#             plt.tight_layout()
#             plt.show()
print("* done recording")    
