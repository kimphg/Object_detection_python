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
import matplotlib.pyplot as plt
import librosa.display

from numpy import load
from numpy import save
from numpy import asarray
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'noise']
all_wave = load('all_wave.npy')
all_label = load('all_label.npy')
all_feature=[]
for word in all_wave:
    norm = np.mean(abs(word))*250
    normal_word = word/norm
    feature_vector_1 = librosa.feature.chroma_stft(y=normal_word, sr=16000)
    feature_vector_2 = librosa.feature.melspectrogram(y=normal_word, sr=16000)[:40]
    feature_vector = np.concatenate((feature_vector_1,feature_vector_2))
    all_feature.append(feature_vector)
from sklearn.preprocessing import LabelEncoder
all_feature = np.array(all_feature)
feature_vector = all_feature[0]
print(all_feature.shape)
feature_size = feature_vector.shape

le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)
from keras.utils import np_utils
y=np_utils.to_categorical(y, num_classes=len(labels))
# all_feature = np.array(all_feature).reshape(-1,feature_size[0],feature_size[1],1)
all_feature2 = np.array(all_feature).reshape(-1,feature_size[0],feature_size[1],1)
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_feature2),np.array(y),stratify=y,test_size = 0.3,random_state=333,shuffle=True)
#thiết kế mô hình deep learning 2D
from keras.models import Model, Sequential
from keras.layers import *
from keras.activations import *
from keras.optimizers import Adam,RMSprop,SGD
model = Sequential()
# model1.add(Conv1D(8, kernel_size=15, strides=1, padding='valid',
#                   input_shape=(feature_size[0],1)))
model.add(Conv2D(16, (7, 7), padding='same',
                 input_shape=(feature_size[0],feature_size[1],1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(11, activation='softmax'))

# adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
# learning rate schedule
def lr_schedule(epoch):
#     return 0.001
    if epoch < 5:
        return 0.016
    elif epoch < 10:
        return 0.008
    elif epoch < 20:
        return 0.004
    elif epoch < 40:
        return 0.002
    else:
        return 0.001
callbacks = [LearningRateScheduler(schedule=lr_schedule, verbose=1),
            ModelCheckpoint(os.path.join("./", "model.h5"),
            monitor='val_loss', verbose=1, save_best_only=True)]
#luyện mô hình 
classifier = model.fit(x_tr,
                    y_tr,validation_data=( x_val, y_val),
                    callbacks=callbacks,
                    validation_steps = 1,
                    steps_per_epoch = 5,
                    epochs=40,
                    batch_size=None)
# lưu kiến trúc của model đã luyện vào file json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# lưu các trọng số của model
# model.save_weights("model.h5")
print("Saved model architecture to disk")
# model.summary()
