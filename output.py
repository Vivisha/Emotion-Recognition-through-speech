#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[2]:


def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result


# In[3]:



emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
# Emotions to observe
observed_emotions=['neutral','calm','happy','sad','angry','fearful', 'disgust','surprised']


# In[14]:


def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\LENOVO\\Downloads\\archive\\audio_speech_actors_01-24\\Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, train_size= 0.80,random_state=9)


# In[15]:



import time
x_train,x_test,y_train,y_test=load_data(test_size=0.20)


# In[16]:


print((x_train.shape[0], x_test.shape[0]))


# In[17]:


print(f'Features extracted: {x_train.shape[1]}')


# In[18]:


model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', 
                    max_iter=500)


# In[19]:


model.fit(x_train,y_train)


# In[20]:


MLPClassifier(activation='relu', alpha=0.01, batch_size=256, beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(300,), learning_rate='adaptive',
              learning_rate_init=0.001, max_iter=500, momentum=0.9,
              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
              random_state=None, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)


# In[21]:



y_pred=model.predict(x_test)


# In[22]:


accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[23]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[24]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print (matrix)


# In[25]:


import pickle
# Save the Model to file in the current working directory
#For any new testing data other than the data in dataset

Pkl_Filename = "Emotion_Recognition_Through_Speech.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)


# In[26]:


with open(Pkl_Filename, 'rb') as file:  
    Emotion_Recognition_Through_Speech = pickle.load(file)

Emotion_Recognition_Through_Speech


# In[27]:


A=Emotion_Recognition_Through_Speech.predict(x_test)
A


# In[28]:


new_feature= extract_feature("C:\\Users\\LENOVO\\Downloads\\archive\\audio_speech_actors_01-24\\Actor_01\\03-01-01-01-01-01-01.wav",mfcc=True, chroma=True, mel=True)
new_feature.shape
Emotion_Recognition_Through_Speech.predict([new_feature])


# In[30]:


file = 'C:\\Users\\LENOVO\\Downloads\\archive\\audio_speech_actors_01-24\\Actor_01\\03-01-03-02-02-01-01.wav'

new_feature= extract_feature(file, mfcc=True, chroma=True, mel=True)


Emotion_Recognition_Through_Speech.predict([new_feature])


# In[ ]:




