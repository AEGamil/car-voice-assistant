import numpy as np
import matplotlib.mlab as mlab
import sys
import torch
import torch.nn as nn
import pyaudio
from queue import Queue
import sys

import wave

Tx = 1198
n_freq = 101
Ty = 296

class TW_model(nn.Module):

    def __init__(self, input_channels, conv_out,hidden_size, dropout_prob):
        super(TW_model, self).__init__()

        self.conv_layer_0 = nn.Conv1d(input_channels, conv_out, kernel_size=15, stride=4)
        self.batch_norm_layer_0 = nn.BatchNorm2d(num_features=conv_out)
        self.relu_0 = nn.ReLU()
        self.dropout_layer_0 = nn.Dropout(p=dropout_prob)

        self.gru_layer_1 = nn.GRU(conv_out, hidden_size, batch_first=True)
        self.dropout_layer_1 = nn.Dropout(p=dropout_prob)
        self.batch_norm_layer_1 = nn.BatchNorm2d(hidden_size)

        self.gru_layer_2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer_2_0 = nn.Dropout(p=dropout_prob)
        self.batch_norm_layer_2 = nn.BatchNorm2d(hidden_size)
        self.dropout_layer_2_1 = nn.Dropout(p=dropout_prob)

        self.dense_3 = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid_3 = nn.Sigmoid()


    def forward(self, x):
        x = x.reshape(n_freq,Tx)
        x = self.conv_layer_0(x)
        x = x.reshape(1, 1,x.shape[1],x.shape[0])
        x = x.permute(0, 3, 1, 2)
        x = self.batch_norm_layer_0(x)
        x = self.relu_0(x)
        x = self.dropout_layer_0(x)
        x=x.reshape(1,x.shape[3],x.shape[1])
        x,_ = self.gru_layer_1(x)
        x = self.dropout_layer_1(x)
        x = x.reshape(1, 1,x.shape[1],x.shape[2])
        x = x.permute(0, 3, 1, 2)
        x = self.batch_norm_layer_1(x)
        x=x.reshape(1,x.shape[3],x.shape[1])
        x,_ = self.gru_layer_2(x)
        x = self.dropout_layer_2_0(x)
        x = x.reshape(1, 1,x.shape[1],x.shape[2])
        x = x.permute(0, 3, 1, 2)
        x = self.batch_norm_layer_2(x)
        x = self.dropout_layer_2_1(x)
        x = x.reshape(1, 1,x.shape[3],x.shape[1])
        x = self.dense_3(x)
        output = self.sigmoid_3(x)
        return output
model = TW_model(input_channels=n_freq,conv_out=196,hidden_size=128,dropout_prob=0.02)
model.load_state_dict(torch.load('TWDetection_model_cpu (6).pt', map_location='cpu'))
print('loaded the twd')

def detect_triggerword_spectrum(x):
    x = torch.from_numpy(x.astype(dtype=np.float32))
    predictions = model(x)
    return predictions.reshape(-1)

def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold):
    i = 0
    pres = predictions
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True      
        else:
            level = pred
        i+=1
    return False
    
chunk_duration = 0.5 # Each read length in seconds from mic.
fs = 48000 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 2
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)

def get_spectrogram(data):
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx


def play_chime(filename):
    wav_file = wave.open(filename, 'rb')
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(wav_file.getsampwidth()),
                        channels=wav_file.getnchannels(),
                        rate=wav_file.getframerate(),
                        output=True)
    data = wav_file.readframes(wav_file.getnframes())
    stream.write(data)
    # Stoping the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()


def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream

# Queue to communiate between the audio callback and main thread
q = Queue()
silence_threshold = 600

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')
i = 0

def callback(in_data, frame_count, time_info, status):
    global run, timeout,i,data, silence_threshold           
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('-')
        return (in_data, pyaudio.paContinue)
    else:
        sys.stdout.write('.')
    data = np.append(data,data0)
    i += 1  
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

def trigger_word_phase():
    global i 
    stream = get_audio_input_stream(callback)
    stream.start_stream()
    try:
        while True:
            data = q.get()
            spectrum = get_spectrogram(data)
            preds = detect_triggerword_spectrum(spectrum)
            new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration,threshold=0.4)
            if new_trigger and i > 3 :
                sys.stdout.write('1')
                play_chime('chime.wav')
                i = 0
                stream.stop_stream()
                stream.close()
                data = np.zeros(feed_samples, dtype='int16')
                return True
    except (KeyboardInterrupt, SystemExit):
        stream.stop_stream()
        stream.close()
     
# trigger_word_phase()

