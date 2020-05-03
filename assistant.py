#!/usr/bin/python

import deepspeech
import numpy as np
import os
import pyaudio
import time
import wave

from termcolor import colored

#speech to text
def speech(filename, model):
    w = wave.open(filename, 'r')
    rate = w.getframerate()
    frames = w.getnframes()
    buffer = w.readframes(frames)
    #print(rate)
    #print(model.sampleRate())
    type(buffer)

    import numpy as np
    data16 = np.frombuffer(buffer, dtype=np.int16)
    type(data16)


    text = model.stt(data16)
    return text;

#INIT
# DeepSpeech parameters
DEEPSPEECH_MODEL_DIR = './some/workspace/path/ds06/deepspeech-0.6.0-models'
MODEL_FILE_PATH = os.path.join(DEEPSPEECH_MODEL_DIR, 'output_graph.pbmm')
BEAM_WIDTH = 500
LM_FILE_PATH = os.path.join(DEEPSPEECH_MODEL_DIR, 'lm.binary')
TRIE_FILE_PATH = os.path.join(DEEPSPEECH_MODEL_DIR, 'trie')
LM_ALPHA = 0.75
LM_BETA = 1.85

# Make DeepSpeech Model
model = deepspeech.Model(MODEL_FILE_PATH, BEAM_WIDTH)
model.enableDecoderWithLM(LM_FILE_PATH, TRIE_FILE_PATH, LM_ALPHA, LM_BETA)

# Create a Streaming session
context = model.createStream()




#Main Program
print colored("Hello,\nState your name",'red');
text = speech (filename = './audio/name.wav', model = model)
print colored(text,'green');

print colored("What time is your appointment?",'red');
text = speech (filename = './audio/time.wav', model = model)
print colored(text,'green');

print colored("Ok,Do you have any medical reports?",'red');
text = speech (filename = './audio/yes.wav', model = model)
print colored(text,'green');

if text == "yes":
    print colored("Email them to doc@gmail.com",'red');

print colored("Please sit and wait for your turn",'red');
