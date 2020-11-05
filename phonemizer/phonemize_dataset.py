from phonemizer import phonemize
import torch
import torchaudio

def convert(data):
    utterances = []
    for i in range(len(data)):
        utterances.append(phonemize(data.__getitem__(i)[2]))
        #Comment previous line and uncomment to change backend
        #utterances.append(phonemize(data.__getitem__(i)[2], backend='espeak'))
    return utterances