from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np


encoder = VoiceEncoder()

def compareVoice(path1, path2):
    fpath1 = Path(path1)
    wav1 = preprocess_wav(fpath1)
    fpath2 = Path(path2)
    wav2 = preprocess_wav(fpath2)
    encoder = VoiceEncoder()
    embed1 = encoder.embed_utterance(wav1)
    embed2 = encoder.embed_utterance(wav2)
    similarity = np.inner(embed1, embed2)
    return similarity

np.set_printoptions(precision=3, suppress=True)

print(compareVoice("./alex1.mp3", "./alex2.mp3"))
