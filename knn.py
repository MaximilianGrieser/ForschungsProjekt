import os

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from sklearn.neighbors import KNeighborsClassifier as wknn

SIZE = 30
DEGREE = 10

k = 3 # k neighbors
q = 1 # monkowski distance
w = "distance" # weighting


def main():
    data = []
    labels = []
    for p in os.listdir("audio"):
        if not os.path.isfile(p):
            for f in os.listdir("audio/"+p):
                s = AudioSegment.from_file(
                    "audio/{}/{}".format(p, f), format="wav")
                fdata = parse_fname(f)
                loudPoly = getLoudPoly(s)
                data.append(loudPoly)
                labels.append(fdata.get("emotion_n"))
        print(p)
        break
        

    test_data = []
    test_data.append(data.pop(0))
    labels.pop(0)
    test_data.append(data.pop(30))
    labels.pop(30)

    model = wknn(n_neighbors=k, weights=w, p=q)
    model.fit(data, labels)

    # predicting
    print(model.predict_proba(test_data))
    


def getLoudPoly(s):
    xdata = []
    ydata = []
    chunks = make_chunks(s, SIZE)
    for i in range(0, len(chunks)):
        if chunks[i].dBFS != -np.inf:
            xdata.append((i+1)*SIZE)
            ydata.append(chunks[i].dBFS)
    p = np.polyfit(np.array(xdata), np.array(ydata), DEGREE)
    print(p)
    return p


def parse_fname(fname):
    '''
    # 03-01-01-01-01-01-01.wav
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    '''
    modality = ["full-AV", "video", "audio"]
    vocal = ["speech", "song"]
    emotion = ["neutral", "calm", "happy", "sad",
               "angry", "fearful", "disgust", "suprised"]
    intensity = ["normal", "strong"]
    statement = ["kids", "dogs"]
    repetition = ["1st rep", "2nd rep"]
    actor = ["female", "male"]

    fname = fname.strip(".wav")
    farray = fname.split("-")
    fdict = {
        "modality": modality[int(farray[0])-1],
        "vocal": vocal[int(farray[1])-1],
        "emotion": emotion[int(farray[2])-1],
        "emotion_n": int(farray[2]),
        "intensity": intensity[int(farray[3])-1],
        "statement": statement[int(farray[4])-1],
        "repetition": repetition[int(farray[5])-1],
        "actor": 1 if (int(farray[6]) % 2 == 0) else 0  # 0 = blue , 1 = red
    }
    return fdict


if __name__ == "__main__":
    main()
