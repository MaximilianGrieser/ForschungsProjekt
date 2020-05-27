import os

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from sklearn.neighbors import KNeighborsClassifier as wknn
from sklearn.model_selection import train_test_split


PRED_DEBUG = True   # print result and correctness of every prediction
# LOUDNESS CONFIG
SIZE = 30           # audio snippet size (in ms) for determining loudness/time
DEGREE = 10         # polynom degree for fitting

# KNN CONFIG
k = 3               # k neighbors
q = 1               # monkowski distance
w = "distance"      # weighting
TEST_SIZE = 0.2     # proportion of the dataset to include in the test split
RANDOM_STATE = 0    # shuffle before split, int for reproducible output

emotion = ["neutral", "calm", "happy", "sad",
           "angry", "fearful", "disgust", "suprised"]


def main():
    counter = 0     # loop index
    traindata = []  # list of data used for training
    labels = []     # list of labels (correct emotions) related to traindata
    #                   (eg traindata[0] is labels[0] emotion)
    for p in os.listdir("audio"):
        if not os.path.isfile(p):
            for f in os.listdir("audio/"+p):
                s = AudioSegment.from_file(
                    "audio/{}/{}".format(p, f), format="wav")
                fdata = parse_fname(f)
                loudPoly = get_loudpoly(s)  # get loudness/time

                traindata.append(loudPoly)
                labels.append(fdata.get("emotion_n"))

                counter += 1
        print("{}: {} files".format(p, counter))
        break

    # split data into test and train
    x_train, x_test, y_train, y_test = train_test_split(
        traindata, labels, test_size=0.2, random_state=0)

    # train model
    model = wknn(n_neighbors=k, weights=w, p=q)
    model.fit(x_train, y_train)

    # predicting
    accuracy = 0
    prediction = model.predict(x_test)
    prediction_prob = model.predict_proba(x_test)

    # print results
    for i in range(0, len(prediction)):
        if y_test[i] == prediction[i]:
            result = "TRUE"
            accuracy += 1
        else:
            result = "FALSE"
        if PRED_DEBUG:
            print("[{:5s}]\nprediction={} key={}\nprob={}\n\t   {}\n---\n".format(result,
                                                                                  emotion[prediction[i] -
                                                                                          1].upper(),
                                                                                  emotion[y_test[i] -
                                                                                          1].upper(),
                                                                                  np.array_repr(
                                                                                      prediction_prob[i]).replace('\n', ''),
                                                                                  emotion))
    print("\t### RESULT ###\n")
    print("\t{:3.2f}% ACCURACY [{}/{}]".format((accuracy /
                                                len(x_test))*100,
                                               accuracy,
                                               len(x_test)))
    print("\tn_train={} n_test={}".format(len(x_train), len(x_test)))
    print_config()


def get_loudpoly(s):
    xdata = []
    ydata = []
    chunks = make_chunks(s, SIZE)
    for i in range(0, len(chunks)):
        if chunks[i].dBFS != -np.inf:
            xdata.append((i+1)*SIZE)
            ydata.append(chunks[i].dBFS)
    p = np.polyfit(np.array(xdata), np.array(ydata), DEGREE)
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


def print_config():
    print("\n# LOUDNESS")
    print("SIZE = {}".format(SIZE))
    print("DEGREE = {}".format(DEGREE))
    print("# KNN CONFIG")
    print("k = {}".format(k))
    print("q = {}".format(q))
    print("w = {}".format(w))
    print("TEST_SIZE = {}".format(TEST_SIZE))
    print("RANDOM_STATE = {}".format(RANDOM_STATE))


if __name__ == "__main__":
    main()
