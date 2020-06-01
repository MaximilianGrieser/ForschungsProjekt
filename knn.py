import os

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from sklearn.neighbors import KNeighborsClassifier as wknn
from sklearn.model_selection import train_test_split
import audiosegment


PRED_DEBUG = False  # print result and correctness of every prediction
# POLYFIT CONFIG
SIZE = 30           # audio snippet size (in ms) for determining loudness/time
DEGREE = 10         # polynom degree for fitting

# KNN CONFIG
k = 3               # k neighbors
q = 1               # monkowski distance
w = "distance"      # weighting
TEST_SIZE = 0.25    # proportion of the dataset to include in the test split
RANDOM_STATE = 0    # shuffle before split, int for reproducible output


def main():
    # read audio data from file
    data, labels = readData()

    # split data into test and train
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # train model
    model = wknn(n_neighbors=k, weights=w, p=q)
    model.fit(x_train, y_train)

    # predicting
    prediction = model.predict(x_test)
    prediction_prob = model.predict_proba(x_test)

    # print and evaluate results
    evalPrediction(prediction, prediction_prob, x_test, y_test, len(data))
    printConfig()


def readData():
    """
    ! FILE STRUCTURE HARDCODED !
    ----------------------------
    Function reads all available audio files and returns an array with all relevant data points:
        - Frequency (Over time, Maximum, Average)
        - Loudness (Over time, Maximum, Average)

    Returns:
        (array, array) -- Returns a tuple consisting of data and related labels
    """
    counter = 0     # loop index
    data = []       # list of data used for training/testing
    labels = []     # list of labels (correct emotions) related to traindata
    #                   (eg traindata[0] is labels[0] emotion)
    for p in os.listdir("audio"):
        if not os.path.isfile(p):
            for f in os.listdir("audio/"+p):
                s1 = AudioSegment.from_file(
                    "audio/{}/{}".format(p, f), format="wav")
                s2 = audiosegment.from_file(
                    "audio/{}/{}".format(p, f))
                freqChange, freqAvg, freqMax = getFrequency(s2)
                fdata = parseFName(f)
                loudPoly, dBFS, maxDBFS = getLoudness(s1)
                data.append(np.append(np.concatenate((loudPoly, getDerivative(
                    loudPoly), freqChange, getDerivative(freqChange))), [dBFS, maxDBFS, freqAvg, freqMax]))
                labels.append(fdata.get("emotion_n"))
                counter += 1
        print("{}: {} files".format(p, counter))
    return data, labels


def getLoudness(s):
    """Returns Loudness attributes:
        - coefficients of a curve fitted to loudness over time
        - average decibel relative to the maximum possible loudness
        - maximum decibel

    Arguments:
        s {AudioSegment} -- Audio as AudioSegment object

    Returns:
        (tuple, float, float) -- ((coefficients), dBFS, maxDBFS)
    """
    xdata = []
    ydata = []
    chunks = make_chunks(s, SIZE)
    for i in range(0, len(chunks)):
        if chunks[i].dBFS != -np.inf:
            xdata.append((i+1)*SIZE)
            ydata.append(chunks[i].dBFS)
    p = np.polyfit(np.array(xdata), np.array(ydata), DEGREE)
    return p, s.dBFS, s.max_dBFS


def getFrequency(s):
    """Return Frequency attributes:
        - coefficients of a curve fitted to frequency over time
        - maximum frequency
        - average frequency

    Arguments:
        s {AudioSegment} -- Audio as AudioSegment object

    Returns:
        (tuple, float, float) -- ((coefficients), maxFreq, avgFreq)
    """
    xdata = []
    ydata = []
    chunks = make_chunks(s, SIZE)
    for i in range(0, len(chunks)):
        void, hist_vals = chunks[i].fft()
        del void
        hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)

        avg = np.mean(hist_vals_real_normed)

        xdata.append((i+1)*SIZE)
        ydata.append(avg)
    p = np.polyfit(np.array(xdata), np.array(ydata), DEGREE)
    max_value = np.amax(np.array(ydata))
    avg_value = np.mean(np.array(ydata))
    return p, avg_value, max_value


def parseFName(fname):
    '''
    ! PLACEHOLDER FUNCTION !
    ------------------------
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


def printConfig():
    """Outputs all relevant values of the current configuration
    """
    print("\n# LOUDNESS")
    print("SIZE = {}".format(SIZE))
    print("DEGREE = {}".format(DEGREE))
    print("# KNN CONFIG")
    print("k = {}".format(k))
    print("q = {}".format(q))
    print("w = {}".format(w))
    print("TEST_SIZE = {}".format(TEST_SIZE))
    print("RANDOM_STATE = {}".format(RANDOM_STATE))


def getDerivative(poly):
    """This function calculates the derivative of a n-th degree polynomial

    Arguments:
        poly {array} -- Polynomial coefficients, in the form of e.g. [a, b, c] where ax^2+bx+c.

    Returns:
        array -- Polynomial coefficients of the derivative, highest power first.
    """
    # This function calculates the derivative of a n-th degree polynomial
    # 'poly' is an array in the form of e.g. [a, b, c] where ax^2+bx+c
    # An array in similar form like 'poly' is returned.
    if len(poly) == 1:
        return [0]
    deriv_poly = [poly[i] * (len(poly)-i-1) for i in range(0, len(poly)-1)]
    return deriv_poly


def evalPrediction(prediction, prediction_prob, x_test, y_test, data_n):
    """Print test results, including prediction results

    Arguments:
        prediction {array} -- Class labels for each data sample
        prediction_prob {array} -- The class probabilities of the input samples.
        x_test {array} -- Data samples used for testing
        y_test {array} -- Labels related to testing data samples
        data_n {int} -- Length of overall dataset (training + testing)
    """
    emotion = ["neutral", "calm", "happy", "sad",
               "angry", "fearful", "disgust", "suprised"]
    accuracy = 0
    x_test_n = len(x_test)
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
                                                x_test_n)*100,
                                               accuracy,
                                               x_test_n))
    print("\tn_train={} n_test={}".format(data_n-x_test_n, x_test_n))


if __name__ == "__main__":
    main()
