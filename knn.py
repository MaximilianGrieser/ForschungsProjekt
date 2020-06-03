import configparser
import os
import sys

import audiosegment
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as wknn


def main():
    cfg = loadConfig()
    # TODO blackbox class etc
    blackBox(config=cfg)


def blackBox(config):
    # read audio data from file
    data, labels = [], []
    for db in config['DATABASES']:
        loop_data, loop_labels = readData(db, config)
        data, labels = data + loop_data, labels + loop_labels

    # split data into test and train
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=float(config['KNN']['test_size']), random_state=int(config['KNN']['random_state']))

    # train model
    model = wknn(n_neighbors=int(config["KNN"]["k"]),
                 weights=config["KNN"]["w"], p=int(config["KNN"]["q"]))
    model.fit(x_train, y_train)

    # predicting
    prediction = model.predict(x_test)
    prediction_prob = model.predict_proba(x_test)

    # print and evaluate results
    evalPrediction(prediction, prediction_prob, x_test, y_test, len(
        data), int(config['DEBUG']['prediction_details']))
    printConfig(config)


def loadConfig():
    """Loads the config file and checks if the file is valid

    Returns:
        [configObject] -- [Config file as object]
    """    
    cfg = configparser.ConfigParser(allow_no_value=True)
    cfile = cfg.read("config.ini")
    # TODO check for valid config
    if not cfile:
        print("ERR: Failed loading config")
        sys.exit()
    return cfg


def readData(database, config):
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
    error = 0       # corrupted files
    data = []       # list of data used for training/testing
    labels = []     # list of labels (correct emotions) related to traindata
    #                   (eg traindata[0] is labels[0] emotion)
    for f in os.listdir("audio/"+database):
        try:
            s1 = AudioSegment.from_file(
                "audio/{}/{}".format(database, f), format="wav")
            s2 = audiosegment.from_file("audio/{}/{}".format(database, f))
        except FileNotFoundError:
            error += 1
            print("[{} ({})] {}/{}".format(counter,
                                           error, database, f), end="\r")
            continue
        freqChange, freqAvg, freqMax = getFrequency(s2, int(
            config['POLYFIT']['snippet_size']), int(config['POLYFIT']['poly_degree']))
        fdata = parseFName(f)
        loudPoly, dBFS, maxDBFS = getLoudness(s1, int(
            config['POLYFIT']['snippet_size']), int(config['POLYFIT']['poly_degree']))
        data.append(np.append(np.concatenate((getDerivative(
                    loudPoly),  getDerivative(freqChange))), [dBFS, maxDBFS, freqAvg, freqMax]))
        labels.append(fdata.get("emotion_n"))
        counter += 1
        print("[{} ({})] Loading {}/{}...\t\t\t".format(counter,
                                                        error, database, f), end="\r")
    print("[{} ({})] Finished loading {}\t\t\t\n".format(counter,
                                                         error, database), end="\r")
    return data, labels


def getLoudness(s, n, degree):
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
    chunks = make_chunks(s, n)
    for i in range(0, len(chunks)):
        if chunks[i].dBFS != -np.inf:
            xdata.append((i+1)*n)
            ydata.append(chunks[i].dBFS)
    p = np.polyfit(np.array(xdata), np.array(ydata), degree)
    return p, s.dBFS, s.max_dBFS


def getFrequency(s, n, degree):
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
    chunks = make_chunks(s, n)
    for i in range(0, len(chunks)):
        void, hist_vals = chunks[i].fft()
        del void
        hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)

        avg = np.mean(hist_vals_real_normed)

        xdata.append((i+1)*n)
        ydata.append(avg)
    p = np.polyfit(np.array(xdata), np.array(ydata), degree)
    max_value = np.amax(np.array(ydata))
    avg_value = np.mean(np.array(ydata))
    return p, avg_value, max_value


def parseFName(name):
    """Parse the file name extracting information about the audio file import for labeling. Only works if the naming scheme is correct!

    Arguments:
        name {file} -- File

    Returns:
        dict -- Returns dict with emotion, actor and id
    """    
    emotion = ["happy", "suprise", "angry", "sad",
               "fear", "disgust", "neutral"]
    actor = ["male", "female"]
    name = name.strip(".wav")
    # doing this twice beacuse some f'd the toronto dataset
    name = name.strip(".wav")
    fdata = name.split("-")
    fdict = {
        "emotion": emotion[int(fdata[0])-1],
        "emotion_n": int(fdata[0]),
        "actor": actor[int(fdata[1])-1],
        "actor_n": int(fdata[1])-1,
        "id": fdata[2]
    }
    return fdict


def printConfig(config):
    """Outputs all relevant values of the current configuration
    """
    for section in config.sections():
        print("[{}]".format(section))
        for key in config[section]:
            if config[section][key] == None:
                print("{}".format(key))
            else: 
                 print("{}={}".format(key, config[section][key]))


def getDerivative(poly):
    """This function calculates the derivative of a n-th degree polynomial

    Arguments:
        poly {array} -- Polynomial coefficients, in the form of e.g. [a, b, c] where ax^2+bx+c.

    Returns:
        array -- Polynomial coefficients of the derivative, highest power first.
    """
    if len(poly) == 1:
        return [0]
    deriv_poly = [poly[i] * (len(poly)-i-1) for i in range(0, len(poly)-1)]
    return deriv_poly


def evalPrediction(prediction, prediction_prob, x_test, y_test, data_n, debug=False):
    """Print test results, including prediction results

    Arguments:
        prediction {array} -- Class labels for each data sample
        prediction_prob {array} -- The class probabilities of the input samples.
        x_test {array} -- Data samples used for testing
        y_test {array} -- Labels related to testing data samples
        data_n {int} -- Length of overall dataset (training + testing)
    """
    emotion = ["happy", "suprise", "angry", "sad",
               "fear", "disgust", "neutral"]
    accuracy = 0
    x_test_n = len(x_test)
    for i in range(0, len(prediction)):
        if y_test[i] == prediction[i]:
            result = "TRUE"
            accuracy += 1
        else:
            result = "FALSE"
        if debug:
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
