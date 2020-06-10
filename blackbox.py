import configparser
import hashlib
import os
import sys
import time as t

import audiosegment
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# TODO do logging
# TODO use config .get() with fallback values


class BlackBox:
    def __init__(self, config):
        self.config_file = config
        self._config = self._loadConfigFile(config)
        self._loadConfig(wipe=True)

        self.data = []
        self.labels = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        rng = np.random.default_rng()
        self.random_1 = rng.integers(0, high=12)*2
        self.random_2 = (rng.integers(0, high=12)*2)+1
        print(f"[+] random numbers {self.random_1}, {self.random_2}")

        self.predictions = [[], []]
        self.predictions_prob = [[], []]
        self.accuracy_topf = []

        self._createModel()

    def loadData(self, database=None):
        if database is None:
            for db in self.databases:
                self.loadData(database=db)
        else:
            counter = 0
            error = 0
            cache_data = []
            cache_labels = []
            if self._isCached(database):
                data = np.load(f"cache/{database}.npy", allow_pickle=True)
                if database.split("_")[0] == "actor" and (int(database.split("_")[1]) == self.random_1 or int(database.split("_")[1]) == self.random_2):
                    for element in data[0]:
                        self.x_test2.append(element)
                    for element in data[1]:
                        self.y_test2.append(element)
                    print(f"[+] [{len(data[0])}] Loaded {database} from cache and added to EXTERNE VALIDITÄT (TOPF)")
                else:
                    for element in data[0]:
                        self.data.append(element)
                    for element in data[1]:
                        self.labels.append(element)
                    print(f"[+] [{len(data[0])}] Loaded {database} from cache")
            else:
                for f in os.listdir("audio/"+database):
                    try:
                        s1 = AudioSegment.from_file(
                            f"audio/{database}/{f}", format="wav")
                        s2 = audiosegment.from_file(f"audio/{database}/{f}")
                    except FileNotFoundError:
                        error += 1
                        print(
                            f"[+] [{counter} ({error})] {database}/{f}\t\t\t", end="\r")
                        continue
                    freqChange, freqAvg, freqMax = self._getFrequency(s2)
                    fdata = self._parseName(f)
                    loudPoly, dBFS, maxDBFS = self._getLoudness(s1)
                    if database.split("_")[0] == "actor" and (int(database.split("_")[1]) == self.random_1 or int(database.split("_")[1]) == self.random_2):
                        self.x_test2.append(np.append(np.concatenate((self._getDerivative(
                            loudPoly),  self._getDerivative(freqChange))), [dBFS, maxDBFS, freqAvg, freqMax]))
                        self.y_test2.append(
                            [fdata.get("emotion_n"), fdata.get("actor_n")])
                    else:
                        self.data.append(np.append(np.concatenate((self._getDerivative(
                            loudPoly),  self._getDerivative(freqChange))), [dBFS, maxDBFS, freqAvg, freqMax]))
                        self.labels.append(
                            [fdata.get("emotion_n"), fdata.get("actor_n")])
                    cache_data.append(np.append(np.concatenate((self._getDerivative(
                        loudPoly),  self._getDerivative(freqChange))), [dBFS, maxDBFS, freqAvg, freqMax]))
                    cache_labels.append(
                        [fdata.get("emotion_n"), fdata.get("actor_n")])
                    counter += 1
                    print(
                        f"[+] [{counter} ({error})] Loading {database}/{f}...\t\t\t", end="\r")
                print(
                    f"[+] [{counter} ({error})] Finished loading {database}\t\t\t\n", end="\r")
                self._cacheData(database, [cache_data, cache_labels])
            self.databases_loaded.append(database)

    def clearCache(self):
        counter = 0
        for f in os.listdir("cache/"):
            os.remove(f"cache/{f}")
            counter += 1
        print("[+] Cleared {counter} file(s) from cache")

    def _isCached(self, database):
        if os.path.isfile(f"cache/{database}.npy"):
            return True
        else:
            return False

    def _cacheData(self, database, data):
        try:
            np.save(f"cache/{database}.npy", data)
            print(f"[+] Added {database} to cache")
        except FileNotFoundError:
            print("[x] ERROR: Cache directory not found")

    def train(self):
        counter = 0
        valid = False
        if len(self.databases_loaded) > 0:
            if self._test_size > 0.0:
                while (not valid):
                    labels = []
                    self._splitData()
                    for array in self.y_test:
                        labels.append(array[1])
                    if labels.count(1)/len(labels) > 0.3 and labels.count(1)/len(labels) < 0.7:
                        valid = True
                    else:
                        print(
                            f"[-] [{counter}] Splitting data... Last ratio {labels.count(0)/len(labels)}m/{labels.count(1)/len(labels)}f \t\t", end="\r")
                    if counter >= 1000:
                        print("")
                        print(
                            f"[x] Unable to find valid data split after {counter} tries.")
                        sys.exit()
                    counter += 1
                print(
                    f"[+] Found valid data split after {counter} run(s). {labels.count(0)/len(labels)} male / {labels.count(1)/len(labels)} female ")
            else:
                self._splitData()
            labels = []
            for array in self.y_train:
                labels.append(array[0])
            b = np.array(self.x_train).view(np.uint8)
            self.train_hash = hashlib.sha1(b).hexdigest()
            self.model.fit(self.x_train, labels)
            self.trained = True
        else:
            print("[x] ERROR: No database loaded")

    def predict(self, f=None):
        if len(self.databases_loaded) == 0:
            print("[x] ERROR: No database loaded")
            sys.exit()
        if self.trained:
            if f is not None:
                data = self._getAudioAttributes(f)
                prediction = self.model.predict(data)
                prediction_prob = self.model.predict_proba(data)
                return prediction, prediction_prob
            elif self._test_size == 0.0:
                self.accuracy = None
            else:
                b = np.array(self.x_test).view(np.uint8)
                self.test_hash = hashlib.sha1(b).hexdigest()

                labels = []
                for array in self.y_test:
                    labels.append(array[0])

                self.predictions[0].append(self.model.predict(self.x_test))
                self.predictions_prob[0].append(self.model.predict_proba(self.x_test))
                self.accuracy = self.model.score(self.x_test, labels)

                labels2 = []
                for array in self.y_test2:
                    labels2.append(array[0])

                self.predictions[1].append(self.model.predict(self.x_test2))
                self.predictions_prob[1].append(self.model.predict_proba(self.x_test2))
                self.accuracy_2 = self.model.score(self.x_test2, labels2)


                self._evaluatePrediction()
        else:
            print("[x] ERROR: Model not trained")

    def printConfig(self):
        config = self._config
        for section in config.sections():
            print(f"[+] [{section}]")
            for key in config[section]:
                if config[section][key] == None:
                    print(f"    {key}")
                else:
                    print(f"    {key}={config[section][key]}")

    def updateConfig(self, section, option, value, wipe=False):
        try:
            if section == "POLYFIT":
                print(
                    f"[-] Updating [POLYFIT] parameters invalidate cache files. It is strongly recommended to 'clearCache()'")
            self._config.set(section, option, str(value))
            self._loadConfig(wipe=wipe)
            print(f"[+] Config updated: [{section}] {option} = {value}")
            self._createModel()
            return True
        except configparser.NoSectionError as err:
            print(f"[x] ERROR: {err}")
            return False

    def getConfigValue(self, section, option):
        try:
            value = self._config.get(section, option)
            return value
        except configparser.NoSectionError as err:
            print(f"[x] ERROR: {err}")
            return None

    def _loadConfig(self, wipe=False):
        self._prediction_details = int(
            self._config['DEBUG']['prediction_details'])
        self._verbose = int(self._config['DEBUG']['verbose'])

        self._n_neighbors = int(self._config['KNN']['k'])
        self._weights = self._config['KNN']['w']
        self._power = int(self._config['KNN']['q'])
        self._n_jobs = int(self._config['KNN']['n_jobs'])

        self._test_size = float(self._config['DATA']['test_size'])
        self._random_state = None if (self._config['DATA']['random_state'] == "None") else int(
            self._config['DATA']['random_state'])

        self._snippet_size = int(self._config['POLYFIT']['snippet_size'])
        self._poly_degree = int(self._config['POLYFIT']['poly_degree'])

        if wipe is True:
            self.databases = [db for db in self._config['DATABASES']]
            self.databases_loaded = []

            self.emotions = [emotion for emotion in self._config['EMOTIONS']]

            self.data = []
            self.labels = []
            self.x_train = []
            self.y_train = []
            self.x_test = []
            self.y_test = []

            self.x_test2 = []
            self.y_test2 = []

            self.test_hash = None
            self.train_hash = None

    def _evaluatePrediction(self):
        emotion = self.emotions
        for x in range(0, 2):
            prediction = self.predictions[x]
            prediction_prob = self.predictions_prob[x]
            y_test_emotion = []
            y_test_gender = []
            if x == 0:
                for array in self.y_test:
                    y_test_emotion.append(array[0])
                for array in self.y_test:
                    y_test_gender.append(array[1])
                self.x_test_n = len(self.x_test)
            else:
                for array in self.y_test2:
                    y_test_emotion.append(array[0])
                for array in self.y_test2:
                    y_test_gender.append(array[1])
                self.x_test_n = len(self.x_test2)
            self.accuracy_topf.append(np.zeros((2, 7, 3)))
            for i in range(0, len(prediction[0])):
                if y_test_emotion[i] == prediction[0][i]:
                    result = "TRUE"
                    self.accuracy_topf[x][y_test_gender[i]][y_test_emotion[i]-1][0] += 1
                    self.accuracy_topf[x][y_test_gender[i]][y_test_emotion[i]-1][1] += 1
                else:
                    result = "FALSE"
                    self.accuracy_topf[x][y_test_gender[i]][y_test_emotion[i]-1][1] += 1

                if self._prediction_details:
                    print("[+] [{:5s}]\nprediction={} key={}\nprob={}\n\t   {}\n---\n".format(result,
                                                                                              emotion[prediction[i] -
                                                                                                      1].upper(),
                                                                                              emotion[y_test_emotion[i] -
                                                                                                      1].upper(),
                                                                                              np.array_repr(
                                                                                                  prediction_prob[i]).replace('\n', ''),
                                                                                              emotion))
            for gender in self.accuracy_topf[x]:
                for emotion in gender:
                    if emotion[1] != 0:
                        emotion[2] = emotion[0]/emotion[1]
                    else:
                        emotion[2] = -1

            # TODO advanced evaluation which emotions are classified wrong the most
            # # accuracy for male / female
            # # accuracy for each emotion

    def _splitData(self):
        if self._test_size == 0.0:
            print("[-] Using entire database for training, no test data available; use predict(f='audio.wav') to pass a file")
            self.x_train, self.y_train = self.data, self.labels
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.data, self.labels, test_size=self._test_size, random_state=self._random_state, shuffle=True)

    def _getAudioAttributes(self, f):
        # TODO error handling for reading single files
        data = []
        s1 = AudioSegment.from_file(f, format="wav")
        s2 = audiosegment.from_file(f)
        freqChange, freqAvg, freqMax = self._getFrequency(s2)
        loudPoly, dBFS, maxDBFS = self._getLoudness(s1)
        data.append(np.append(np.concatenate((self._getDerivative(
                    loudPoly),  self._getDerivative(freqChange))), [dBFS, maxDBFS, freqAvg, freqMax]))
        return data

    def _getLoudness(self, s):
        xdata = []
        ydata = []
        chunks = make_chunks(s, self._snippet_size)
        for i in range(0, len(chunks)):
            if chunks[i].dBFS != -np.inf:
                xdata.append((i+1)*self._snippet_size)
                ydata.append(chunks[i].dBFS)
        p = np.polyfit(np.array(xdata), np.array(ydata), self._poly_degree)
        return p, s.dBFS, s.max_dBFS

    def _getFrequency(self, s):
        xdata = []
        ydata = []
        chunks = make_chunks(s, self._snippet_size)
        for i in range(0, len(chunks)):
            void, hist_vals = chunks[i].fft()
            del void
            hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)
            avg = np.mean(hist_vals_real_normed)
            xdata.append((i+1)*self._snippet_size)
            ydata.append(avg)
        p = np.polyfit(np.array(xdata), np.array(ydata), self._poly_degree)
        return p, np.mean(np.array(ydata)), np.amax(np.array(ydata))

    def _getDerivative(self, polynom):
        if len(polynom) == 1:
            return [0]
        deriv_poly = [polynom[i] * (len(polynom)-i-1)
                      for i in range(0, len(polynom)-1)]
        return deriv_poly

    def _parseName(self, name):
        emotion = self.emotions
        actor = ["male", "female"]
        name = name.strip(".wav")
        # doing this twice because some f'd the toronto dataset
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

    def _createModel(self):
        self.model = KNeighborsClassifier(
            n_neighbors=self._n_neighbors, weights=self._weights, p=self._power, n_jobs=self._n_jobs)
        self.trained = False
        self.accuracy = False
        print(f"[+] Model created {self.model}")

    def _loadConfigFile(self, config):
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfile = cfg.read(config)
        # TODO check for valid config and database files
        if not cfile:
            print("[x] ERROR: Failed loading config")
            sys.exit()
        return cfg
