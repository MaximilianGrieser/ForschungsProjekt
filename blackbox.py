import configparser
import os
import sys

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

        self.predictions = []
        self.predictions_prob = []

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
                    self.data.append(np.append(np.concatenate((self._getDerivative(
                        loudPoly),  self._getDerivative(freqChange))), [dBFS, maxDBFS, freqAvg, freqMax]))
                    self.labels.append(fdata.get("emotion_n"))
                    cache_data.append(np.append(np.concatenate((self._getDerivative(
                        loudPoly),  self._getDerivative(freqChange))), [dBFS, maxDBFS, freqAvg, freqMax]))
                    cache_labels.append(fdata.get("emotion_n"))
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
        if len(self.databases_loaded) > 0:
            self._splitData()
            self.model.fit(self.x_train, self.y_train)
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
                self.predictions = self.model.predict(self.x_test)
                self.predictions_prob = self.model.predict_proba(self.x_test)
                self.accuracy = self.model.score(self.x_test, self.y_test)
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
            if section=="POLYFIT":
                print(f"[-] Updating [POLYFIT] parameters invalidate cache files. It is strongly recommended to 'clearCache()'")
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

    def _evaluatePrediction(self):
        emotion = self.emotions
        prediction = self.predictions
        prediction_prob = self.predictions_prob
        y_test = self.y_test
        self.x_test_n = len(self.x_test)
        for i in range(0, len(prediction)):
            if y_test[i] == prediction[i]:
                result = "TRUE"

            else:
                result = "FALSE"
            if self._prediction_details:
                print("[+] [{:5s}]\nprediction={} key={}\nprob={}\n\t   {}\n---\n".format(result,
                                                                                          emotion[prediction[i] -
                                                                                                  1].upper(),
                                                                                          emotion[y_test[i] -
                                                                                                  1].upper(),
                                                                                          np.array_repr(
                                                                                              prediction_prob[i]).replace('\n', ''),
                                                                                          emotion))
        # TODO advanced evaluation which emotions are classified wrong the most

    def _splitData(self):
        if self._test_size == 0.0:
            print("[-] Using entire database for training, no test data available; use predict(f='audio.wav') to pass a file")
            self.x_train, self.y_train = self.data, self.labels
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.data, self.labels, test_size=self._test_size, random_state=self._random_state)

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
