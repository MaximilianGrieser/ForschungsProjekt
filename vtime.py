import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks

SIZE = 30
TYPE = "dBFS"


def split_sound(s, i):
    ssound = s[::i]
    return ssound


def loudness(s, type=TYPE):
    if type == "dBFS":
        value = s.dBFS
    elif type == "max_dBFS":
        value = s.max_dBFS
    elif type == "rms":
        value = s.rms
    elif type == "max":
        value = s.max
    return value


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


def analyze():
    fig, axs = plt.subplots(2, 4)
    subplots = [axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1],
                axs[0, 2], axs[1, 2], axs[0, 3], axs[1, 3]]
    axs[0, 0].set_title('Axis [0,0]')
    axs[1, 0].set_title('Axis [1,0]')
    axs[0, 1].set_title('Axis [0,1]')
    axs[1, 1].set_title('Axis [1,1]')
    axs[0, 2].set_title('Axis [0,2]')
    axs[1, 2].set_title('Axis [1,2]')
    axs[0, 3].set_title('Axis [0,3]')
    axs[1, 3].set_title('Axis [1,3]')
    for ax in axs.flat:
        ax.set(xlabel='time (ms)', ylabel=TYPE)
    dataset = ["neutral", "calm", "happy", "sad",
               "angry", "fearful", "disgust", "suprised"]
    for x in range(0, len(subplots)):
        print(x)
        subplots[x].set_title(dataset[x])
        subplots[x].set_xlim([0, 5000])
        subplots[x].set_ylim([-90, -10])
    red = mpatches.Patch(color='red', label='female')
    blue = mpatches.Patch(color='blue', label='male')
    fig.legend(handles=[red, blue])

    counter = 0
    colors = ["b", "r"]
    for p in os.listdir("audio"):
        if not os.path.isfile(p):
            for f in os.listdir("audio/"+p):
                s = AudioSegment.from_file(
                    "audio/{}/{}".format(p, f), format="wav")
                xdata = []
                ydata = []
                fdata = parse_fname(f)
                fsplit = make_chunks(s, SIZE)
                for i in range(0, len(fsplit)):
                    lval = loudness(fsplit[i])
                    xdata.append((i+1)*SIZE)
                    ydata.append(lval)
                subplots[fdata.get("emotion_n")-1].plot(xdata,
                                                        ydata, c=colors[fdata.get("actor")], alpha=0.5)
                print(xdata)
                print(ydata)
                counter += 1
                
        
    title = "sample using pydub {} and the RAVDESS dataset, n = {}, intervall = {}ms".format(TYPE, counter, SIZE)
    fig.canvas.set_window_title(title)

    plt.show()


if __name__ == "__main__":
    analyze()
