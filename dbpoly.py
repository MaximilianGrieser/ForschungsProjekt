from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import matplotlib.pyplot as plt

FILE = "test.wav"
SIZE = 10
DEGREE = 10


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    s = AudioSegment.from_file(FILE, format="wav")
    xdata = []
    ydata = []
    chunks = make_chunks(s, SIZE)
    for i in range(0, len(chunks)):
        if chunks[i].dBFS != -np.inf:
            xdata.append((i+1)*SIZE)
            ydata.append(chunks[i].dBFS)
    p = np.polyfit(np.array(xdata), np.array(ydata), DEGREE)
    f = np.poly1d(p)

    plt.plot(np.array(xdata), f(np.array(xdata)),
             'r', label="Polyfit n="+str(DEGREE))
    plt.plot(np.array(xdata), np.array(ydata), 'bo', label="Data")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.show()

    # return p
    # print(p)

if __name__ == "__main__":
    main()
