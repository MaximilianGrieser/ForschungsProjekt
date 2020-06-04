import numpy as np
import matplotlib.pyplot as plt

from blackbox import BlackBox


def main():
    blackbox = BlackBox(config="repack/config.ini")
    blackbox.loadData(database="Toronto-DB")
    blackbox.loadData(database="RAVDESS-DB")

    ###
    # Config Parameter test
    ###
    xdata = []
    ydata = []
    for n in range(1, 51):
        blackbox.updateConfig("KNN", "k", n, wipe=False)
        blackbox.train()
        blackbox.predict()
        xdata.append(n)
        ydata.append(blackbox.accuracy)
        print(f"k={n}\t\t", end="\r")
    plt.plot(xdata, ydata, "b")
    plt.ylabel("\naccuracy")
    plt.xlabel("k neighbors")
    plt.show()

    ###
    # Accuracy average test
    ###
    #accuracy = []
    #f = open("testruns.log", "a")
    # for i in range(1, 1001):
    #    blackbox.train()
    #    blackbox.predict()
    #    f.write("{:5s} {:3.5f} {}\n".format(
    #        str(i), blackbox.accuracy, blackbox.databases_loaded))
    #    print(i, end="\r")
    #    accuracy.append(blackbox.accuracy)
    # print("\n")
    # print(np.mean(accuracy))
    # f.write(f"### AVG {np.mean(accuracy)} ### \n")
    # f.close()


if __name__ == "__main__":
    main()
