import numpy as np
import matplotlib.pyplot as plt

from blackbox import BlackBox


def main():
    blackbox = BlackBox(config="repack/config.ini")
    #blackbox.clearCache()
    blackbox.loadData()

    blackbox.train()
    blackbox.predict()
    #blackbox.updateConfig("POLYFIT", "poly_degree", 15, wipe=False)
    print(f"[+] {blackbox.accuracy}, {blackbox.test_hash}, {blackbox.train_hash}")
    print(blackbox.accuracy_topf)

    ###
    # Config Parameter test
    ###
    # xdata = []
    # ydata = []
    # for n in range(1, 51):
    #    blackbox.updateConfig("KNN", "k", n, wipe=False)
    #    blackbox.train()
    #    blackbox.predict()
    #    xdata.append(n)
    #    ydata.append(blackbox.accuracy)
    #    print(f"Parameter {n}\t\t", end="\r")
    # plt.plot(xdata, ydata, "b")
    # plt.ylabel("\naccuracy")
    # plt.xlabel("k")
    # plt.show()

    ###
    # Accuracy average test
    ###
    #accuracy = []
    #f = open("test.log", "a")
    #for i in range(1, 101):
    #   blackbox.train()
    #   blackbox.predict()
    #   f.write("{:5s} {} {} {}\n".format(
    #       str(i), blackbox.test_hash, blackbox.accuracy, blackbox.databases_loaded))
    #   print(i, end="\r")
    #   accuracy.append(blackbox.accuracy)
    #print("\n")
    #print(np.mean(accuracy))
    #f.write(f"### AVG {np.mean(accuracy)} ### \n")
    #f.close()


if __name__ == "__main__":
    main()
