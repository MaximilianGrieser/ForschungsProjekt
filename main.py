import numpy as np

from blackbox import BlackBox


def main():
    blackbox = BlackBox(config="repack/config.ini")
    blackbox.loadData(database="Toronto-DB")
    blackbox.loadData(database="RAVDESS-DB")
    accuracy = []
    f = open("testruns.log", "a")
    for i in range(1, 1001):
        blackbox.train()
        blackbox.predict()
        f.write("{:5s} {:3.5f} {}\n".format(
            str(i), blackbox.accuracy, blackbox.databases_loaded))
        print(i, end="\r")
        accuracy.append(blackbox.accuracy)
    print("\n")
    print(np.mean(accuracy))
    f.write(f"### AVG {np.mean(accuracy)} ### \n")
    f.close()


if __name__ == "__main__":
    main()
