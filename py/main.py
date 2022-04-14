import time
import math
import pandas as pd
from matplotlib import pyplot as plt

from BoltzmannMachineCpp import BoltzmannMachineCpp
from BoltzmannMachinePy import BoltzmannMachinePY


def showDigits(datas):
    fig, axs = plt.subplots(2, 2)
    rng = range(0, 2)

    c = 154
    for i in rng:
        for j in rng:
            axs[i, j].axis('off')
            axs[i, j].imshow((datas[c][1:].reshape(28, 28)), cmap="gray")
            c += 1
    plt.show()


def truncate(number, digits) -> int:
    stepper = 10.0 ** digits
    return int(math.trunc(stepper * number) / stepper)


if __name__ == '__main__':

    print("Starting a time:"+str(time.time()))
    # ---------------------------  LOADING PHASE ---------------------------
    start_time = time.time()
    initial_time = start_time
    print("Loading dataset")

    dataset = pd.read_csv("src/dataset.csv")
    ds = dataset.to_numpy()

    elapsed = str(truncate((time.time() - start_time) * 1000, 4))
    print("Loading complete(it takes :" + elapsed + " ms)")
    # ---------------------------  LOADING PHASE ---------------------------

    # ---------------------------  LEARNING PHASE ---------------------------
    start_time = time.time()
    print("Learning phase")

    rbm = BoltzmannMachinePY(vNode=784, hNode=81, lRate=1, bSize=64)
    rbm.init()
    rbm.fit(dataset=ds, epochs=2)

    elapsed = str(truncate((time.time() - start_time), 4))
    print("Learning complete(it takes :" + elapsed + " s)")
    # ---------------------------  LEARNING PHASE ---------------------------

    # ---------------------------  LOGISTIC REGRESSION PHASE ---------------------------
    start_time = time.time()
    print("Reconstructing phase")

    newDs = rbm.reconstructDS()

    elapsed = str(truncate((time.time() - start_time), 4))
    print("Reconstructing complete(it takes :" + elapsed + " s)")
    # ---------------------------  LOGISTIC REGRESSION PHASE ---------------------------

    # ---------------------------  LOGISTIC REGRESSION PHASE ---------------------------
    start_time = time.time()
    print("Logistic Regression phase")

    rbm.logisticRegression(newDs=newDs)

    elapsed = str(truncate((time.time() - start_time), 4))
    print("Logistic Regression complete(it takes :" + elapsed + " s)")
    # ---------------------------  LOGISTIC REGRESSION PHASE ---------------------------

    elapsed = str(truncate((time.time() - initial_time), 4))
    print("Total execution in : " + elapsed + " s")

    """
    rbm = BoltzmannMachineCpp(vNode=784, hNode=81, labels=dataset['label'].to_numpy())
    filters = pd.read_csv("src/weights.csv", header=None)
    # rbm.importFilter(weights=filters.to_numpy())
    # rbm.plotFilter()
    restrictedDs = pd.read_csv("src/restrictedDataset.csv",header=None)
    rbm.findAccuracy(dataset=restrictedDs.to_numpy())
    """
