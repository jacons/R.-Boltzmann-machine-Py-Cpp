from numpy import ndarray
from sklearn.linear_model import LogisticRegression

from BoltzmannMachine import BoltzmannMachine


class BoltzmannMachineCpp(BoltzmannMachine):

    def __init__(self, vNode: int, hNode: int, labels: ndarray):
        """
        Constructor of BoltzmannMachine written in c++
        :param vNode: Number of Visible node
        :param hNode: Number of Hidden node
        :param labels: Array of labels (using for last predicting phase)
        """
        super().__init__(vNode, hNode)
        self.__labels = labels

    def importFilter(self, weights: ndarray):
        self._weights = weights

    def findAccuracy(self, dataset: ndarray):

        # get numbers of examples in the dataset
        exs = dataset.shape[0]
        # get the number of example using for train and test set
        tr_set = int(exs*self._tv_vl)

        clf = LogisticRegression(random_state=0, max_iter=500)
        clf = clf.fit(dataset[0:tr_set], self.__labels[0:tr_set])

        acc = clf.score(dataset[tr_set:], self.__labels[tr_set:])*100
        print("Accuracy archived: "+str(acc)+"% ")
