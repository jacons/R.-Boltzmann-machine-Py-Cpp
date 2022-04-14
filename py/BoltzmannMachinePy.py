import numpy as np
import random as rn

from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from BoltzmannMachine import BoltzmannMachine


class BoltzmannMachinePY(BoltzmannMachine):

    def __init__(self, vNode: int, hNode: int, lRate: float, bSize: int):
        """
         Constructor of BoltzmannMachine written in python
        :param vNode: Number of visible nodes
        :param hNode: Number of hidden nodes
        :param lRate: Learning rate
        :param bSize: Batch size (minibatch approach)
        """
        super().__init__(vNode, hNode)

        self.__lr = lRate
        self.__bSize = bSize

        self.__dataset = None

        self.__biasV = None  # Bias of visible node
        self.__biasH = None  # Bias of hidden node

        self.__deltaW = None  # delta of weights
        self.__deltaV = None  # delta of bias of visible node
        self.__deltaH = None  # delta of bias of hidden node

        # define activation function mapping
        self.__sigmoid = np.vectorize(self.__sig)
        return

    def init(self):
        """
        Initializing the matrices and biases weights randomly
        :return:
        """
        self._weights = np.random.uniform(0, 1, (self._hNode, self._vNode))
        self.__biasV = np.random.uniform(0, 1, self._vNode)
        self.__biasH = np.random.uniform(0, 1, self._hNode)

        self.__deltaW = np.zeros((self._hNode, self._vNode))
        self.__deltaV = np.zeros(self._vNode)
        self.__deltaH = np.zeros(self._hNode)

        return

    def fit(self, dataset: ndarray, epochs: int):
        """
        Fitting phase, scann entire dataset for number of epochs and update
        the weights applying the minibatch approach
        :param dataset: original dataset
        :param epochs: number of time that we must scan the dataset
        :return:
        """
        # get the dataset using for learning and testing
        self.__dataset = dataset

        # counter used for keep track when we should upgrade the deltas
        count = 0

        # Foreach epochs
        for e in range(0, epochs):
            print("epoch #" + str(e))
            # Foreach element in the dataset
            for x in self.__dataset:

                # Update deltas every bSize steps
                if count == self.__bSize:
                    self.__updateW(count)
                    count = 0

                # Applying the learning phase
                self.__cd0(x[1:])
                count += 1
            # Flushing deltas, for the next dataset iteration
            self.__updateW(count)
        return

    @staticmethod
    def __sig(x: ndarray):
        """
        Sigmoid activation function applied to vector
        :param x: vector
        :return: mapped vector (nodes activated)
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __norm(x):
        """
        Normalize vectors, transform the range between 0-255 to (0 or 1) applying threshold function
        :param x: vector
        :return: normalized vector
        """
        return np.vectorize(lambda z: 1 if z > 128 else 0)(x)

    @staticmethod
    def __sampling(x: ndarray):
        """
        Stochastic sampling, applied to vector
        :param x: vector
        :return: sampled vector
        """
        return np.vectorize(lambda z: 1 if z > rn.uniform(0, 1) else 0)(x)

    def __cd0(self, v0: ndarray):
        """
        Contrastive Divergence algorithm
        :param v0: Visible nodes a time 0 (input data)
        :return:
        """
        # Normalizing to binary vector
        v0 = self.__norm(v0)
        # Perform the probability of hidden state a time 0
        h0_prob = self.__sigmoid(np.dot(self._weights, v0) + self.__biasH)
        # Sampling the hidden state a time 0
        h0_smpl = self.__sampling(h0_prob)
        # Perform the probability of visible state a time 1 (Gibbs-Sampling)
        v1_prob = self.__sigmoid(np.dot(self._weights.T, h0_smpl) + self.__biasV)
        # Sampling the visible state a time 1
        v1_s = self.__sampling(v1_prob)
        # Perform the probability of hidden state a time 1
        h1_prob = self.__sigmoid(np.dot(self._weights, v1_s.T) + self.__biasH)
        wake = np.outer(h0_prob, v0)
        dream = np.outer(h1_prob, v1_s)
        # updating deltas
        self.__deltaW += wake - dream
        self.__deltaV += v0 - v1_prob
        self.__deltaH += h0_prob - h1_prob
        return

    def __updateW(self, elms):
        """
        Updating the weights from "deltas" accumulated
        :param elms: number of time that we accumulated the deltas
        :return:
        """
        if elms != 0:
            self._weights += (self.__deltaW / elms) * self.__lr
            self.__biasH += (self.__deltaH / elms) * self.__lr
            self.__biasV += (self.__deltaV / elms) * self.__lr
            self.__deltaH.fill(0)
            self.__deltaV.fill(0)
            self.__deltaW.fill(0)
        return

    def __inference(self, v0):
        """
        The core of RBM is the ability to apply the representation of learning,
        indeed it has learnt by unlabelled examples how to transform the huge visible state
        in a smaller representation. (Feature reduction).
        :param: input data (visible nodes)
        :return: hidden state (hidden nodes)
        """
        # Normalizing to binary vector
        v0 = self.__norm(v0)
        res_prob = self.__sigmoid(np.dot(self._weights, v0) + self.__biasH)
        # Sampling
        res_smpl = self.__sampling(res_prob)
        return res_smpl

    def reconstructDS(self):
        """
        The reconstructing phase creates a new dataset with less feature respect to
        the previous one, foreach element in the old dataset, perform the "hidden state"
        and put it in the new dataset
        :return: new dataset
        """
        # define a Restricted dataset from the original one
        newDs = np.zeros((self.__dataset.shape[0], self._hNode))
        i = 0
        # Foreach element into dataset we perform an inference, it returns a vector with only H nodes
        # Oss. start from 1 because the first col il a label
        for x in self.__dataset:
            newDs[i] = self.__inference(x[1:])
            i += 1
        return newDs

    def logisticRegression(self, newDs: ndarray):
        """
        Given a new dataset, perform a Logistic regression.
        :param newDs: now dataset
        :return:
        """
        # Number of examples used for training phase in the logistic regression
        tr_set = int(self.__dataset.shape[0] * self._tv_vl)
        # define a Logistic Regressor
        lReg = LogisticRegression(random_state=0, max_iter=500)
        # fix the reconstructed dataset
        lReg.fit(newDs[0:tr_set], self.__dataset[0:tr_set, 0])
        # retrieve the accuracy archived
        acc = lReg.score(newDs[tr_set:], self.__dataset[tr_set:, 0]) * 100
        print("Accuracy achieved : "+str(acc)+"%")
