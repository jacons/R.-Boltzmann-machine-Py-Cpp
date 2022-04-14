import numpy as np
from matplotlib import pyplot as plt


class BoltzmannMachine:

    def __init__(self, vNode: int, hNode: int):
        """
        :param vNode: Number of visible units
        :param hNode: Number of hidden units
        """
        self._vNode = vNode  # Visible nodes
        self._hNode = hNode  # Hidden nodes
        self._weights = None  # Matrix of weights

        self._tv_vl = 0.8  # Training and test set split

    def plotFilter(self):
        """
            Plotting the filters
        :return:
        """

        # Number of filter for each row/cols
        t = int(np.sqrt(self._hNode))

        side = range(0, t)
        count = 0
        fig, axs = plt.subplots(t, t, figsize=(20, 20))

        for i in side:
            for j in side:
                axs[i, j].axis('off')
                axs[i, j].imshow(self._weights[count, :].reshape(28, 28), cmap="gray")
                count += 1

        plt.show()
