__copyright__ = \
    """
    Copyright (C) 2022 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the CC BY-NC-SA-4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/). 
    It is to be used for academic research purposes only, no commercial use is permitted.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 29, 2023
    """
__author__ = "Alexandre Delplanque"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.2.0"


import torch
import matplotlib.pyplot as plt 
import random
import itertools
from typing import Optional
from torchvision.transforms import ToPILImage
from ..data.transforms import UnNormalize, GaussianMap

__all__ = ['PlotPrecisionRecall']

class PlotPrecisionRecall:

    def __init__(
        self,
        figsize: tuple = (7,7), 
        legend: bool = False, 
        seed: int = 1
        ) -> None:
        
        self.figsize = figsize
        self.legend = legend
        self.seed = seed

        self._data = []
        self._labels = []

    def feed(self, recalls: list, precisions: list, label: Optional[str] = None) -> None:
        # recalls.append(recalls[-1])
        # precisions.append(0)
        self._data.append((recalls, precisions))
        self._labels.append(label)
    
    def plot(self) -> None:
        
        random.seed(self.seed)
        colors = self._gen_colors(len(self._data))
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(0,1.02)
        ax.set_ylim(0,1.02)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        markers = self._markers
        for i, (recall, precision) in enumerate(self._data):
            ax.plot(recall, precision,
                color=colors[i],
                marker=next(markers),
                markevery=0.1,
                alpha=0.7,
                label=self._labels[i])
        
        if self.legend:
            lg = plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
        
        self.fig = fig
    
    def save(self, path: str) -> None:
        if 'fig' not in self.__dict__:
            self.plot()

        self.fig.savefig(path, dpi=300, format='png', bbox_inches='tight')

    def _gen_colors(self, n: int) -> list:

        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            for i in range(n)]

        return colors
    
    @property
    def _markers(self) -> itertools.cycle:
        return itertools.cycle(('^','o','s','x','D','v','>'))
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from sklearn.metrics import f1_score

class PlotF1ScoreCurve:

    def __init__(self, figsize: tuple = (7, 7), legend: bool = False, seed: int = 1) -> None:
        self.figsize = figsize
        self.legend = legend
        self.seed = seed
        self._data = []
        self._labels = []

    def feed(self, model_output, ground_truth, label: str = None) -> None:
        thresholds = np.linspace(0.01, 0.99, 100)
        f1_scores = []
        probabilities = torch.sigmoid(model_output).detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()

        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            f1 = f1_score(ground_truth, predictions)
            f1_scores.append(f1)

        self._data.append((thresholds, f1_scores))
        self._labels.append(label)

    def plot(self) -> None:
        random.seed(self.seed)
        colors = self._gen_colors(len(self._data))

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')

        markers = self._markers()
        for i, (thresholds, f1_scores) in enumerate(self._data):
            ax.plot(thresholds, f1_scores,
                    color=colors[i],
                    marker=next(markers),
                    markevery=0.1,
                    alpha=0.7,
                    label=self._labels[i])
        
        if self.legend:
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        plt.show()

    def save(self, path: str) -> None:
        if 'fig' not in self.__dict__:
            self.plot()

        self.fig.savefig(path, dpi=300, format='png', bbox_inches='tight')

    def _gen_colors(self, n: int) -> list:
        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                  for i in range(n)]
        return colors
    
    def _markers(self) -> itertools.cycle:
        return itertools.cycle(('^', 'o', 's', 'x', 'D', 'v', '>'))
##############################################################################
class PlotTradeOff:
    def __init__(self, figsize=(10, 6), legend=True):
        self.figsize = figsize
        self.legend = legend
        self.data = []

    def feed(self, beta, gamma_values, scores):
        self.data.append((beta, gamma_values, scores))

    def plot(self):
        plt.figure(figsize=self.figsize)
        for beta, gamma_values, scores in self.data:
            plt.plot(gamma_values, scores, marker='o', label=f'beta={beta}')
        plt.xlabel('Gamma')
        plt.ylabel('F1 Score')
        plt.title('Trade-Off between Beta, Gamma and F1 Score')
        if self.legend:
            plt.legend()
        self.fig = plt.gcf()  # Get current figure

    def save(self, path):
        if not hasattr(self, 'fig'):
            self.plot()
        self.fig.savefig(path, format='pdf', bbox_inches='tight')
        plt.close(self.fig)