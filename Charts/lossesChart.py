import os

import matplotlib.pyplot as plt
import pandas
from sympy.stats.sampling.sample_numpy import numpy

H1_2DLoss_label = "H1_2DLoss"
n_layers = 4
epoch = 200 - 1
H1_2DLoss_df = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', H1_2DLoss_label + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

L2_2Dloss_label = "L2_2Dloss"
L2_2Dloss_df = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', L2_2Dloss_label + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

fig, ax = plt.subplots(figsize=(11, 8.5))
ax.plot(H1_2DLoss_df.index.astype(int),
        H1_2DLoss_df["avg_loss"],
        label=H1_2DLoss_label + ' for train, n_layers =' + str(n_layers),
        color='red',
        linestyle='solid')

ax.plot(H1_2DLoss_df.index.astype(int),
        H1_2DLoss_df["avg_loss_for_comparison"],
        label=H1_2DLoss_label + ' for comparison, n_layers =' + str(n_layers),
        color='red',
        linestyle='dashed')

ax.plot(L2_2Dloss_df.index.astype(int),
        L2_2Dloss_df["avg_loss"],
        label=L2_2Dloss_label + ' for train, n_layers =' + str(n_layers),
        color='green',
        linestyle='solid')

ax.plot(L2_2Dloss_df.index.astype(int),
        L2_2Dloss_df["avg_loss_for_comparison"],
        label=L2_2Dloss_label + ' for comparison, n_layers =' + str(n_layers),
        color='green',
        linestyle='dashed')


ax.legend(loc='upper right')
ax.grid(which='major', alpha=0.5)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.set_ylabel('Loss')
ax.set_yticks(numpy.arange(0, 30, 1))
plt.yscale(value="log")
ax.set_title('Losses during training. \nTrain resolution = 16x16[const], test_resolutions=[16, 32]')
ax.set_xlabel('Epoch')
plt.grid()
plt.show()

pass
