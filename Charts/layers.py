import os

import matplotlib.pyplot as plt
import pandas
from matplotlib import ticker

training_loss_1 = "H1_2DLoss"
training_loss_2 = "L2_2Dloss"

epoch = 200 - 1

n_layers = 1
layer1h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer1l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 2
layer2h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer2l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 3
layer3h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer3l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 4
layer4h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer4l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 5
layer5h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer5l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 6
layer6h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer6l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 7
layer7h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer7l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 8
layer8h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer8l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 9
layer9h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer9l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 10
layer10h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer10l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

n_layers = 11
layer11h1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))
layer11l2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

layer_h1 = [layer1h1["avg_loss"][epoch],
            layer2h1["avg_loss"][epoch],
            layer3h1["avg_loss"][epoch],
            layer4h1["avg_loss"][epoch],
            layer5h1["avg_loss"][epoch],
            layer6h1["avg_loss"][epoch],
            layer7h1["avg_loss"][epoch],
            layer8h1["avg_loss"][epoch],
            layer9h1["avg_loss"][epoch],
            layer10h1["avg_loss"][epoch],
            layer11h1["avg_loss"][epoch]
            ]

layer_l2 = [layer1l2["avg_loss"][epoch],
            layer2l2["avg_loss"][epoch],
            layer3l2["avg_loss"][epoch],
            layer4l2["avg_loss"][epoch],
            layer5l2["avg_loss"][epoch],
            layer6l2["avg_loss"][epoch],
            layer7l2["avg_loss"][epoch],
            layer8l2["avg_loss"][epoch],
            layer9l2["avg_loss"][epoch],
            layer10l2["avg_loss"][epoch],
            layer11l2["avg_loss"][epoch]
            ]

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        layer_h1,
        label=training_loss_1 + ' for train',
        color='green',
        linestyle='dashed')

ax.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        layer_l2,
        label=training_loss_2 + ' for train',
        color='red')

ax.legend(loc='upper right')
ax.grid(which='major', alpha=0.5)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.set_ylabel('Loss')
plt.yscale(value="log")
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
ax.set_title(f'Losses during training. epoch = {epoch + 1}\nTrain resolution = 16x16[const], test_resolutions=[16, 32]')
ax.set_xlabel('Layers')
plt.grid()
plt.show()

pass

pass
