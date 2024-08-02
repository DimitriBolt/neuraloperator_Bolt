import os
import pandas
import matplotlib.pyplot as plt

training_loss_1 = "H1_2DLoss"
n_layers =  4
epoch = 200 - 1
err_1 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_1 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))

training_loss_2 = "L2_2Dloss"
err_2 = pandas.read_pickle(os.path.join(os.path.expanduser('~'), 'Documents', training_loss_2 + "_l" + str(n_layers) + "_e" + str(epoch + 1) + '.pkl'))


fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(err_1.index.astype(int),
        err_1["train_err"],
        label=training_loss_1 + ' for train, n_layers =' + str(n_layers),
        color='green',
        linestyle='dashed')
# ax.plot(epoch_list, train_err_list_l2_h1, label='H1 as reference (L2 for train)', color='green')

ax.plot(err_2.index.astype(int),
        err_2["train_err"],
        label=training_loss_2 + ' for train, n_layers =' + str(n_layers),
        color='red')
# ax.plot(epoch_list, train_err_list_h1_l2, label='L2 as reference (H1 for train)', linestyle='dashed', color='red')


ax.legend(loc='upper right')
ax.grid(which='major', alpha=0.5)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.set_ylabel('Loss')
plt.yscale(value="log")
ax.set_title('Losses during training. \nTrain resolution = 16x16[const], test_resolutions=[16, 32]')
ax.set_xlabel('Epoch')
plt.grid()
plt.show()


pass