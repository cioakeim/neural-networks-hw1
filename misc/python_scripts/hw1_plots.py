import numpy as np
import matplotlib.pyplot as plt
import glob


root="../../data/saved_networks/adam/network_"

exp_list={
    "8":"SGD",
    "19":"Adam"
}

plt.figure(0)
for idx,label in exp_list.items():
    name=root+idx+"/epoch_accuracy_log.csv"
    data=np.loadtxt(name,delimiter=",",skiprows=1)
    plt.plot(data[:,0],data[:,3],label=label)

plt.legend()
plt.title("Adam testing")
plt.xlabel("Training epoch")
plt.ylabel("Accuracy")
plt.show()
