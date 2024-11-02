import numpy as np 
import os
from matplotlib import pyplot as plt 




def main():
    folder_path="../data/csv_logs/"
    filenames=["nc_log","nn_1_log","nn_3_log"]
    training_filename="training_log"

    # plot training data 
    data=np.genfromtxt(folder_path+training_filename+".csv",delimiter=",",skip_header=0);
    print(data[:,0])
    print(data[:,1])
    plt.figure(0)
    plt.plot(data[:,0],data[:,1],marker="o")
    plt.title("Training time")
    plt.xlabel("Training set size")
    plt.ylabel("Training time (sec)")

    plt.figure(1) # Times
    plt.figure(2) # Success rates
    for file in filenames:
        data=np.genfromtxt(folder_path+file+".csv",delimiter=",",skip_header=0)
        plt.figure(1)
        plt.plot(data[:,0],data[:,1],label=file,marker="o")
        plt.figure(2)
        plt.plot(data[:,0],data[:,2],label=file,marker="o")

    plt.figure(1)
    plt.xlabel("Training set size")
    plt.ylabel("Total testing time (sec)")
    plt.figure(2)
    plt.xlabel("Training set size")
    plt.ylabel("Success rate (%)")

    plt.show()

if __name__=="__main__":
    main();
