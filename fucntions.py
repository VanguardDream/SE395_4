import numpy as np

def onehot(Y):
    output = []

    for i in range(len(Y)):
        if Y[i] == 0:
            tmp = np.array([1,0,0,0,0])
        elif Y[i] == 1:
            tmp = np.array([0,1,0,0,0])
        elif Y[i] == 2:
            tmp = np.array([0,0,1,0,0])
        elif Y[i] == 3:
            tmp = np.array([0,0,0,1,0])
        else:
            tmp = np.array([0,0,0,0,1])
        output.append(tmp)

    return np.array(output)

def accuracy(Y,Y_hat,total):
    count = 0
    if(len(Y) == len(Y_hat)):
        for i in range(len(Y)):
            if(Y[i] == Y_hat[i]):
                count = count + 1
        return count / total.shape[0]
    
    else:
        return -1


        