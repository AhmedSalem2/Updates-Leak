# custom weights initialization called on netG and netD
import torch
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        #torch.nn.init.xavier_uniform(m.weight)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        #torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)


def load_dataSingle(data_name):
    with np.load( data_name,allow_pickle=True) as f:
        train_x = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x


#Loading the posterior difference for the different updating sets.    
def getDifference(name,numOfPoints,sizeOfUpdateSet,numOfFiles=1):
    tempHolder = []
    for i in range(1,numOfFiles+1):
        path='../models output/updating'+sizeOfUpdateSet+'Points/'+name+'OutputDifferences'+str(i)+'.0.npz'
        diffTemp = load_dataSingle(path)[0]
        for x in diffTemp:
            tempHolder.append(torch.from_numpy(np.concatenate(x)[:numOfPoints].flatten()))
    return (torch.stack(tempHolder))
