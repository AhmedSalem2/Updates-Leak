'''
Created on 25 Jan 2019

@author: ahmed.salem
'''
import torch
from torch import nn
from torchvision import transforms
import numpy as np
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from network import labelPredMNIST
from utils import getDifference,load_dataSingle

    

#Loading the images and labels for the different updating sets.    
def getImages(name,sizeOfUpdateSet):
    path='../data indices/updating'+sizeOfUpdateSet+'Points/MNISTdataIndex'+name
    dataroot = './data/'
    imgs=[]
    labels=[]
    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=dataroot, train=True,
                                            download=True, transform=transform)
    indices = load_dataSingle(path+'.npz')  
    indices = indices[0]
    for indx in indices[1:]:
        train_sampler = SubsetRandomSampler(indx)
        trainloaderTemp = torch.utils.data.DataLoader(trainset, batch_size=len(indx),
                                              shuffle=False, num_workers=0,sampler=train_sampler)   
        dataiter = iter(trainloaderTemp)
        img,label =  dataiter.next()
        #Setting the label for our evaluation and training by calculating its normalized distribution 
        label = (torch.bincount(label,minlength=10)).to(dtype=torch.float)/float(sizeOfUpdateSet)
        labels.append(label)
        img = img
        imgs.append(img)
    return imgs,labels



#Creating the dataset for this attack, which consists of the labels and the posterior differences.
def getDataset(name,numOfPoints,sizeOfUpdateSet,numOfFiles):
    #The images are not needed for this attack, thus not loaded here.
    _,labels = getImages(name,sizeOfUpdateSet)
    labels = (torch.stack(labels))
    diff = getDifference(name,numOfPoints,sizeOfUpdateSet,numOfFiles)
    #Creating a dataset of posterior differences and labels
    dataset  = torch.utils.data.TensorDataset(diff,labels)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                              shuffle=True, num_workers=0)
    return dataLoader,diff.size(1)


#Traing the attack model.
def trainModel(model,trainloader,num_epochs=100,learning_rate = 1e-3):
    criterion1 = nn.KLDivLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
    for epoch in range(num_epochs):
        for mini_batch_data,label in trainloader:
            mini_batch_data = mini_batch_data.to(device)
            label = label.to(device)
            # ===================forward=====================
            output = model(mini_batch_data)
            loss = criterion1(output, label)
            #loss = loss1+loss2
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


#Testing the attack model.
def testModel(model,testLoader,sizeOfUpdateSet):
    #Initializing counters and data holders.
    totalCounter = totalAcc = totalMSE = totalKLDiv = totalAcc_basline = totalMSE_basline = totalKLDiv_basline = 0

    #Generating data for the baseline for 10 classes -since MNIST is a 10 class dataset-.
    randomBAseline = np.random.randint(10, size=int(sizeOfUpdateSet))
    randomBAseline = np.bincount(randomBAseline,minlength=10)+0.1
    randomBAseline = randomBAseline/float(randomBAseline.sum())
    baselineTemp = torch.from_numpy((randomBAseline))
    
    #Testing the attack and baseline for the different batches.
    for mini_batch_data,labelTesting in testLoader:
        
        mini_batch_data = mini_batch_data.to(device)
        labelTesting = labelTesting.to(device)        
        baseline = baselineTemp.repeat(labelTesting.size(0),1)
        baseline = baseline.to(device)
        totalCounter = totalCounter+1
        output_log = model(mini_batch_data)
        output = torch.exp(output_log)
        
        #For the single-sample label case
        if(sizeOfUpdateSet=='1'):
            _,outputInd = output.max(1)
            _,labelsInd = labelTesting.max(1)
            correct = (outputInd == labelsInd).float().sum()
            totalAcc = totalAcc + correct/labelTesting.size(0)
        else:
        #For the multi-sample label case
            _,outputInd = output.max(1)
            _,labelsInd = labelTesting.max(1)
            correct = (outputInd == labelsInd).float().sum()
            totalAcc = totalAcc + correct/labelTesting.size(0)
            _,outputInd_basline = baseline.max(1)
            correct_basline = (outputInd_basline == labelsInd).float().sum()
            totalAcc_basline = totalAcc_basline + correct_basline/labelTesting.size(0)
            
        #Calculating the Mean square error(MSE) and the Kullback–Leibler divergence (KL-div) for our attack
        MSE  = torch.sqrt(torch.mean((output - labelTesting).pow(2)))
        klDiv = F.kl_div(output_log, labelTesting)
        totalKLDiv = totalKLDiv + klDiv
        totalMSE = totalMSE + MSE
        
        
        #Calculating the Mean square error(MSE) and the Kullback–Leibler divergence (KL-div) for the  baseline
        baseline = baseline.type(torch.FloatTensor)
        labelTesting = labelTesting.type(torch.FloatTensor)
        MSE_basline  = torch.sqrt(torch.mean((baseline- labelTesting).pow(2)))
        kldiv_basline = F.kl_div(torch.log(baseline), labelTesting)
        totalMSE_basline = totalMSE_basline + MSE_basline
        totalKLDiv_basline = totalKLDiv_basline + kldiv_basline
                
    return round((totalAcc/totalCounter).item(),4), round((totalKLDiv/totalCounter).item(),4),(round((totalAcc_basline/totalCounter).item(),4)),(round((totalKLDiv_basline/totalCounter).item(),4))
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
#The number of points used to update the shadow/training models.       
for sizeOfShadowUpdateSet in ['100']:
    #The number of points used to update the target/testing models.       
    for sizeOfUpdateSet in ['100']:
                
        numOfPoints=100
        #Loading data for the testing models. Not to confuse "shadow" and "target" here, they are just namings. Target is what is used to train the attack model and shadow is what is used to test the attack model.
        testingLoader,sizeOfAttackVector = getDataset('target',numOfPoints,sizeOfShadowUpdateSet,numOfFiles=1)
        
        #Loading data for the training model.
        trainingLoader,_                  = getDataset('shadow',numOfPoints,sizeOfUpdateSet,numOfFiles=10)

        model = labelPredMNIST(sizeOfAttackVector)
        accAllRuns = ''

        accAllRunsSum = 0
        klAllRunsSum = 0
        numOfRuns = 10
        accAllRunsSum_basline =0
        klAllRunsSum_basline =0
        
        #Number of epochs used to train the attack model
        for numOfEp in [50]:
            accAllRunsSum = 0
            klAllRunsSum = 0
            accAllRunsSum_basline =0
            klAllRunsSum_basline =0
            
            #Training different attack models and taking the average.
            for i in range(0,numOfRuns):
                model = labelPredMNIST(sizeOfAttackVector)
                model = model.to(device)
                    
                    
                model = trainModel(model, trainingLoader,num_epochs=numOfEp)
                tempAcc,tempKl,baselineAcc,baslineKL = testModel(model,testingLoader,sizeOfShadowUpdateSet)
                
                #Baseline's calculations
                accAllRunsSum_basline = accAllRunsSum_basline+baselineAcc
                klAllRunsSum_basline = klAllRunsSum_basline+baslineKL
                
                #Our attack's calculations
                accAllRunsSum = accAllRunsSum+tempAcc
                klAllRunsSum = klAllRunsSum+tempKl


        print('Models updated with %s points used to train the attack model to attack models updated with %s points'%(sizeOfUpdateSet,sizeOfShadowUpdateSet))
        print('Avg acc for all runs: ' + str(accAllRunsSum/numOfRuns))
        print('Avg kl for all runs: ' + str(klAllRunsSum/numOfRuns))
        

        print('Baseline')
        print('Avg acc for all runs: ' + str(accAllRunsSum_basline/numOfRuns))
        print('Avg kl for all runs: ' + str(klAllRunsSum_basline/numOfRuns))
