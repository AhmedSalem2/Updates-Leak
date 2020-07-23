import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torch.backends.cudnn as cudnn
import os
import argparse
import sys
from random import randint


 
parser = argparse.ArgumentParser()
parser.add_argument('--sizeOfUpdateSet', type=int, default=100, help='Size of the update set')


opt = parser.parse_args()
folderName = 'updating'+str(opt.sizeOfUpdateSet)+'Points'
    
    
    
    
MNISTFolder = './data/'

def load_data(data_name):
    with np.load( data_name) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y


try:
    os.makedirs('./data indices/'+folderName)
    os.makedirs('./models output/'+folderName)
    os.makedirs('./models/'+folderName)
except OSError:
    print('cannot create folder')
    pass

def load_dataset(trainset,listOfIndex,numOfClusters,nameToStore,split = 10000):
    #splitting the dataset into a training set and updating sets

    indicesLeft = listOfIndex
    trainloaders = []
    listOfIndexToSave = []
    for i in range(0,numOfClusters):
        if i!=0:
            #The updating sets
            split = opt.sizeOfUpdateSet
            batch_size = 64
        else:
            #The training set
            split = 10000
            batch_size = 64
                    
                
        dataI = np.random.choice(indicesLeft, size=split, replace=False)
        if i == 0:
            indicesLeft = list(set(indicesLeft) - set(dataI))
        train_sampler = SubsetRandomSampler(dataI)
        trainloaderTemp = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=0,sampler=train_sampler)
        trainloaders.append(trainloaderTemp)
        listOfIndexToSave.append(dataI)
        
    np.savez_compressed('./data indices/'+folderName+'/MNISTdataIndex'+nameToStore+'.npz', np.array(listOfIndexToSave))
    testset = torchvision.datasets.MNIST(root=MNISTFolder, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0)
    return trainloaders,testloader
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def save_checkpoint(state, d):
    filename='./models/'+folderName+'/'+d+'.pth'
    torch.save(state, filename)  # save checkpoint



def trainModel(trainloaderHolder,optimizer,model,epochsNum=25):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochsNum):  # loop over the dataset multiple times
        trainloader = iter(trainloaderHolder)
        running_loss = 0.0
        for i, (dataHolder, labelsHolder) in enumerate(trainloader, 0):
            # get the inputs
            if use_cuda:
                inputs, labels = dataHolder.to(device), labelsHolder.to(device)
            else:
                inputs, labels = dataHolder, labelsHolder
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
            if i % len(trainloader) == len(trainloader)-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    return model

def updateModel(trainloaderHolder,optimizer,model,epochsNum=1):
    #net = model#copy.deepcopy(model) 
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochsNum):  # loop over the dataset multiple times
        trainloader = iter(trainloaderHolder)
        running_loss = 0.0
        labelsTotal=[]
        for i, (dataHolder, labelsHolder) in enumerate(trainloader, 0):
            # get the inputs
            if use_cuda:
                inputs, labels = dataHolder.to(device), labelsHolder.to(device)
            else:
                inputs, labels = dataHolder, labelsHolder
#                 print(labels)
#                 print(len(labels))
            # zero the parameter gradients
            optimizer.zero_grad()
            print(labels)
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            labelsTotal.append(labels.cpu().detach().numpy())
            running_loss += loss.item()
            
    return model,labelsTotal


def testModel(testloader,model):
    #net = model
    model.eval()
    correct = 0
    total = 0
    outputPoints = []
    with torch.no_grad():
        for (dataHolder, labelsHolder) in testloader:
            if use_cuda:
                images, labels = dataHolder.to(device), labelsHolder.to(device)
            else:
                images, labels = dataHolder, labelsHolder
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            outputPoints.append(torch.exp(outputs).cpu().detach().numpy())
#     outputs = net(images)
#     print outputs
#     outputs = net(images)
#     print outputs
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return outputPoints
    
    

def genData(indices,dataset,savingName,numOfModels):
    trainloaders,testloader = load_dataset(dataset,indices,numOfModels+1,savingName)
    model = Net()
    if use_cuda:
        model = model.to(device)
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    modelOrig = trainModel(trainloaders[0], optimizer, model)
    save_checkpoint(modelOrig.state_dict(), savingName+str(0))
    origModelOutput = testModel(testloader, modelOrig)
    #Saving the output of the orgininal model before being updated
    np.savez_compressed('./models output/'+folderName+'/'+savingName+'ModelOutput.npz', origModelOutput)
    
    
    labels =[]
    outputDifferences = []
    #updating the model for a "numOfModels" times in parallel, i.e., the model is updated on each data batch independently
    for i in range(1,numOfModels+1):
        tempOrigModel = copy.deepcopy(modelOrig) 
        optimizer = optim.SGD(tempOrigModel.parameters(), lr=0.001, momentum=0.9)
        
        model,label = updateModel(trainloaders[i], optimizer, tempOrigModel)
        labels.append(label)
        save_checkpoint(model.state_dict() , savingName+str(i))
        updatedOutputs = testModel(testloader, model)
        outputDiff = np.array(origModelOutput) - np.array(updatedOutputs)
        outputDifferences.append(outputDiff)
        if i%1000==0:
            #Saving the output of every 1000 updated model in a single file
            print(i)
            np.savez_compressed('./models output/'+folderName+'/'+savingName+'OutputDifferences'+str(i/1000)+'.npz', np.array(outputDifferences))
            np.savez_compressed('./models output/'+folderName+'/'+savingName+'Labels'+str(i/1000)+'.npz', np.array(labels))
            labels =[]
            outputDifferences = []
            
    #saving the output of the last batch of models
    np.savez_compressed('./models output/'+folderName+'/'+savingName+'OutputDifferences'+str((i/1000)+1)+'.npz', np.array(outputDifferences))
    np.savez_compressed('./models output/'+folderName+'/'+savingName+'Labels'+str((i/1000)+1)+'.npz', np.array(labels))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    use_cuda = torch.cuda.is_available()
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

    traingSet = torchvision.datasets.MNIST(root=MNISTFolder, train=True,
                                            download=True, transform=transform)
    
    num_train = len(traingSet)
    print(num_train)
    totalIndices = list(range(num_train))
    targetIndices = np.random.choice(totalIndices, size=20000, replace=False)
    shadowIndices =  list(set(totalIndices) - set(targetIndices))

    genData(targetIndices, traingSet, 'shadow',numOfModels = 10000)
    genData(shadowIndices, traingSet, 'target',numOfModels = 1000)
    pass
    
    
    
