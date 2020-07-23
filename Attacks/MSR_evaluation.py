'''
Created on 29 Oct 2018

@author: ahmed.salem
'''

from network import _GeneratorMNIST
import torch
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
from sklearn.cluster import KMeans
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils as vutils
import os
from sklearn.metrics import mean_squared_error
from scipy.optimize import linear_sum_assignment
from utils import getDifference,load_dataSingle


#Size of probing set
sizeOfProbingSet = 100
#Number of epochs the CBM-GAN was trained with
numOfEpoch = 20
#Size of updating set
sizeOfUpdatingSet = '100'

#Main folder to load and store the data for this attack
outf = './outputImages_updateSetSize:'+sizeOfUpdatingSet+'_probingSetSize:'+str(sizeOfProbingSet)
#Datasets/models to attack, this is the other data that the CBM-GAN is NOT trained on
name = 'target'
#Path to the trained CBM-GAN, this is for a specific epoch
path =  '%s/models/netG_epoch_%d.pth' % (outf, numOfEpoch)
#This is for the latest trained model
#path =  '%s/models/netG.pth' % (outf)

#Folder to save generated images in
outputFolder = '%s/eval/numberOfEpochs_%d'%(outf, numOfEpoch)

try:
    os.makedirs(outputFolder)
except OSError:
    pass

#Noise and latent (output of the encoder) vectors sizes, same as the ones used when training the CBM-GAN
attackZ = 50
nz = 50
#Back size of one, as every entry corresponds to a different updated model
batch_size = 1



#Creating the graph needed for the Hungarian algorithm, i.e., it create a matrix with rows being the images and columns the cluster's representatives
#Each cell corresponds to the MSE between an image and a cluster's representative
def getGraph(imgs, clusters):
    graph = []
    #Looping over all images
    for f in imgs:
        #Calculating the MSE between this image and all different clusters' representatives independently
        if(len(clusters.size()) > 2):
            inpTemp = f.repeat(clusters.size(0),1,1,1)
        else:
            inpTemp = f.repeat(clusters.size(0),1,1)
            
        inpTemp = inpTemp-clusters    
        
        if(len(clusters.size()) > 2):
            inpTemp = inpTemp.view(-1,clusters.size(1)*clusters.size(2)*clusters.size(3))
        else:
            inpTemp = inpTemp.view(-1,clusters.size(1)*clusters.size(2))   
        
        dist = torch.pow(inpTemp, 2).sum(1)/inpTemp.size(1)
        graph.append(dist.detach().numpy())
    return graph
        
        
#Preprocessing/denormalizing the image for calculating the MSE
#The denormalization (line 78 and 79) are commented for a better visualization for the plotted images
def to_img(x):
    x = (0.3081 * x) + 0.1307
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1,28,28)
    return x


def getImages(name,sizeOfUpdateSet):
    path='../data indices/updating'+sizeOfUpdateSet+'Points/MNISTdataIndex'+name
    dataroot = './data/'
    imgs=[]
    labels=[]
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root=dataroot, train=True,
                                            download=False, transform=transform)
    indices = load_dataSingle(path+'.npz')  
    indices = indices[0]
    for indx in indices[1:]:
        train_sampler = SubsetRandomSampler(indx)
        trainloaderTemp = torch.utils.data.DataLoader(trainset, batch_size=len(indx),
                                              shuffle=False, num_workers=0,sampler=train_sampler)   
        dataiter = iter(trainloaderTemp)
        img,label =  dataiter.next()
        labels.append(label)
        img = img
        imgs.append(img)
    return imgs,labels


#Finding the clusters' representatives
def getClosestImg(imgs,labels,centriods):
    counter = 0
    res = []
    #Looping over all different clusters
    for cent in centriods:
        indx = np.where(labels == counter)[0]
        tempPoints = imgs[indx]
        min = 99999
        minHolder =0
        #Looping over all images in the cluster to find the closest one to the center
        for point in tempPoints:
            
            mse =  mean_squared_error(cent, point)
            if(mse < min):
                min = mse
                minHolder = point
        res.append(torch.from_numpy(minHolder))
        counter = counter+1
    return res



#Number of images to generate with our CBM-GAN
generateImgs = 20000

#Creating the dataset of the images and posterior difference
images,_ = getImages(name,sizeOfUpdatingSet)
images = (torch.stack(images))
attackVectorData = getDifference(name,numOfPoints=sizeOfProbingSet,sizeOfUpdateSet=sizeOfUpdatingSet,numOfFiles=1)
dataset  = torch.utils.data.TensorDataset(attackVectorData,images)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                              shuffle=True, num_workers=0)


#Loading the trained CBM-GAN
netG =  _GeneratorMNIST(nz,attackZ,attackVectorData.size()[1])
checkpoint = torch.load(path,map_location='cpu')
netG.load_state_dict(checkpoint)


#Initializing calculations containers
trainloader = iter(trainloader)
totalMSE =0

#Number of updated models to test
NumberOfModels=10
#Initializing the best (lowest) value for the MSE form our CBM-GAN
minMSE_bm=9999999
#Initializing the best (lowest) value for the MSE form the one to one attack, i.e., find the generated image closest to each real image in the updating set
minMSE=9999999

for i in range(0,NumberOfModels):
    print('Evaluating updated model: %d'%(i))
    totalDist =0
    #Loading a single updated model to test
    attackVec, imgs = trainloader.next()
    imgs = imgs[0]
    imgs = to_img(imgs)
    
    #Generating the images with the CBM-GAN
    attackVec = attackVec.repeat(generateImgs,1)
    noise = torch.FloatTensor(generateImgs, nz)
    noise = Variable(noise)
    noise.data.resize_(generateImgs, nz).normal_(0, 1)
    noise_ = np.random.normal(0, 1, (generateImgs, nz))
    noise_ = (torch.from_numpy(noise_))
    noise.data.copy_(noise_.view(generateImgs, nz))
    fake = netG(noise,attackVec)
    fake = to_img(fake)
    fake = fake.view(fake.size(0),1,28,28)
    
    
    #Preparing data to be clustered with K-Means clustering
    random_state = 170
    numpyFake = fake.view(fake.size(0),28*28)
    numpyFake = numpyFake.detach().numpy()
    numpyImgs = imgs.view(imgs.size(0),28*28)
    numpyImgs = numpyImgs.detach().numpy()    
    
    #Clustering the data
    kmeans  = KMeans(n_clusters=int(sizeOfUpdatingSet), random_state=random_state).fit(numpyFake)
    y_pred  = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    #Finding the clusters' representatives by finding the image closest to the center of each cluster
    clusterRep = getClosestImg(numpyFake,y_pred,centers)
    #Converting clusters' representatives back to tensors
    clusterRep = (torch.stack(clusterRep))
    clusterRepImg = clusterRep.view(clusterRep.shape[0],1,28,28)

    
    #Creating the input to the Hungarian algorithm, i.e., a graph of the  MSE between the clusters' representatives and the images inside the updating set
    graph = getGraph(imgs, clusterRepImg)
    
    #Calculating the matches with the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(graph)


    #Calculating the MSE between the matched images, i.e., the result of the Hungarian algorithm
    #Container for the results of the Hungarian algorithm, i.e., the generated and real images
    attackRes = []
    for indImg,indClust in zip(row_ind, col_ind):
        attackResult = torch.from_numpy(np.array(clusterRep[indClust])) 
        origImg = torch.from_numpy(np.array(numpyImgs[indImg])) 
        totalDist = totalDist + (mean_squared_error(np.array(clusterRep[indClust]), np.array(numpyImgs[indImg])))
        #Adding the real images
        attackRes.append(origImg.view(1,28,28))
        #Adding the generated images
        attackRes.append(attackResult.view(1,28,28))

    #Calculating the average MSE over all images in side this updated model
    totalDist = totalDist/len(numpyImgs)
    
    #Calculating the average MSE over different updated models
    totalMSE = totalMSE +totalDist
    
    #Finding the minimum calculated MSE
    if(minMSE > totalDist):
        minMSE = totalDist
        attackRes = torch.stack(attackRes)
        #Plotting the matched image pairs
        for ii in range(0,10):
            vutils.save_image(
                            attackRes[ii*20:(ii+1)*20],
                            '%s/matchedPair_%s.png' % (outputFolder,str(ii)),nrow=2
                        )


    
print('avg MSE: ' + str(totalMSE/NumberOfModels))

    


