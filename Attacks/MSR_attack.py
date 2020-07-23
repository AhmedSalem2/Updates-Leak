'''
Created on 11 Jan 2019

@author: ahmed.salem
Based on the code of the DC-GAN from: https://gist.github.com/ptrblck/a867c562d3e194420e7d9d41127af961
'''

import argparse
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import torchvision.utils as vutils
from network import _GeneratorMNIST, _DiscriminatorMNIST
from torch.utils.data.sampler import SubsetRandomSampler
from utils import getDifference,load_dataSingle, weights_init




parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./data/', help='path to dataset')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--nz', type=int, default=50, help='size of the latent z vector')
parser.add_argument('--attackZ', type=int, default=50, help='size of the output of the encoder, i.e., the latent code')
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='Flag to use cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu_id', default='0', help='The ID of the specified GPU')
parser.add_argument('--numOfPoints', type=int, default=100, help='Size of the probing set')
parser.add_argument('--sizeOfUpdateSet', default='100', help='Size of the update set')
parser.add_argument('--outf', default='.', help='The directory to save the generated images')





              
#loading the images for training our CBM-GAN
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


#Finding the closest fake images to the real one to calculate the BM loss.
def closestPoints(realImgs, fakeImgs):
    fakeT = realImgs.view(-1,realImgs.size(1),1)
    inputT = fakeImgs.view(-1,fakeImgs.size(1),1)
    if opt.cuda:
        fakeT  = fakeT.cuda()
        inputT = inputT.cuda()
    minL2 = fakeT - inputT.permute(2,1,0)
    minL2 = minL2.pow(2).sum(1)
    indx = torch.argmin(minL2,dim=1)
    return fakeImgs[indx]




opt = parser.parse_args()
#Size of the probing set
trainingPoints = int(opt.numOfPoints)
#Id of the gpu to runt he code
gpusToRun = opt.gpu_id
attackZ = opt.attackZ

#Folder to save the images generated
outf = opt.outf+'/outputImages_updateSetSize:'+opt.sizeOfUpdateSet+'_probingSetSize:'+str(trainingPoints)

print(opt)

#The model used to train the CBM-GAN, 
name = 'shadow'
attackVectorData = getDifference(name,numOfPoints=trainingPoints,sizeOfUpdateSet = opt.sizeOfUpdateSet,numOfFiles=10)
images,_ = getImages(name,opt.sizeOfUpdateSet)
images = (torch.stack(images))


dataset  = torch.utils.data.TensorDataset(attackVectorData,images)



#The gpu id to use
if opt.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpusToRun

#Creating the folder to store the generated images
try:
    os.makedirs(outf+'/images')
    os.makedirs(outf+'/models')
except OSError:
    print('cannot create folder')
    pass

#Setting a random seed is none is given as input
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
    
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
#Batch size need
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                              shuffle=True, num_workers=int(opt.workers))

# some hyper parameters
nz = int(opt.nz)


if opt.cuda:
    netG = _GeneratorMNIST(nz,attackZ,attackVectorData.size()[1]).cuda()
else:
    netG = _GeneratorMNIST(nz,attackZ,attackVectorData.size()[1])

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.cuda:
    netD = _DiscriminatorMNIST().cuda()
else:
    netD = _DiscriminatorMNIST()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)







criterion = nn.BCELoss()
recons_criterion = nn.MSELoss() 
evalAttackVec,evalImg =  iter(trainloader).next()

evalImg = evalImg[0]
evalAttackVec = evalAttackVec.repeat(evalImg.size(0),1)
if opt.cuda:
    eval_noise = torch.randn(evalAttackVec.size(0), nz).cuda()
    evalAttackVec = evalAttackVec.cuda()
else:
    eval_noise = torch.randn(evalAttackVec.size(0), nz)


vutils.save_image(
                evalImg.data,
                outf+'/images/realImgs.png'
            )
real_label = 1
fake_label = 0

# setup optimizers
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(trainloader, 0):  
              
        ############################
        # (1) Update D network: using the normal GAN's discriminator loss.
        ###########################
        # train with real
        netD.zero_grad()
        attackVec, imgs = data
        imgs = imgs[0]
        attackVec = attackVec.repeat(imgs.size(0),1)
        batch_size = attackVec.size(0)
        if opt.cuda:
            attackVec = attackVec.cuda()
            imgs = imgs.cuda()
            label = torch.full((batch_size,), real_label).cuda()
        else:
            label = torch.full((batch_size,), real_label)
        
        imgs = imgs.view(imgs.size(0),imgs.size(2)*imgs.size(3))
        output = netD(imgs)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        if opt.cuda:
            noise = torch.randn(batch_size, nz).cuda()
        else:
            noise = torch.randn(batch_size, nz)
        fake = netG(noise,attackVec)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        optimizerD.step()

        ############################
        # (2) Update G network: Uses both the normal GANs' generator -minmax- loss and the Best Match loss 
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        #The normal GAN loss
        advErrG = criterion(output, label)
        #Finding the best matches
        close = closestPoints(imgs,fake)
        #The Best Match loss
        reconsLoss = recons_criterion(imgs, close)
        #Adding both losses to update the generator
        errG = reconsLoss + advErrG
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
    print('[%d/%d] AdvLoss: %.4f ReconsLoss: %.4f TotalLoss: %.4f '%(epoch,opt.niter,advErrG.item(),reconsLoss.item() ,errG.item() ))
    if(epoch%10 ==0):
        #Save the generator's output every 10 epochs
        fake = netG(eval_noise,evalAttackVec)
        fake = fake.view(fake.size(0),1,28,28)
        vutils.save_image(fake.data,
                '%s/images/fake_samples_epoch_%03d.png' % (outf, epoch))


    if(epoch%100 ==0):
        #Save the generator and discriminator every 100 epochs
        torch.save(netG.state_dict(), '%s/models/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.state_dict(), '%s/models/netD_epoch_%d.pth' % (outf, epoch))
    else:
        #Save the latest generator and discriminator
        torch.save(netG.state_dict(), '%s/models/netG.pth' % (outf))
        torch.save(netD.state_dict(), '%s/models/netD.pth' % (outf))
