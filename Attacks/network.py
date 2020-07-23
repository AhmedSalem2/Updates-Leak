import torch
import torch.nn as nn
import torch.nn.functional as F

    
#Model for the multi-sample label estimation attack for the MNIST dataset
class labelPredMNIST(nn.Module):
    def __init__(self, attackInput,numOfClasses=10):
        super(labelPredMNIST, self).__init__()
        #The encoder is considered all layers except the last one, which is considered to be the decoder in this attack.
        self.preAttack = nn.Sequential(
            nn.Linear(attackInput, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, numOfClasses),   
            nn.LogSoftmax(dim= 1)
            )
    def forward(self,inputAttack):
        return self.preAttack(inputAttack)

#Model for the single-sample label estimation attack for the MNIST dataset
class singleLabelPredMNIST(nn.Module):
    def __init__(self, attackInput,numOfClasses=10):
        super(singleLabelPredMNIST, self).__init__()
        #The encoder is considered all layers except the last one, which is considered to be the decoder in this attack.
        self.preAttack = nn.Sequential(
            nn.Linear(attackInput, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, numOfClasses),   
            nn.LogSoftmax(dim= 1)
            )
    def forward(self,inputAttack):
        return self.preAttack(inputAttack)


class _GeneratorMNIST(nn.Module):
    def __init__(self, nz, attackZ,attackInput):
        super(_GeneratorMNIST, self).__init__()
        self.nz = nz
        self.attackZ = attackZ
        self.attackInput = attackInput
        self.fc1 = nn.Linear(self.attackZ+self.nz, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 784)
        self.relu = nn.ReLU()
        
        #Encoder
        self.preAttack = nn.Sequential(
            nn.Linear(attackInput, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, attackZ),
            nn.Tanh()
        )
    # forward method
    def forward(self, inputZ,inputAttack):
        #Passing the input to the encoder first.
        pre    = self.preAttack(inputAttack)
        #Concatenating the encoder result, i.e., the latent code, with the noise vector
        combinedInput  = torch.cat((inputZ,pre), dim=1)
        combinedInput = combinedInput.view(combinedInput.size(0), (self.attackZ+self.nz))
        #Inputting the concatenated vector the the generator
        x = self.relu(self.bn1(self.fc1(combinedInput)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))

        return x


class _DiscriminatorMNIST(nn.Module):
    def __init__(self):
        super(_DiscriminatorMNIST, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.sgmd =  nn.Sigmoid()

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.bn1(self.fc1(input)),0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = torch.tanh(self.fc4(x))

        return self.sgmd(x).view(-1)
    
    
