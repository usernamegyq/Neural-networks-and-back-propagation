#采用lenet，vgg，resnet-18进行mnist图像分类
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import utils

transform = transforms.Compose([
    #transforms.Resize((32,32)),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,transform=transform,download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False,transform=transform,download=True)
trainloader = utils.data.DataLoader(trainset, batch_size=60, shuffle=True)
testloader = utils.data.DataLoader(testset, batch_size=60, shuffle=False)

class LeNet(nn.Module):
    def __init__(self):#需要传入的数据
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        out = self.pool1(torch.relu(self.conv1(x)))
        out = self.pool2(torch.relu(self.conv2(out)))
        out = out.view(-1,16*4*4)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(1,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128,128,3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*7*7,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,10)

    def forward(self,x):
        out = torch.relu(self.conv1(x))
        out = torch.relu(self.conv2(out))
        out = self.pool1(out)
        out = torch.relu(self.conv3(out))
        out = torch.relu(self.conv4(out))
        out = self.pool2(out)
        out = out.view(-1,128*7*7)  
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(
            ResidualBlock(64,64),
            ResidualBlock(64,64)
        )
        # self.layer2 = nn.Sequential(
        #     ResidualBlock(64,128),
        #     ResidualBlock(128,128)
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,10)

    def forward(self,x): 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        # out = self.layer2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = ResNet()
#model = VGG()
#model = LeNet()
cost = nn.CrossEntropyLoss()
lr = 1e-3
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
total_step = len(trainloader)
epochs = 10
def train(model, criterion, optimizer, trainloader, epochs):
    for epoch in range(epochs):
        for image,label in trainloader:
            output = model(image)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, epochs, loss.item()))

def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

train(model,cost,optimizer,trainloader,epochs)
evaluate(model, testloader)