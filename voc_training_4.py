
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


readvdnames = lambda x: open(x).read().rstrip().split('\n')
root = 'D:/desktop/python_work/VOC/'
train_ids = readvdnames(root + 'ImageSets/Main/train.txt')
train_ids = train_ids[:6000]
test_ids = readvdnames(root + 'ImageSets/Main/val.txt')
test_ids = test_ids[:624]
num_ids = len(train_ids) + len(test_ids)

classes = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor')

labels = np.zeros((num_ids, len(classes)), dtype=int)

labels = np.zeros((num_ids,20))
for i in classes:
    train_res = readvdnames(root + 'ImageSets/Main/' + i + '_train.txt')
    train_res = train_res[:6000]
    labels[:len(train_ids),classes.index(i)] = [i.split()[1] for i in train_res]
    val_res = readvdnames(root + 'ImageSets/Main/' + i + '_val.txt')
    val_res = val_res[:624]
    labels[len(train_ids):,classes.index(i)] = [i.split()[1] for i in val_res]
labels[labels < 0] = 0

class CustomDataset(Dataset):
    def __init__(self, ids, labels, root_dir, transform=None):
        self.ids = ids
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        image_path = self.root_dir + 'JPEGImages/' + img_name + '.jpg'
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])


trainset = CustomDataset(train_ids, labels[:len(train_ids)], root, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)  
testset = CustomDataset(test_ids, labels[len(train_ids):], root, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)  

class VGG(nn.Module):
    def __init__(self,image_size=224,in_channels=3,num_classes=20):
        super(VGG, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,20)
        )

    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out


lr = 1e-4
epochs =10


model = VGG()
vgg16_pretrained = torchvision.models.vgg16(pretrained=True)
# vgg16_pretrained = torchvision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
# vgg16_pretrained = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT')
vgg16_pretrained.classifier[6] = nn.Linear(4096,20)


import os
path = 'vgg_voc_training_4.pth'
if os.path.exists(path):
    model.load_state_dict(torch.load(path))
else:
    state_dict = vgg16_pretrained.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
def train(model,criterion,optimizer,epochs):

    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def evaluate(model, testloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs >= 0.5).int()
            all_preds.extend(outputs.numpy())
            all_labels.extend(labels.numpy())

    precision = precision_score(all_labels, all_preds, average='macro',zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro',zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro',zero_division=0)
    
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(classification_report(all_labels, all_preds, target_names=classes,zero_division=0))

train(model, criterion, optimizer, epochs)
torch.save(model.state_dict(), 'vgg_voc_training_4.pth')
evaluate(model, testloader)