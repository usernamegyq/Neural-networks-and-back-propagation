from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import torchvision.models as models


readvdnames = lambda x: open(x).read().rstrip().split('\n')
root = 'D:/desktop/python_work/VOC/'
train_ids = readvdnames(root + 'ImageSets/Main/train.txt')[:6000]
test_ids = readvdnames(root + 'ImageSets/Main/val.txt')[:624]
num_ids = len(train_ids) + len(test_ids)

classes = ('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor')
num_classes = len(classes)
labels = np.zeros((num_ids, len(classes)), dtype=int)

labels = np.zeros((num_ids,20))
for i in classes:
    train_res = readvdnames(root + 'ImageSets/Main/' + i + '_train.txt')[:6000]
    labels[:len(train_ids),classes.index(i)] = [i.split()[1] for i in train_res]
    val_res = readvdnames(root + 'ImageSets/Main/' + i + '_val.txt')[:624]
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = CustomDataset(train_ids, labels[:len(train_ids)], root, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = CustomDataset(test_ids, labels[len(train_ids):], root, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

lr = 1e-3
epochs = 20

import torchvision.models as models
model = models.resnet18(pretrained=True)
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
# model = models.mobilenet_v2(pretrained=True)
# model = models.mobilenet_v3_small(pretrained=True)
import torch.nn as nn
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
def train(model, criterion, optimizer, epochs):
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


def evaluate(model, testloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            outputs = (outputs >= 0.5).int()
            all_preds.extend(outputs.numpy())
            all_labels.extend(labels.numpy())

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(classification_report(all_labels, all_preds, target_names=classes))

train(model, criterion, optimizer, epochs)
evaluate(model, testloader)
