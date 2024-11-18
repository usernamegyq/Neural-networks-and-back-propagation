# vision transformer实现voc图像分类
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from torchvision.models.vision_transformer import VisionTransformer


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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])



trainset = CustomDataset(train_ids, labels[:len(train_ids)], root, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = CustomDataset(test_ids, labels[len(train_ids):], root, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patches = (image_size // patch_size) ** 2

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=patch_size,stride=patch_size)

    def forward(self, x):
        out = self.conv1(x)
        out = out.flatten(2)
        out = out.transpose(1,2)
        return out

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super().__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        positions = torch.zeros(self.num_patches+1,self.embedding_dim)

        for i in range(self.embedding_dim):
            for j in range(self.num_patches+1):
                if j%2 == 0:
                    positions[j,i] = np.sin(j/10000**(i/self.embedding_dim))
                else:
                    positions[j,i] = np.cos(j/10000**(i/self.embedding_dim))
        self.position = nn.Parameter(positions)

    def forward(self, x):
        self.cls_token_expanded = self.cls_token.expand(x.size(0), -1, -1)
        #self.cls_token = nn.Parameter(self.cls_token)
        out = torch.cat((self.cls_token_expanded, x), dim=1)
        out = out + self.position
        return out


class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = head_size
        self.query = nn.Linear(embedding_dim, head_size)
        self.key = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = self.softmax((q@k.transpose(-2,-1))/np.sqrt(self.head_dim)@v)
        return attention

class MultiHeadAttention(nn.Module):    
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.attention_heads = nn.ModuleList([AttentionHead(embedding_dim, self.head_size) for _ in range(num_heads)])
        #self.output = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(self.head_size * num_heads, embedding_dim)

    def forward(self,x):
        out = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        out = self.output(out)
        return out
    
class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        out = (x - mean) / (std + 1e-6)
        out = self.gamma * out + self.beta
        return out

class FeedForwad(nn.Module):
    def __init__(self, embedding_dim, feedforward_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.fc1 = nn.Linear(embedding_dim, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x):
        out = self.dropout(self.gelu(self.fc1(x)))
        out = self.fc2(out)
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
        x = x.transpose(1,2).unsqueeze(3)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        out = out.squeeze(3).transpose(1,2)
        return out
    
class Encoder(nn.Module):
    def __init__(self,embedding_dim,feedforward_dim,num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.norm1 = LayerNorm(self.embedding_dim)
        self.mha = MultiHeadAttention(self.embedding_dim, self.num_heads)
        self.rc1 = ResidualBlock(self.embedding_dim, self.embedding_dim)
        self.norm2 = LayerNorm(self.embedding_dim)
        self.ff = FeedForwad(self.embedding_dim, self.feedforward_dim)
        self.rc2 = ResidualBlock(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x):
        out = self.norm1(x)
        out = self.mha(out)
        out = self.dropout(out)
        out = self.rc1(out)
        out = self.norm2(out)
        out = self.ff(out)
        out = self.dropout(out)
        out = self.rc2(out)
        return out

class VisionTransformer(nn.Module):
    def __init__(self, image_size,patch_size,in_channels,embedding_dim, feedforward_dim, num_heads,num_classes,num_layers=6):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(self.image_size, self.patch_size, self.in_channels,self.embedding_dim)
        self.position_embedding = PositionEmbedding(self.num_patches, self.embedding_dim)
        self.encoder = Encoder(self.embedding_dim, self.feedforward_dim, self.num_heads)
        self.encoder_layers = nn.ModuleList([Encoder(self.embedding_dim, self.feedforward_dim, self.num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self,x):
        out = self.patch_embedding(x)
        out = self.position_embedding(out)
        # out = self.encoder(out)
        # out = self.fc(out[:,0])
        for layer in self.encoder_layers:
             out = layer(out)
        out = out.mean(dim=1)
        
        out = self.fc(out)
        return out

image_size = 224
patch_size = 16
in_channels = 3
embedding_dim = 128
feedforward_dim = 512
num_heads = 16
num_classes = 20
lr = 5e-4
epochs = 30


model = VisionTransformer(image_size,patch_size,in_channels,embedding_dim, feedforward_dim, num_heads, num_classes)
import os
path = 'model_voc_2_0.pt'
if os.path.exists(path):
    model.load_state_dict(torch.load(path,weights_only=True))
# class_counts = torch.tensor([343, 284, 370, 248, 341, 208, 571, 541, 553, 152, 
#                              269, 654, 245, 261, 2093, 258, 154, 250, 271, 285], dtype=torch.float32)
class_counts = torch.tensor([ 670,  552,  765,  508,  706,  421, 1161, 1080, 1119,  303,  538, 1286,
  482,  526, 4087,  527,  325,  507,  544,  575])

class_weights = 1.0 / (class_counts)


# criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
#损失函数用hanming loss
# criterion = nn.HingeEmbeddingLoss()

#optimizer = optim.Adam(model.parameters(),lr=lr)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
#用SGD优化
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=lr, steps_per_epoch=len(trainloader), epochs=epochs
# )
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




####################################################
# from sklearn.metrics import precision_recall_curve
# model.eval()  # 切换到评估模式
# train_probs = []  # 存储预测概率
# train_labels = []  # 存储真实标签

# with torch.no_grad():  # 禁用梯度计算
#     for images, labels in trainloader:
#         outputs = model(images)  # 获取模型输出
#         probs = torch.sigmoid(outputs)  # 使用 Sigmoid 函数计算每个类别的概率
#         train_probs.append(probs.numpy())  # 保存概率
#         train_labels.append(labels.numpy())  # 保存真实标签

# all_probs = np.vstack(train_probs)  # 将批次预测概率堆叠为完整数组
# all_labels = np.vstack(train_labels)  # 将批次标签堆叠为完整数组

# #all_labels = labels[:len(train_ids)]
# optimal_thresholds = []
# for i in range(num_classes):
#     # 获取当前类别的预测概率和实际标签
#     probs = all_probs[:, i]
#     labels = all_labels[:, i]
    
#     # 计算当前类别的精确率、召回率和阈值
#     precision, recall, thresholds = precision_recall_curve(labels, probs)
    
#     # 计算 F1 分数
#     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
#     # 找到 F1 分数最高的阈值
#     # optimal_idx = np.argmax(f1_scores)
#     # optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
#     # optimal_thresholds.append(optimal_threshold)
#     optimal_idx = np.argmax(precision)
#     optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 1.0
#     optimal_thresholds.append(optimal_threshold)

    
# print("Optimal thresholds per class:", optimal_thresholds)


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def evaluate(model, testloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            
            #print(outputs.shape)
            # for i in range(outputs.shape[0]):
            #     if torch.all(outputs[i] < 0.5):
            #         max_index = torch.argmax(outputs[i])
            #         outputs[i] = torch.zeros_like(outputs[i])
            #         outputs[i][max_index] = 1
            #     else:
            #         outputs[i] = (outputs[i] >= 0.5).int()
            outputs = torch.sigmoid(outputs)
            print(outputs)
            print(outputs.max(dim=1))
            outputs = (outputs >= 0.5).int()
            # for i in range(outputs.shape[0]):
            #     for j in range(outputs.shape[1]):
            #         if outputs[i][j] >= optimal_thresholds[j]:
            #             outputs[i][j] = 1
            #         else:
            #             outputs[i][j] = 0
            # for i in range(outputs.shape[1]):
            #     outputs[:,i] = (outputs[:,i] >= optimal_thresholds[i]).int()
            all_preds.extend(outputs.numpy())
            all_labels.extend(labels.numpy())

    # # 计算精确率
    # precision = precision_score(all_labels, all_preds, average='macro',zero_division=0)
    # # 计算召回率
    # recall = recall_score(all_labels, all_preds, average='macro',zero_division=0)
    # # 计算F1分数
    # f1 = f1_score(all_labels, all_preds, average='macro',zero_division=0)
    
    # print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(classification_report(all_labels, all_preds, target_names=classes,zero_division=0))

train(model, criterion, optimizer, epochs)

torch.save(model.state_dict(), path)
evaluate(model, testloader)

# vit_base = VisionTransformer(
#     patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True
# )

# checkpoint = torch.load('D:/desktop/python_work/vit_base_patch16_224.pth', map_location='cpu')
# # Helper to read filenames
# readvdnames = lambda x: open(x).read().rstrip().split('\n')

# # Data preparation
# root = 'D:/desktop/python_work/VOC/'
# train_ids = readvdnames(root + 'ImageSets/Main/train.txt')[:6000]
# # train_ids = readvdnames(root + 'ImageSets/Main/selected_images.txt')
# test_ids = readvdnames(root + 'ImageSets/Main/val.txt')[:624]
# num_ids = len(train_ids) + len(test_ids)

# classes = (
#     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
#     'sheep', 'sofa', 'train', 'tvmonitor'
# )

# labels = np.zeros((num_ids, len(classes)), dtype=int)
# for i in classes:
#     train_res = readvdnames(root + 'ImageSets/Main/' + i + '_train.txt')[:6000]
#     labels[:len(train_ids), classes.index(i)] = [int(x.split()[1]) for x in train_res]
#     val_res = readvdnames(root + 'ImageSets/Main/' + i + '_val.txt')[:624]
#     labels[len(train_ids):, classes.index(i)] = [int(x.split()[1]) for x in val_res]
# labels[labels < 0] = 0

# # Custom Dataset
# class CustomDataset(Dataset):
#     def __init__(self, ids, labels, root_dir, transform=None):
#         self.ids = ids
#         self.labels = labels
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         img_name = self.ids[idx]
#         image_path = self.root_dir + 'JPEGImages/' + img_name + '.jpg'
#         image = Image.open(image_path).convert('RGB')
#         label = self.labels[idx]
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, torch.tensor(label, dtype=torch.float32)


# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(20),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# trainset = CustomDataset(train_ids, labels[:len(train_ids)], root, transform=transform)
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
# testset = CustomDataset(test_ids, labels[len(train_ids):], root, transform=transform)
# testloader = DataLoader(testset, batch_size=64, shuffle=False)
# # from PIL import Image
# # import torch
# # import torchvision.transforms as transforms
# # import numpy as np
# # from torch.utils.data import Dataset, DataLoader
# # import os

# # # Helper to read filenames
# # readvdnames = lambda x: open(x).read().rstrip().split('\n')

# # # Data preparation
# # root = 'D:/desktop/python_work/VOC/'
# # train_ids = readvdnames(root + 'ImageSets/Main/train.txt')[:6000]
# # test_ids = readvdnames(root + 'ImageSets/Main/val.txt')[:624]
# # num_ids = len(train_ids) + len(test_ids)

# # classes = (
# #     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
# #     'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
# #     'sheep', 'sofa', 'train', 'tvmonitor'
# # )

# # labels = np.zeros((num_ids, len(classes)), dtype=int)
# # for i in classes:
# #     train_res = readvdnames(root + 'ImageSets/Main/' + i + '_train.txt')[:6000]
# #     labels[:len(train_ids), classes.index(i)] = [int(x.split()[1]) for x in train_res]
# #     val_res = readvdnames(root + 'ImageSets/Main/' + i + '_val.txt')[:624]
# #     labels[len(train_ids):, classes.index(i)] = [int(x.split()[1]) for x in val_res]
# # labels[labels < 0] = 0

# # # Custom Dataset
# # class CustomDataset(Dataset):
# #     def __init__(self, ids, labels, root_dir, transform=None):
# #         self.ids = ids
# #         self.labels = labels
# #         self.root_dir = root_dir
# #         self.transform = transform

# #     def __len__(self):
# #         return len(self.ids)

# #     def __getitem__(self, idx):
# #         img_name = self.ids[idx]
# #         image_path = self.root_dir + 'JPEGImages/' + img_name + '.jpg'
# #         image = Image.open(image_path).convert('RGB')
# #         label = self.labels[idx]

# #         if self.transform:
# #             image = self.transform(image)

# #         return image, torch.tensor(label, dtype=torch.float32)

# # # Data augmentation and preprocessing
# # transform = transforms.Compose([
# #     transforms.RandomHorizontalFlip(),
# #     transforms.RandomRotation(20),
# #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
# #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# # ])

# # # Handle imbalance by undersampling the 'person' class
# # from collections import Counter

# # # 1. Count samples per class
# # class_counts = np.sum(labels[:len(train_ids)], axis=0)
# # print("Original class sample counts:", dict(zip(classes, class_counts)))

# # # 2. Undersample 'person' class
# # target_count = int(np.mean(class_counts[class_counts != class_counts[classes.index('person')]]))  # Mean count for other classes
# # person_indices = np.where(labels[:len(train_ids), classes.index('person')] == 1)[0]  # Indices for 'person' class
# # non_person_indices = np.where(labels[:len(train_ids), classes.index('person')] == 0)[0]  # Indices for non-'person' class

# # # Randomly select target_count 'person' samples
# # np.random.seed(42)  # Fix random seed for reproducibility
# # selected_person_indices = np.random.choice(person_indices, target_count, replace=False)

# # # Combine 'person' and non-'person' indices
# # final_indices = np.concatenate((selected_person_indices, non_person_indices))
# # np.random.shuffle(final_indices)  # Shuffle the indices

# # # 3. Create undersampled training set
# # undersampled_train_ids = [train_ids[i] for i in final_indices]
# # undersampled_labels = labels[final_indices]

# # # 4. Create balanced trainloader
# # trainset_balanced = CustomDataset(undersampled_train_ids, undersampled_labels, root, transform=transform)
# # trainloader_balanced = DataLoader(trainset_balanced, batch_size=64, shuffle=True)

# # # Create testloader
# # testset = CustomDataset(test_ids, labels[len(train_ids):], root, transform=transform)
# # testloader = DataLoader(testset, batch_size=64, shuffle=False)

# # # Output adjusted class counts
# # balanced_class_counts = np.sum(undersampled_labels, axis=0)
# # print("Balanced class sample counts:", dict(zip(classes, balanced_class_counts)))





# # Vision Transformer components
# class PatchEmbedding(nn.Module):
#     def __init__(self, image_size, patch_size, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = out.flatten(2).transpose(1, 2)
#         return out


# class PositionEmbedding(nn.Module):
#     def __init__(self, num_patches, embedding_dim):
#         super().__init__()
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
#         positions = torch.zeros(num_patches + 1, embedding_dim)
#         for i in range(embedding_dim):
#             for j in range(num_patches + 1):
#                 if j % 2 == 0:
#                     positions[j, i] = np.sin(j / 10000**(i / embedding_dim))
#                 else:
#                     positions[j, i] = np.cos(j / 10000**(i / embedding_dim))
#         self.position = nn.Parameter(positions)

#     def forward(self, x):
#         cls_token_expanded = self.cls_token.expand(x.size(0), -1, -1)
#         out = torch.cat((cls_token_expanded, x), dim=1)
#         out = out + self.position
#         return out


# class MultiHeadAttention(nn.Module):
#     def __init__(self, embedding_dim, num_heads):
#         super().__init__()
#         head_size = embedding_dim // num_heads
#         self.attention_heads = nn.ModuleList([nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)])
#         self.output = nn.Linear(embedding_dim, embedding_dim)

#     def forward(self, x):
#         out = self.attention_heads[0](x, x, x)[0]
#         out = self.output(out)
#         return out


# class FeedForward(nn.Module):
#     def __init__(self, embedding_dim, feedforward_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(embedding_dim, feedforward_dim)
#         self.fc2 = nn.Linear(feedforward_dim, embedding_dim)
#         self.gelu = nn.GELU()

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.gelu(out)
#         out = self.fc2(out)
#         return out


# class Encoder(nn.Module):
#     def __init__(self, embedding_dim, feedforward_dim, num_heads):
#         super().__init__()
#         self.mha = MultiHeadAttention(embedding_dim, num_heads)
#         self.ff = FeedForward(embedding_dim, feedforward_dim)

#     def forward(self, x):
#         x = self.mha(x)
#         x = self.ff(x)
#         return x


# class VisionTransformer(nn.Module):
#     def __init__(self, image_size, patch_size, in_channels, embedding_dim, feedforward_dim, num_heads, num_classes, num_layers=12):
#         super().__init__()
#         self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embedding_dim)
#         self.position_embedding = PositionEmbedding((image_size // patch_size)**2, embedding_dim)
#         self.encoders = nn.ModuleList([Encoder(embedding_dim, feedforward_dim, num_heads) for _ in range(num_layers)])
#         self.fc = nn.Linear(embedding_dim, num_classes)

#     def forward(self, x):
#         x = self.patch_embedding(x)
#         x = self.position_embedding(x)
#         for encoder in self.encoders:
#             x = encoder(x)
#         x = x.mean(dim=1)
#         x = self.fc(x)
#         return x


# # Model setup
# model = VisionTransformer(224, 16, 3, 128, 512, 16, 20)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# # Early stopping
# best_loss = float('inf')
# patience = 5
# trigger_times = 0


# def train(model, criterion, optimizer, scheduler, epochs):
#     global best_loss, trigger_times
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in trainloader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         avg_loss = running_loss / len(trainloader)
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

#         # if avg_loss < best_loss:
#         #     best_loss = avg_loss
#         #     trigger_times = 0
#         #     torch.save(model.state_dict(), 'best_model.pth')
#         # else:
#         #     trigger_times += 1
#         #     if trigger_times >= patience:
#         #         print("Early stopping!")
#         #         return
#         scheduler.step()


# def evaluate(model):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for inputs, labels in testloader:
#             outputs = model(inputs)
#             preds = (torch.sigmoid(outputs) >= 0.5).int()
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     f1 = f1_score(all_labels, all_preds, average='macro')
#     print(classification_report(all_labels, all_preds, target_names=classes))
#     print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


# train(model, criterion, optimizer, scheduler, epochs=10)
# torch.save(model.state_dict(), 'final_model.pth')
# model.load_state_dict(torch.load('best_model.pth'))
# evaluate(model)