#构建一个vision transformers对mnist进行分类
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as utils
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),# 28*28
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,transform=transform,download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False,transform=transform,download=True)
trainloader = utils.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = utils.DataLoader(testset, batch_size=64, shuffle=False)

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
        self.output = nn.Linear(embedding_dim, embedding_dim)

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
    
    def forward(self,x):
        out = self.gelu(self.fc1(x))
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
    
    def forward(self,x):
        out = self.norm1(x)
        out = self.mha(out)
        out = self.rc1(out)
        out = self.norm2(out)
        out = self.ff(out)
        out = self.rc2(out)
        return out

class VisionTransformer(nn.Module):
    def __init__(self, image_size,patch_size,in_channels,embedding_dim, feedforward_dim, num_heads):
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
        self.fc = nn.Linear(self.embedding_dim, 10)

    def forward(self,x):
        out = self.patch_embedding(x)
        out = self.position_embedding(out)
        out = self.encoder(out)
        out = self.fc(out[:,0])
        return out

image_size = 28
patch_size = 14
in_channels = 1
embedding_dim = 64
feedforward_dim = 128
num_heads = 4
lr = 1e-3
epchos = 10

model = VisionTransformer(image_size,patch_size,in_channels,embedding_dim, feedforward_dim, num_heads)
cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
def train(model,cost,optimizer,epchos):
    for epoch in range(epchos):
        for image,data in trainloader:
            out = model(image)
            loss = cost(out,data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} loss {loss.item()}')

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
    print(f'Accuracy of the vision transformer on the test set: {100 * correct / total:.2f}%')

train(model,cost,optimizer,epchos)
evaluate(model,testloader)

        
