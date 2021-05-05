# Kerakli kutubxonalarni chaqirib olish
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training paramterlari
batch_size = 64  # batch o'lchami 64
device = 'cuda' if cuda.is_available() else 'cpu'  # GPU (cuda)da train(agar mavjud bo'lsa)
print(f'MNIST modelini {device} da train qilish !\n{"=" * 44}')

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader 
train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class MNIST_Net(nn.Module):

    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Ma'lumotni reformalash (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = MNIST_Net() #model chaqirib olish
model.to(device)  # cuda ga joylash (agar mavjud bo'lsa)
criterion = nn.CrossEntropyLoss()  # loss (CrossEntropy)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # optimizer SGD

# Training uchun funksiya
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Holati: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Test uchun funkisya
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # batchdagi loss larni yig'indisini hisoblash
        test_loss += criterion(output, target).item()
        # max qiymatning indeksini olish
        pred = output.data.max(1, keepdim=True)[1]   # torch.max() funkisyasi har ikkala qiymat va indekslarni qaytaradi
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest : Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')


if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training uchun ketgan vaqt: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Test uchun ketgan vaqt: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Umumiy vaqt: {m:.0f}m {s:.0f}s\nModel  {device} qurilmada train qilindi!')

