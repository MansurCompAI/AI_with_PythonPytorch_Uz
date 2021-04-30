#Kerakli kutubxonalrni chaqirib olish
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ma'lumotlarni ham class yordamida tartibga keltirib DataLoaderdan foydalanamiz.
class DiabetesDataset(Dataset):
    """ Qandli diabet ma'lumotlar to'plami"""

    # Ma'lumotlarga dastalbki ishlov berish, yuklab olish, tensorga o'girish
    def __init__(self):
        xy = np.loadtxt('../Data/diabetes.csv.',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=64,
                          shuffle=True)
                          
# xy_data = np.loadtxt('../Data/diabetes.csv', delimiter=',', dtype = np.float32)
# # x va y data larga ajratib chiqish (Traning data)
# x_data = torch.from_numpy(xy_data[:750, 0:-1])
# y_data = torch.from_numpy(xy_data[:750, [-1]])

# Class yordamida model qurib olish --> "Model"
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Chiziqli modellar
        self.linear1 = torch.nn.Linear(8, 6) # kirish 8 va chiqish 6
        self.linear2 = torch.nn.Linear(6, 4) # kirish 6 va chiqish 4
        self.linear3 = torch.nn.Linear(4, 1) # kirish 4 va chiqish 1
        # Aktivatsiya funksiyasi (Sigmoid&ReLU)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
    # metod yordamida forward ni belgilash    
    def forward(self, x):
        natija1 = self.relu(self.linear1(x))
        natija2 = self.relu(self.linear2(natija1))
        y_pred = self.sigmoid(self.linear3(natija2))
        return y_pred
# Bizning modelimiz
model = Model()   

# Lossni va optimizer tanlash
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 


loss_list = []
for epoch in range(4):
    for i, data in enumerate(train_loader, 0):
        # kirish ma'lumotlarini ajratib olish
        inputs, labels = data

        # forward (to'g'ri xisoblash)
        y_pred = model(inputs) 

        # natijalarni chop qilish
        loss = criterion(y_pred, labels)
        print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')
            
        # gradientni nolga tenglash,back propogation, w ni qiymatini yangilash.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

