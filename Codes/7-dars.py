#Kerakli kutubxonalrni chaqirib olish
import torch 
import numpy as np

#Ma'lumotlarni numpy yordamida yuklab olish
xy_data = np.loadtxt('../Data/diabetes.csv', delimiter=',', dtype = np.float32)
# x va y data larga ajratib chiqish (Traning data)
x_data = torch.from_numpy(xy_data[:750, 0:-1])
y_data = torch.from_numpy(xy_data[:750, [-1]])

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
criterioin = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 

# Training
for epoch in range(15000):     # Epochlar soni 15000
    y_pred = model(x_data)  # Forward (to'g'ri xisoblash)
    loss = criterioin(y_pred, y_data)
    if epoch % 1000 == 0:# loss ni hisoblash
        print(f'epoch # {epoch} --> Loss {loss.item():.4f}')
    optimizer.zero_grad()  # Optimizerni nolga tenglab olish
    loss.backward() # Teskari hisoblash (Back prop)
    optimizer.step()  # Optimizer orqali w ni qiymatini yangilash 
print(f"\n Trainingdan so'ng test qilib ko'ramiz \n{'=' * 50}")   
# Testing data
x_test = torch.from_numpy(xy_data[752:753, 0:-1])
print(f' test uchun data == {x_test}')
test = model(x_test)
print(f"\n Natija quyidagicha \n{'=' * 50}")
print(f"Bashorat qiymat: {test.item():.4f} | diabet mavjudligi: { test.item() > 0.5}")


for i, data in enumerate(train_loader):
    # kirish ma'lumotlar (input)larni ajratib olish 
    # Variables (Tensorga) larga o'girib olish
    inputs, labels = Variable(inputs), Variable(labels)
    
    # Trainingni amalga oshirish
  
from torch.utils.data import Dataset, DataLoader
  
class CustomDataset(Dataset):
    
    # Ma'lumotlarni tayyorlash
    def __init__(self):
        
        
        
    def __getitem__(self, index):
        return 
        
        
    def __len__(self):
        return

dataset = CustomDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=64,
                          shuffle=True)

