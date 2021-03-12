#Kerakli kutubconalrni chaqirib olish
import torch 
import numpy as np

#Ma'lumotlarni numpy yordamida yuklab olish
xy_data = np.loadtxt('../Data/diabetes.csv', delimiter=',', dtype = np.float32)
# x va y data larga ajratib chiqish
x_data = torch.from_numpy(xy_data[:, 0:-1])
y_data = torch.from_numpy(xy_data[:, [-1]])

# Class yordamida model qurib olish --> "Model"
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Chiziqli modellar
        self.linear1 = torch.nn.Linear(8, 6) # kirish 8 va chiqish 6
        self.linear2 = torch.nn.Linear(6, 4) # kirish 6 va chiqish 4
        self.linear3 = torch.nn.Linear(4, 1) # kirish 4 va chiqish 1
        # Aktivatsiya funksiyasi (Sigmoid)
        self.sigmoid = torch.nn.Sigmoid()
    # metod yordamida forward ni belgilash    
    def forward(self, x):
        natija1 = self.sigmoid(self.linear1(x))
        natija2 = self.sigmoid(self.linear2(natija1))
        y_pred = self.sigmoid(self.linear3(natija2))
        return y_pred
# Bizning modelimiz
model = Model()    
    
        
