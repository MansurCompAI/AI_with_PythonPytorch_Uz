#Kerakli kutubconalrni chaqirib olish
import torch 
import numpy as np

#Ma'lumotlarni numpy yordamida yuklab olish
xy_data = np.loadtxt('../Data/diabetes.csv', delimiter=',', dtype = np.float32)
# x va y data larga ajratib chiqish
x_data = torch.from_numpy(xy_data[:, 0:-1])
y_data = torch.from_numpy(xy_data[:, [-1]])

# Class yordamida model qurib olish --> "Model"
class Model()
