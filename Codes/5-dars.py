#Kerakli kutubxonalarni chaqirib olish 
import torch 
import numpy as np
#Ma'lumotlarni tensor ko'rinishida yuklab olish
x_soat = torch.Tensor([[1.0],
                       [2.0],
                       [3.0]])
y_baho = torch.Tensor([[2.0],
                       [4.0],
                       [6.0]])

#(1) Class yordamida model qurib olish --> "Model"
class Model(torch.nn.Module):
    def __init__(self):
        #Bu yerda torch.nn.Module bu yerda super class(Pytorch)
        super().__init__()
        #torch.nn.Linear(#kirish, #chiqish) chiziqli model
        self.linear = torch.nn.Linear(1,1) #1ta kirish & 1ta chiqish
    #Metod yordamida to'g'ri hisoblash funksiyasini kiritamiz(forward pass)    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
#Bizning model    
model=Model()
# print(model)
#(2) Loss va optimizer larni tanlab olish 
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#(3) Training(3.1),   Backward(3.2), Step(3.3)
#(3.1)-->Training
for epoch in range(500):    #Epochlar soni 500
    y_pred = model(x_soat)
    #Loss|||xatolikni hisoblash va chop qilish
    loss = criterion(y_pred, y_baho)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    optimizer.zero_grad() #Har bir epoch uchun grad ni 0 ga tenglashtirib olish
    #(3.2)-->Backpropagation|||Teskari hisoblash
    loss.backward()
    #(3.3)--> Step||| w ning qiymatini  yangilash
    optimizer.step()
#Bashorat uchun qiymat||| Ushbu qiymatimiz ham tensor bo'lishi kerak    
soat_test = torch.Tensor([[4.]])
print("Bashorat (training dan keyin),  4 saot o'qilganda:", model.forward(soat_test).data[0][0].item())


