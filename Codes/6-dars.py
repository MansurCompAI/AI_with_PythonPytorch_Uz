#Kerakli kutubxonalarni chaqirib olish 
import torch 
#Ma'lumotlarni tensor ko'rinishida yuklab olish
x_soat = torch.Tensor([[1.],
                       [2.],
                       [3.],
                       [4.]])
y_ikkilik = torch.Tensor([[0.],
                       [0.],
                       [1.],
                       [1.]])

#(1) Class yordamida model qurib olish --> "Model"
class Model(torch.nn.Module):
    def __init__(self):
        #Bu yerda nn.Module bu yerda super class(Pytorch)
        super().__init__()
        #torch.nn.Linear(#kirish, #chiqish) chiziqli model
        self.linear = torch.nn.Linear(1,1) #1ta kirish & 1ta chiqish
    #Metod yordamida to'g'ri hisoblash arxitikturasini kiritamiz(forward pass)    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
#Bizning model
model=Model()
# print(model)
#(2) Loss va optimizer larni tanlab olish 
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#(3) Training(3.1),   Backward(3.2), Step(3.3)
#(3.1)-->Training
for epoch in range(1000):    #Epochlar soni 1000
    y_pred = model(x_soat)
    #Loss|||xatolikni hisoblash va chop qilish
    loss = criterion(y_pred, y_ikkilik)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    optimizer.zero_grad() #Har bir epoch uchun grad ni 0 ga tenglashtirib olish
    #(3.2)-->Backpropagation|||Teskari hisoblash
    loss.backward()
    #(3.3)--> Step||| w ning qiymatini  yangilash
    optimizer.step()
#Bashorat uchun qiymat||| Ushbu qiymatimiz ham tensor bo'lishi kerak    
print(f"\n Trainingdan so'ng bashorat qilib ko'ramiz \n{'=' * 50}")
# 1 soat uchun bashorat
hour_var = model(torch.tensor([[1.0]]))
print(f"1 soat o'qilganda imtihondan o'ta olish: {hour_var.item():.4f} |  50% dan yuqori: {hour_var.item() > 0.5}")
# 7 soat uchun bashorat
hour_var = model(torch.tensor([[7.0]]))
print(f"7 soat o'qilganda imtihondan o'ta olish: {hour_var.item():.4f} |  50% dan yuqori : { hour_var.item() > 0.5}")


