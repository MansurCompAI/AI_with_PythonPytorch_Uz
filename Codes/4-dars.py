#Kerakli kutubxonalrni chaqirib olish
import torch

x_soat = [1.0, 2.0, 3.0]
y_baho = [2.0, 4.0, 6.0]


w = torch.tensor([1.0], requires_grad=True) #Taxminiy qiymat

# (Modelimiz)To'g'ri hisoblash uchun funksiya
def forward(x):
    return x * w


# Xatolik (Loss) ning funkisyasi
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2          

# Training dan avval
print("Bashorat (training dan avval)",  "4 soat o'qilganda:", forward(4))
# Training zanjiri (loop)
learning_rate = 0.01
for epoch in range(10):
    for x_hb_qiym, y_hb_qiym in zip(x_soat, y_baho):
        y_pred = forward(x_hb_qiym) # 1) Forward hisoblash
        l = loss(y_pred, y_hb_qiym) # 2) Loss ni hisoblash
        l.backward() # 3) backward hisoblash
        print("\tgrad: ", x_hb_qiym, y_hb_qiym, '{:.3f}'.format(w.grad.item()))
        w.data = w.data - learning_rate * w.grad.item()  #W ning qiymatini yangilash

        # w ning qiymattini yangilagach, nolga tenglashtirish
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# Traningdan so'ng
print("Bashorat (training dan keyin)",  "4 saot o'qilganda: ", forward(4).item())


