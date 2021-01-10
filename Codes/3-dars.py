# Training Data(O'rgatishdagi ma'lumotlar)
x_soat = [1.0, 2.0, 3.0]
y_baho = [2.0, 4.0, 6.0]

w = 1.0  #w uchun dastalbki taxminiy qiymat


# (Modelimiz)To'g'ri hisoblash uchun funksiya
def forward(x):
    return x * w


# Xatolik (Loss) ning funkisyasi
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# Gradient uchun funksiya
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


# Training dan avval
print("Bashorat (training dan avval)",  "4 soat o'qilganda:", forward(4))

# Training zanjiri (loop)
learning_rate =0.01
for epoch in range(10):
    for x_hb_qiym, y_hb_qiym in zip(x_soat, y_baho):
        # Hosilani hisoblash
        # w ning qiymatini yangilash
        # xatolikni hisoblab progressni chop qilish
        grad = gradient(x_hb_qiym, y_hb_qiym)
        w = w - learning_rate * grad
        print("\tgrad: ", x_hb_qiym, y_hb_qiym, round(grad, 2))
        l = loss(x_hb_qiym, y_hb_qiym)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# Traningdan so'ng
print("Bashorat (training dan keyin)",  "4 saot o'qilganda: ", forward(4))

