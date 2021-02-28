import numpy as np
from random import uniform

# ------------------------------
MIN, MAX = 0, 20
a0, a1, a2, a3 = 1, 2, 2, 3

# -------------------------------
X = np.empty((8, 3), dtype=float)  # X - масив значень факторів(усі точки X)
Y = np.empty(8)  # Y -  масив значень функції відгуку (рівняння регресії Y = a0 + a1X1 + a2X2 + a3X3)
X0 = np.empty(3)  # X0 - масив нульових рівнів для кожного фактору
DX = np.empty(3)  # DX - масив інтервалів зміни фактора
XNormalized = np.empty((8, 3), dtype=float)  # масив нормалізованих значень факторів

# генерація плану експерименту(усіх точнок x)(випадкові числа)
for i in range(8):
    for j in range(3):
        X[i, j] = uniform(MIN, MAX)
# обчислення значень функції відгуків
for i in range(8):
    Y[i] = a0 + a1 * X[i, 0] + a2 * X[i, 1] + a3 * X[i, 2]

# обчислення нульових рівнів та інтервалів зміни факторів
for i in range(3):
    X0[i] = (X[:, i].max() + X[:, i].min()) / 2
    DX[i] = X[:, i].max() - X0[i]
    # xmax=X[:,i].max() #омтимізовоно(не треба двічі обчислювати xmax)
    # X0[i]=(xmax+X[:,i].min())/2
    # DX[i]=xmax-X0[i]

Y_et = a0 + a1 * X0[0] + a2 * X0[1] + a3 * X0[2]  # Y_et - Y-еталонне
# нормалізація значень факторів
for i in range(8):
    for j in range(3):
        XNormalized[i, j] = (X[i, j] - X0[j]) / DX[j]

dY = 999999  # різниця між Y таY_et
number = -1  # номер точки, що щадовольняє критерій
# знаходження номеру точки, що задовольняє критерій опримальності
# Yэт
# пошук точки в якій значення Y найближче справа до Y_et
for i in range(8):
    if Y[i] - Y_et < dY and Y[i] - Y_et > 0:
        dY = Y[i] - Y_et
        number = i

# функції відгуку, отримана підставленням замість Хi
# значення факторів ту точку, що задовольняє критерію вибору
Y2 = a0 + a1 * X[number, 0] + a2 * X[number, 1] + a3 * X[number, 2]

print("X:\n", X)
print("Y:\n", Y)
print("X0: \n", X0)
print("T_et = ", Y_et)
print("XNormalized: \n", XNormalized.round(4))
print("number = ", number)
