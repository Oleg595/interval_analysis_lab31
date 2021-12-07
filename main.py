import math
import matplotlib.pyplot as plt
import matplotlib.patches as pth
from copy import deepcopy
from plot_images import plot_tolmax

from tolsolvty import tolsolvty
import numpy as np

def plotRectangle(inf, sup, text):
    fig, ax = plt.subplots()

    x1_low = inf[0]
    x1_high = sup[0]
    x2_low = inf[1]
    x2_high = sup[1]
    ax.add_patch(
        pth.Rectangle((x1_low, x2_low), (x1_high - x1_low), (x2_high - x2_low),
                    linewidth=1, edgecolor='r', facecolor='none'))

    ax.plot()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(text)
    plt.show()

def rightCorrection(inf_A1, sup_A1, inf_b1, sup_b1):
    inf_A = deepcopy(inf_A1)
    sup_A = deepcopy(sup_A1)
    inf_b = deepcopy(inf_b1)
    sup_b = deepcopy(sup_b1)
    [tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
    iterations = [tolmax]
    while tolmax < -10 ** (-10):
        inf_b[int(envs[0][0]) - 1][0] += tolmax - 0.1
        sup_b[int(envs[0][0]) - 1][0] -= tolmax - 0.1
        [tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
        iterations.append(tolmax)
    plot_tolmax(iterations)
    return inf_b, sup_b, argmax

def new_A(inf_A, sup_A, tolmax, envs, argmax):
    tolmax = math.fabs(tolmax)
    count = 0
    for i in range(len(envs)):
        index = int(envs[i][0]) - 1
        rad1 = (sup_A[index][0] - inf_A[index][0]) / 2
        rad2 = (sup_A[index][1] - inf_A[index][1]) / 2
        if rad1 == rad2 == 0:
            if (count + 1) == len(envs):
                print("not answer")
                return [], []
            else:
                count += 1
            continue
        if rad1 == 0:
            arr = [math.fabs(sup_A[index][1]) * math.fabs(argmax[1]), math.fabs(inf_A[index][1]) * math.fabs(argmax[1])]
            e = tolmax / max(arr)
            if rad2 >= e:
                sup_A[index][1] -= e
                inf_A[index][1] += e
                return inf_A, sup_A
            else:
                e = rad2
                sup_A[index][1] -= e
                inf_A[index][1] += e
                continue
        if rad2 == 0:
            arr = [math.fabs(sup_A[index][0]) * math.fabs(argmax[0]), math.fabs(inf_A[index][0]) * math.fabs(argmax[0])]
            e = tolmax / max(arr)
            if rad1 >= e:
                sup_A[index][0] -= e
                inf_A[index][0] += e
                return inf_A, sup_A
            else:
                e = rad1
                sup_A[index][0] -= e
                inf_A[index][0] += e
                continue
        m = rad2 / rad1
        arr = [math.fabs(argmax[0] * inf_A[index][0] + argmax[1] * inf_A[index][1]),
               math.fabs(argmax[0] * inf_A[index][0] + argmax[1] * sup_A[index][1]),
               math.fabs(argmax[0] * sup_A[index][0] + argmax[1] * inf_A[index][1]),
               math.fabs(argmax[0] * sup_A[index][0] + argmax[1] * sup_A[index][1])]
        e = tolmax / max(arr)
        if rad1 >= e:
            sup_A[index][0] -= e
            inf_A[index][0] += e
            sup_A[index][1] -= m * e
            inf_A[index][1] += m * e
            return inf_A, sup_A
        else:
            e = rad1
            sup_A[index][0] -= e
            inf_A[index][0] += e
            sup_A[index][1] -= m * e
            inf_A[index][1] += m * e
    return inf_A, sup_A

def matrixCorrection(inf_A1, sup_A1, inf_b1, sup_b1):
    inf_A = deepcopy(inf_A1)
    sup_A = deepcopy(sup_A1)
    inf_b = deepcopy(inf_b1)
    sup_b = deepcopy(sup_b1)
    [tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
    iterations = [tolmax]
    while tolmax < -10 ** (-10):
        inf_A, sup_A = new_A(inf_A, sup_A, tolmax, envs, argmax)
        if len(inf_A) == 0:
            return [], [], []
        [tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
        iterations.append(tolmax)
    plot_tolmax(iterations)
    return inf_A, sup_A, argmax

def var_answer(inf_A, sup_A, inf_b, sup_b, argmax):
    min_radius = 100.
    for i in range(len(sup_A)):
        ri = (sup_b[i][0] - inf_b[i][0]) / 2
        sum_sup = (sup_b[i][0] + inf_b[i][0]) / 2
        sum_inf = (sup_b[i][0] + inf_b[i][0]) / 2
        sum = 0
        for j in range(len(sup_A[0])):
            sum_sup -= argmax[j][0] * sup_A[i][j]
            sum_inf -= argmax[j][0] * inf_A[i][j]
            sum += max([math.fabs(sup_A[i][j]), math.fabs(inf_A[i][j])])
            sum_inf, sum_sup = min([sum_inf, sum_sup]), max([sum_inf, sum_sup])
        ri += max([math.fabs(sum_inf), math.fabs(sum_sup)])
        if sum == 0:
            ri = min_radius
        else:
            ri /= sum
        if min_radius > ri:
            min_radius = ri
    inf_answer = []
    sup_answer = []
    for i in range(len(inf_A[0])):
        inf_answer.append(-min_radius)
        sup_answer.append(min_radius)
    return inf_answer, sup_answer


inf_A = np.array([[1, .75],
                  [1, -3],
                  [.75, 0]], dtype=float)
sup_A = np.array([[3, 3.25],
                  [1, -1],
                  [1.25, 0]], dtype=float)
inf_b = np.array([[2], [0], [1]], dtype=float)
sup_b = np.array([[4], [0], [4]], dtype=float)

[tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
print(tolmax)

inf_b1, sup_b1, argmax_b = rightCorrection(inf_A, sup_A, inf_b, sup_b)
print(inf_b1)
print(sup_b1)
inf_A1, sup_A1, argmax_A = matrixCorrection(inf_A, sup_A, inf_b, sup_b)
print(inf_A1)
print(sup_A1)

if len(inf_A1) == 0:
    print("не получается корректировать матрицу")

inf, sup = var_answer(inf_A, sup_A, inf_b1, sup_b1, argmax_b)
plotRectangle(inf, sup, "Коррекция правой части")

if len(inf_A1) != 0:
    inf, sup = var_answer(inf_A1, sup_A1, inf_b, sup_b, argmax_A)
    plotRectangle(inf, sup, "Коррекция матрицы")
