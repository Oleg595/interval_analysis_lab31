import math
import matplotlib.pyplot as plt
import matplotlib.patches as pth
from copy import deepcopy
from plot_images import plot_tolmax

from tolsolvty import tolsolvty
import numpy as np

def plotRectangle(inf1, sup1, name1, inf2, sup2, name2, title):
    fig, ax = plt.subplots()

    x1_low = inf1[0]
    x1_high = sup1[0]
    x2_low = inf1[1]
    x2_high = sup1[1]
    ax.add_patch(
        pth.Rectangle((x1_low, x2_low), (x1_high - x1_low), (x2_high - x2_low),
                    linewidth=1, edgecolor='r', facecolor='none'))

    x1_low = inf2[0]
    x1_high = sup2[0]
    x2_low = inf2[1]
    x2_high = sup2[1]
    ax.add_patch(
        pth.Rectangle((x1_low, x2_low), (x1_high - x1_low), (x2_high - x2_low),
                      linewidth=1, edgecolor='g', facecolor='none'))

    ax.plot()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend([name1, name2])
    plt.title(title)
    plt.show()

def rightCorrection(inf_A1, sup_A1, inf_b1, sup_b1):
    inf_A = deepcopy(inf_A1)
    sup_A = deepcopy(sup_A1)
    inf_b = deepcopy(inf_b1)
    sup_b = deepcopy(sup_b1)
    [tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
    iterations = [tolmax]
    while tolmax < -10 ** (-10):
        for i in range(len(inf_b)):
            inf_b[i][0] += tolmax - 0.1
            sup_b[i][0] -= tolmax - 0.1
        [tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
        iterations.append(tolmax)
    print(tolmax)
    print(argmax)
    plot_tolmax(iterations)
    return inf_b, sup_b, argmax

def new_A(inf_A, sup_A, tolmax, argmax, envs):
    index = 0
    while np.array_equal(inf_A[int(envs[index][0] - 1)], sup_A[int(envs[index][0] - 1)]):
        index += 1
    index = int(envs[index][0] - 1)
    for i in range(len(inf_A[index])):
        r = (sup_A[index][i] - inf_A[index][i]) / 2
        if -tolmax > r * argmax[i]:
            sup_A[index][i] -= r
            inf_A[index][i] += r
            tolmax += r * argmax[i]
            print(tolmax)
        else:
            sup_A[index][i] += tolmax
            inf_A[index][i] -= tolmax
            break
    '''for index in range(len(inf_A)):
        rad1 = (sup_A[index][0] - inf_A[index][0]) / 4
        rad2 = (sup_A[index][1] - inf_A[index][1]) / 4
        mid1 = (sup_A[index][0] + inf_A[index][0]) / 2
        mid2 = (sup_A[index][1] + inf_A[index][1]) / 2
        inf_A[index][0] = mid1 - rad1
        sup_A[index][0] = mid1 + rad1
        inf_A[index][1] = mid2 - rad2
        sup_A[index][1] = mid2 + rad2'''
    return inf_A, sup_A

def matrixCorrection(inf_A1, sup_A1, inf_b1, sup_b1):
    inf_A = deepcopy(inf_A1)
    sup_A = deepcopy(sup_A1)
    inf_b = deepcopy(inf_b1)
    sup_b = deepcopy(sup_b1)
    [tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
    iterations = [tolmax]
    while tolmax < -10 ** (-10):
        inf_A, sup_A = new_A(inf_A, sup_A, tolmax, argmax, envs)
        if len(inf_A) == 0:
            return [], [], []
        [tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
        iterations.append(tolmax)
    print(tolmax)
    print(argmax)
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
[tolmax, argmax, envs, ccode] = tolsolvty(inf_A, sup_A, inf_b, sup_b)
plotRectangle(inf, sup, 'Коррекция', [-tolmax / 2, -tolmax / 2], [tolmax / 2, tolmax / 2], 'Допусковое множество', "Коррекция правой части")
plt.ion()
plt.show()
