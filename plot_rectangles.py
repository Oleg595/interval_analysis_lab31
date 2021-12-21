from matplotlib import pyplot as plt
import matplotlib.patches as pth

def plotRect(ive, argmax, text):
    fig, ax = plt.subplots()
    x1_low = argmax[0] - ive
    x1_up = argmax[0] + ive
    x2_low = argmax[1] - ive
    x2_up = argmax[1] + ive
    ax.add_patch(
        pth.Rectangle((x1_low, x2_low), (x1_up - x1_low), (x2_up - x2_low),
                      linewidth=1, edgecolor='r', facecolor='none'))
    ax.plot()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(text)
    plt.show()

# correction right part

rve = 0.1392
ive = .0702

argmax = [1.06,
    0.44]

plotRect(ive, argmax, "Коррекция правой части")

# matrix correction

rve = 8.1 * 10 ** (-11)
ive = 4.53 * 10 ** (-11)

argmax = [1,
          0.5]

plotRect(ive, argmax, "Коррекция матрицы")
