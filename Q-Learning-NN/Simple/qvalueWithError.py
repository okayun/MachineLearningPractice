import math
import matplotlib.pyplot as plt


#filename = ["ep0_1", "ep0_3", "ep0_5", "ep0_7", "ep0_9"]

print("Please input the number of files : ", end="")
n = int(input())

# ファイルを n 個読み込む
print("Please input filenames")
filename = [input() for i in range(0, n)]

iteration = [5000 * i for i in range(1, 201)]
qvalues = []

# すべてのファイルについて, 各行の特定の列の値を読み込む
for f in filename:
    qval = []
    fop = open(f, 'r')
    for line in fop:
        dat = line.rstrip('\n')
        dat = dat.split()
        qval += [float(dat[4])]
    qvalues += [qval]
    fop.close()
    #plt.plot(iteration, error_of_mean_square, label=f)

# 平均と標準誤差を計算する

average = [0 for i in range(0, len(iteration))]
error = [0 for i in range(0, len(iteration))]


# 各イテレーションでのQの値の平均と標準偏差を計算する
for i in range(0, len(iteration)):
    # 平均を計算
    ave = 0.0
    for j in range(0, n):
        ave += qvalues[j][i]
    average[i] = ave / n

    # 標準偏差を計算
    err = 0.0
    for j in range(0, n):
        err += (qvalues[j][i] - average[i])**2
    err = math.sqrt(err / n)
    error[i] = err

plt.plot(iteration, average, label="average")
plt.errorbar(iteration, average, yerr=error, fmt='ro', ecolor='g')

#print(error)

plt.xscale("log")
plt.xlabel("Iteration")
plt.ylabel("Q value")
plt.title("ep0.3")
plt.legend(loc="best")
plt.show()

