import matplotlib.pyplot as plt
import os
import time
import numpy as np

#directories = list(filter(lambda x: "gen" not in x and "indepth" not in x, os.listdir("data_no_av_pool_start")))


while True:
    #for directory in directories:
    file = open("47", "r")
    dist = []
    distj = []
    distk = []
    gens = []
    for line in file:
        if "," in line:
            dist.append(float(line.split(",")[1]))
            distj.append(float(line.split(",")[5]))
            distk.append(float(line.split(",")[3]))
            gens.append(int(line.split(",")[0]))
    file.close()

    dist_i = []
    dist_j = []
    a = 10
    for i in range(0, len(dist)-a, a):
        dist_i.append(0)
        dist_j.append(0)
        for j in range(a):
            dist_i[-1] += dist[i+j]
            dist_j[-1] += distj[i+j]
        dist_i[-1] /= a



    t = gens
    data1 = dist
    data2 = distj
    data3 = distk

    t_increase_per_gen = (data3[-1] - data3[0]) / t[-1]
    time_to_double_gens = (data3[0] + t_increase_per_gen * t[-1] * 1.5) * t[-1]
    print(time_to_double_gens / 60 / 60 / 24)


    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('gen')
    ax1.set_ylabel('fitness', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    time.sleep(10)