import matplotlib.pyplot as plt
import os

directories = list(filter(lambda x: "gen" not in x and "indepth" not in x, os.listdir("data")))



for directory in directories: 
    file = open("data/" + directory, "r")
    dist = []
    for line in file: 
        dist.append(float(line.split(",")[1]))
    file.close()

    dist_i = []
    a = 1
    for i in range(0, len(dist)-a, a): 
        dist_i.append(0)
        for j in range(a): 
            dist_i[-1] += dist[i+j]
        dist_i[-1] /= a

    plt.plot(dist_i)
plt.show()
plt.savefig("my_plot.png")
