import badcrossbar
import numpy as np
import torch
import matplotlib.pyplot as plt



applied_voltages = np.ones([196, 1])
# applied_voltages = np.ones([196, 1])



torch.set_printoptions(threshold=10_000)


# w =np.full((196, 50), 10000)
w = np.random.randint(1000, 20000, size=(196, 50))
w1 = np.array(w)



n_neurons1 = 50

n_test=40

percents = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07]

err_all, sol_mean_all = np.zeros(14), np.zeros(14)
for k in range(n_test):
    print(k)
    err, sol_mean = [], []
    for probabil in percents:

        w1 = np.array(w)
        mask = np.random.binomial(n=1, p=probabil, size=[196, n_neurons1])
        #print(np.sum(mask) / 196 / n_neurons1 * 100)
        for i in range(196):
            for j in range(n_neurons1):
                if mask[i][j] == 1:
                    w1[i][j] = 1000

        # probabil1 = 0.005
        # mask1 = np.random.binomial(n=1, p=probabil1, size=[196, n_neurons1])
        # for i in range(196):
        #     for j in range(n_neurons1):
        #         if mask1[i][j] == 1:
        #             w1[i][j] = np.min(w)
        r_i = 1
        solution = badcrossbar.compute(applied_voltages, w, r_i)
        v = solution.voltages.word_line
        c = solution.currents.device
        solution1 = badcrossbar.compute(applied_voltages, w1, r_i)
        v1 = solution1.voltages.word_line
        c1 = solution1.currents.device
        v3 = np.abs(v - v1)
        c3 = c - c1
        diff = np.abs(solution1.currents.output - solution.currents.output) / solution.currents.output * 100
        # print(solution1.currents.output)
        # print(solution.currents.output)
        # print(diff)
        #print("Mean", np.mean(diff))
        # print("min", np.min(diff))
        # print("max", np.max(diff))
        print("Std", np.std(diff))
        sol_mean.append(np.mean(diff))
        err.append(np.std(diff))
    #     print(sol_mean)
    # print(sol_mean)
    sol_mean = np.array(sol_mean)
    err = np.array(err)
    err_all += err
    sol_mean_all += sol_mean
    #print(sol_mean_all)
# fig.savefig("V_I"+str(i))
# fig1.savefig("I_out" + str(i))

# with open('result' + '.txt', 'a+') as f:
#     f.write("\nMean " + str(np.mean(diff)))
#     f.write("\nMax " + str(np.max(diff)))
#     f.write("\nMin " + str(np.min(diff)))

np.save('data_mean.npy', sol_mean_all)  # save
np.save('data_std.npy', err_all)  # save
#new_num_arr = np.load('data.npy') # load
percents = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
plt.plot(percents, sol_mean_all / n_test, 'b-', linewidth=1)
plt.errorbar(percents, sol_mean_all / n_test, err_all / n_test, fmt='s', markersize=4, capsize=4,linewidth=0.6)
plt.xlabel("Stuck elements, %")
plt.ylabel("Deviation, %")
plt.grid(True)
plt.savefig("fig0")
plt.show()
