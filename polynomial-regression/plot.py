import json
import os
import matplotlib.pyplot as plt

N = [2, 5, 10, 20, 50, 100, 200]
degrees = [1, 2, 4, 8, 16, 32, 64]
sd = [0.05, 0.2]
os.makedirs('./plots', exist_ok=True)


def get_error(file_name):
    f = open(file_name)
    data = json.load(f)
    f.close()
    return data


def plot_comparison(data1, data2, folder):
    for n in N:
        E_in1 = {0.05: [], 0.2: []}
        E_out1 = {0.05: [], 0.2: []}
        E_in2 = {0.05: [], 0.2: []}
        E_out2 = {0.05: [], 0.2: []}
        for d in degrees:
            for s in sd:
                var = str((n, d, s))
                E_in1[s].append(data1[var][0])
                E_out1[s].append(data1[var][1])
                E_in2[s].append(data2[var][0])
                E_out2[s].append(data2[var][1])
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(degrees, E_in1[0.05], color='blue', label='$E_{in}, \sigma=0.05$')
        ax1.plot(degrees, E_out1[0.05], color='red', label='$E_{out}, \sigma=0.05$')
        ax1.plot(degrees, E_in1[0.2], color='blue', linestyle='dotted', label='$E_{in}, \sigma=0.2$')
        ax1.plot(degrees, E_out1[0.2], color='red', linestyle='dotted', label='$E_{out}, \sigma=0.2$')
        ax2.plot(degrees, E_in2[0.05], color='blue', label='$E_{in}, \sigma=0.05$')
        ax2.plot(degrees, E_out2[0.05], color='red', label='$E_{out}, \sigma=0.05$')
        ax2.plot(degrees, E_in2[0.2], color='blue', linestyle='dotted', label='$E_{in}, \sigma=0.2$')
        ax2.plot(degrees, E_out2[0.2], color='red', linestyle='dotted', label='$E_{out}, \sigma=0.2$')

        fig.suptitle('Error for sample size ($N$) = {}'.format(n), fontsize=18)
        ax1.legend()
        ax2.legend()
        ax1.set_xlim([0, 64])
        ax2.set_xlim([0, 64])
        max_error1 = max(max(E_in1[0.05], E_out1[0.05], E_in1[0.2], E_out1[0.2]))
        ax1.set_ylim([0, max_error1])
        max_error2 = max(max(E_in2[0.05], E_out2[0.05], E_in2[0.2], E_out2[0.2]))
        ax2.set_ylim([0, max_error2])
        ax1.set_title("a) Non-regularized", fontsize=14)
        ax2.set_title("b) Regularized", fontsize=14)
        fig.text(0.5, 0.04, "Model complexity ($d$)", ha='center', va='center', fontsize=18)
        fig.text(0.06, 0.5, "Error", ha='center', va='center', rotation='vertical', fontsize=18)
        title = 'comparison_sample_size_{}'.format(n)
        fig.savefig(folder + title + '.png')
        fig.show()

    for d in degrees:
        E_in1 = {0.05: [], 0.2: []}
        E_out1 = {0.05: [], 0.2: []}
        E_in2 = {0.05: [], 0.2: []}
        E_out2 = {0.05: [], 0.2: []}
        for n in N:
            for s in sd:
                var = str((n, d, s))
                E_in1[s].append(data1[var][0])
                E_out1[s].append(data1[var][1])
                E_in2[s].append(data2[var][0])
                E_out2[s].append(data2[var][1])

        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(N, E_in1[0.05], color='blue', label='$E_{in}, \sigma=0.05$')
        ax1.plot(N, E_out1[0.05], color='red', label='$E_{out}, \sigma=0.05$')
        ax1.plot(N, E_in1[0.2], color='blue', linestyle='dotted', label='$E_{in}, \sigma=0.2$')
        ax1.plot(N, E_out1[0.2], color='red', linestyle='dotted', label='$E_{out}, \sigma=0.2$')
        ax2.plot(N, E_in2[0.05], color='blue', label='$E_{in}, \sigma=0.05$')
        ax2.plot(N, E_out2[0.05], color='red', label='$E_{out}, \sigma=0.05$')
        ax2.plot(N, E_in2[0.2], color='blue', linestyle='dotted', label='$E_{in}, \sigma=0.2$')
        ax2.plot(N, E_out2[0.2], color='red', linestyle='dotted', label='$E_{out}, \sigma=0.2$')

        fig.suptitle('Error for model complexity ($d$) = {}'.format(d), fontsize=18)
        ax1.legend()
        ax2.legend()
        ax1.set_xlim([0, 200])
        ax2.set_xlim([0, 200])
        max_error1 = max(max(E_in1[0.05], E_out1[0.05], E_in1[0.2], E_out1[0.2]))
        ax1.set_ylim([0, max_error1])
        max_error2 = max(max(E_in2[0.05], E_out2[0.05], E_in2[0.2], E_out2[0.2]))
        ax2.set_ylim([0, max_error2])
        ax1.set_title("a) Non regularized", fontsize=14)
        ax2.set_title("b) Regularized", fontsize=14)
        fig.text(0.5, 0.04, "Sample size ($N$)", ha='center', va='center', fontsize=18)
        fig.text(0.06, 0.5, "Error", ha='center', va='center', rotation='vertical', fontsize=18)
        title = 'comparison_complexity_{}'.format(d)
        fig.savefig(folder + title + '.png')
        fig.show()


if __name__ == '__main__':
    error_nreg = get_error('output/non_regularized.json')
    error_reg = get_error('output/regularized.json')
    plot_comparison(error_nreg, error_reg, folder='plots/')
