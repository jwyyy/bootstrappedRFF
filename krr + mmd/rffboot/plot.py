import numpy as np
import matplotlib.pyplot as plt


def plot_true_vs_boot(q, label, s_vec, error_infty, true_q, sd, boot_q, idx=0, ymax=1.0):
    plt.figure()
    if ymax > 0:
        plt.ylim(0.0, ymax)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.plot(s_vec, true_q, label=f"true, error infty={error_infty}" if error_infty > 0 else "true",
             linestyle='solid', linewidth=1.0, color="seagreen")
    plt.plot(s_vec, boot_q, label="boot", linestyle='solid', linewidth=1.0, color="red")
    plt.plot(s_vec, [boot_q[i] + 3 * sd[i] for i in range(len(s_vec))],
             label="boot +3 SD",
             linestyle='solid', linewidth=1.0, color="pink")
    plt.plot(s_vec, [boot_q[i] - 3 * sd[i] for i in range(len(s_vec))],
             label="boot -3 SD",
             linestyle='solid', linewidth=1.0, color="pink")

    plt.hlines(y=0.0, xmin=s_vec[0], xmax=s_vec[-1], colors="black", linestyles=":")

    if idx >= 0:
        plt.plot(s_vec[idx:], [boot_q[idx] * np.sqrt(s_vec[idx] / s_vec[i]) for i in range(idx, len(s_vec))],
                 label=f"basic extrapolation (start from {s_vec[idx]})",
                 linestyle='solid', linewidth=1.0, color="olive")

    legend = plt.legend(fontsize=10, frameon=False, loc=0)
    legend.get_frame().set_facecolor('none')

    plt.savefig("{0}_level={1}_plot.png".format(q, label), format='png')
    plt.savefig("{0}_level={1}_plot.eps".format(q, label), format='eps')
    plt.close()


def plot_results(q, folder, label, s_vec, err_infty, true_q, sd, boot_q, idx=0, relative=False, ymax=0.1, band_w=1,
                 legend=True):
    if relative:
        true_q = np.array(true_q) / err_infty
        boot_q = np.array(boot_q) / err_infty
        sd = np.array(sd) / err_infty

    plt.figure(figsize=(8.5, 5))

    if ymax > 0:
        plt.ylim(-ymax * 0.025, ymax)

    plt.plot(s_vec, true_q, 'k--s', lw=4, label=f"true {q}-quantile")
    plt.plot(s_vec, boot_q, '-o', lw=4, color='#3182bd', label='avg. bootstrap quantile')

    if idx >= 0:
        extrapolation_q = [boot_q[idx] * np.sqrt(s_vec[idx] / s_vec[i]) for i in range(idx, len(s_vec))]
        plt.plot(
            s_vec[idx:],
            extrapolation_q,
            lw=4,
            color='#de2d26',
            label=f"extrapolation ($\pm{int(band_w)}$ sd)",
        )

        extrapolation_up = [(boot_q[idx] + band_w * sd[idx]) * np.sqrt(s_vec[idx] / s_vec[i]) for i in
                            range(idx, len(s_vec))]
        extrapolation_lw = [(boot_q[idx] - band_w * sd[idx]) * np.sqrt(s_vec[idx] / s_vec[i]) for i in
                            range(idx, len(s_vec))]

        plt.fill_between(s_vec[idx:], extrapolation_lw, extrapolation_up, facecolor='b', color='#de2d26', alpha=0.1)

        plt.plot(s_vec[idx:], extrapolation_up, lw=2, color='#de2d26', alpha=0.3)
        plt.plot(s_vec[idx:], extrapolation_lw, lw=2, color='#de2d26', alpha=0.3)

    # plt.ylabel("relative error %" if relative else "error", fontsize=14)
    # plt.xlabel("random feature number", fontsize=14)
    if legend: plt.legend(loc="best", fontsize=22)
    # list(range(s_vec[idx], s_vec[-1] + 100, 200))
    # plt.xticks(list(range(100, 700, 100)))
    plt.tick_params(axis="y", labelsize=22)
    plt.tick_params(axis="x", labelsize=22)

    plt.tight_layout()

    plt.savefig(folder + ("_r_" if relative else "_") + "{0}_level={1}_plot.pdf".format(q, label), format='pdf')
    plt.close()
