import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 12})

for (file, type) in [("./slo_f_192.168.31.21_chunky.csv", "chunky"), ("./slo_f_192.168.31.21_smooth.csv", "smooth")]:
    df = pd.read_csv(file)  # .iloc[:200]

    plt.figure(figsize=(6, 3.5))
    plt.plot(df.index, df['evidence'], marker='x', linestyle='--', label=r'Evidence ($\mathit{e_r}$)', alpha=0.5)
    plt.plot(df.index, df['expected'], marker='x', linestyle='--', label='Expected ($\mathit{p_\phi}$)')
    plt.plot(df.index, df['reality'], marker='x', linestyle='--', label='Reality ($\mathit{W_\phi}$)')
    # plt.plot(df_O.index, df_O['time_ms'], marker='o', linestyle='-', label='Check Offload')

    plt.vlines([250], ymin=0, ymax=1000,
               color='grey',
               linestyle='-',
               linewidth=1.8,
               alpha=1)

    # plt.title(f'Cycle Duration for {device}')
    plt.xlabel('Cycle Iteration')
    plt.ylabel('Time consumed (ms)')
    plt.xlim(0, 420)
    plt.ylim(0.0, 1.05)
    plt.legend()
    # plt.tight_layout()

    plt.savefig(f"./eager_learning_{type}.eps", dpi=300, bbox_inches="tight", format="eps")  # default dpi is 100
    plt.show()

