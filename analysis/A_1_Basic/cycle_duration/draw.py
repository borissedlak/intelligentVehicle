import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 15})

for (file, device) in [("./cycle_length_192.168.31.183.csv", "AGX"), ("./cycle_length_192.168.31.205.csv", "NX")]:
    # Read the CSV file and take the first 200 rows
    df = pd.read_csv(file)  # .iloc[:200]

    # Split the DataFrame based on the 'category' column
    df_T = df[df['category'] == 'training']
    df_O = df[df['category'] == 'offloading']

    plt.figure(figsize=(6, 3.0))
    plt.plot(df_T.index, df_T['time_ms'], marker='x', linestyle='--', label='Training')
    plt.plot(df_O.index, df_O['time_ms'], marker='o', linestyle='-', label='Check Offload')

    plt.vlines([50, 100, 150], ymin=0, ymax=1000,
               color='grey',
               linestyle=':',
               linewidth=1.8,
               alpha=0.5)

    # plt.title(f'Cycle Duration for {device}')
    plt.xlabel('Cycle Iteration')
    plt.ylabel('Time consumed (ms)')
    plt.xlim(0, 200)
    plt.ylim(0, 900)
    plt.legend()
    # plt.tight_layout()

    plt.subplots_adjust(top=0.95, bottom=0.18)
    plt.savefig(f"./cycle_overhead_{device}.eps", dpi=300, format="eps")  # default dpi is 100
    plt.show()
