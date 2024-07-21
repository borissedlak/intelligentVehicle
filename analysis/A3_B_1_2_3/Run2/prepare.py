import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 12})

color_service_dict = {"QR-1": 'red', "CV-2": 'green', "CV-3": 'blue', "CV-4": 'orange'}

for (file, device) in [("./slo_f_192.168.31.21.csv", "Laptop"), ("./slo_f_192.168.31.183.csv", "AGX")]:
    df = pd.read_csv(file)

    df_slo_f = df[(df['v1'].isna()) & (df['v2'].isna())]
    df_hw = df[(df['v1'].notna()) & (df['v2'].isna())]
    df_offload = df[(df['v1'].notna()) & (df['v2'].notna())]

    # df_slo_f["category"] = "SLO-F"
    # df_hw["category"] = "HW"
    # df_offload["category"] = "OFF"

    # df_hw['target_device'] = df_hw.iloc[:, 4]
    # df_hw['target_services'] = df_hw.iloc[:, 5]
    # df_hw['origin_load_p'] = df_hw.iloc[:, 6]
    # df_hw['target_conv_load'] = df_hw.iloc[:, 7]
    #
    # df_hw['target_model_name'] = df_hw.iloc[:, 4]
    # df_hw['slo_local_estimated_initial'] = df_hw.iloc[:, 5]
    # df_hw['slo_local_estimated_offload'] = df_hw.iloc[:, 6]
    # df_hw['slo_target_estimated_initial'] = df_hw.iloc[:, 7]
    # df_hw['slo_target_estimated_offload'] = df_hw.iloc[:, 8]

    df_slo_f.to_csv(f"{device}.csv", index=False)
    df_slo_f = pd.read_csv(f"{device}.csv")
    df_slo_f['timestamp'] = pd.to_datetime(df_slo_f['timestamp'])

    plt.figure(figsize=(6, 3.5))

    grouped = df_slo_f.groupby('service')
    subsets = {category: group for category, group in grouped}
    for service_id, subset in subsets.items():
        plt.plot(subset['timestamp'], subset['reality'], marker='x', linestyle='--', label=service_id + '($\mathit{W_\phi}$)',
                 color=color_service_dict[service_id])

    # plt.vlines([pd.to_datetime('2024-07-05 23:15:52'),
    #             pd.to_datetime('2024-07-05 23:17:32')], ymin=0, ymax=1000,
    #            color='black',
    #            linestyle='-',
    #            linewidth=1.8,
    #            alpha=0.9)

    start_date = pd.to_datetime('2024-07-05 23:14:49')
    end_date = pd.to_datetime('2024-07-05 23:19:19')
    plt.xlim(start_date, end_date)

    # plt.xlabel('Cycle Iteration')
    plt.ylabel('SLO fulfillment ($\mathit{W_\phi}$)')
    # plt.xticks([(pd.to_datetime('2024-07-05 23:15:52'), "abcd"),
    #             pd.to_datetime('2024-07-05 23:17:32')])
    plt.xticks(ticks=[start_date, pd.to_datetime('2024-07-05 23:15:52'), pd.to_datetime('2024-07-05 23:17:32'),
                      pd.to_datetime('2024-07-05 23:18:33')], labels=['0s', '30s', '90s', '120s'])
    # plt.xticks(rotation=90)
    # plt.xlim(0, 420)
    plt.ylim(0.0, 1.05)
    plt.legend()
    # plt.tight_layout()

    plt.savefig(f"./slo_f_{device}.eps", dpi=300, bbox_inches="tight", format="eps")  # default dpi is 100
    plt.show()
