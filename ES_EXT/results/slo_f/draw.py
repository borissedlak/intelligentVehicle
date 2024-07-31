import pandas as pd
from matplotlib import pyplot as plt

df_CV = {"name": "CV", "data": [pd.read_csv('./slo_f_CV_AGX_MAX.csv'), pd.read_csv('./slo_f_CV_AGX_LIM.csv'),
                                pd.read_csv('./slo_f_CV_NX_MAX.csv'), pd.read_csv('./slo_f_CV_NX_LIM.csv')]}
df_QR = {"name": "QR", "data": [pd.read_csv('./slo_f_QR_AGX_MAX.csv'), pd.read_csv('./slo_f_QR_AGX_LIM.csv'),
                                pd.read_csv('./slo_f_QR_NX_MAX.csv'), pd.read_csv('./slo_f_QR_NX_LIM.csv')]}
df_LI = {"name": "LI", "data": [pd.read_csv('./slo_f_LI_AGX_MAX.csv'), pd.read_csv('./slo_f_LI_AGX_LIM.csv'),
                                pd.read_csv('./slo_f_LI_NX_MAX.csv'), pd.read_csv('./slo_f_LI_NX_LIM.csv')]}

device_color_dict = [(r'$AGX_+$', 'chocolate'), (r'$AGX_-$', 'dimgray'), (r'$NX_+$', 'firebrick'), (r'$NX_-$', 'steelblue')]
plt.rcParams.update({'font.size': 12})

for service in [df_CV, df_QR, df_LI]:
    fig, ax = plt.subplots()

    for index, device in enumerate(service["data"]):
        plt.plot(device.index, device['pv'], color=device_color_dict[index][1], label=device_color_dict[index][0])

    fig.set_size_inches(3.0, 3.2)
    ax.set_ylim(0.0, 1.045)
    ax.set_xlabel('AIF Cycle Iteration')
    ax.set_ylabel('SLO Fulfillment Rate')
    ax.set_xticks([0,10,20,30,40,50])
    ax.set_xlim(0, 30 if service['name'] == 'CV' else 50)
    ax.legend()

    # Show the plot
    plt.savefig(f"figures/slo_f_{service['name']}.eps", dpi=600, bbox_inches="tight", format="eps")  # default dpi is 100
    plt.show()

# plt.plot(x, df_slo_change['ra'], color='red', label="RA SLOs")

# df_slo_change['change_in_config'] = ((df_slo_change['pixel'] != df_slo_change['pixel'].shift(1))
#                                      & (df_slo_change['fps'] != df_slo_change['fps'].shift(1)))

# pos_dict = [(0.95, 1.02), (6.0, 0.87), (8.0, 1.02)]
# i_text = 0
# first_label = True
# for index, row in df_slo_change.iterrows():
#     if row['change_in_config']:
#         if index > 5:
#             plt.scatter(index + 1, row['pv'], marker='o', color='blue', label="Conf Change" if first_label else None)
#             plt.scatter(index + 1, row['ra'], marker='o', color='blue')
#             first_label = False
# print(pos_dict[i_text])
# plt.text(pos_dict[i_text][0], pos_dict[i_text][1], s=f"{row['fps']} fps", color='black', fontsize=10)
# i_text += 1

# plt.scatter([3, 3], [df_slo_change.iloc[2]['pv'], df_slo_change.iloc[2]['ra']], marker='*', color='orange',
#             label="SLO Change")
