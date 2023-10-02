import pandas as pd
from pathlib import Path

root_path = Path(__file__).absolute().parent
save_dir_path = f"{root_path}\datasets\\no_goal"
save_data_path = "\S11-S12\S11 to S12"

dir_path = f"{root_path}\datasets\goal_2"
data_path = "\S11-S12\S11 to S12.pkl"

df = pd.read_pickle(dir_path + data_path)

print("--load done--")
print('df length:', df.shape[0])

observations = df["observations"]

print("--start removing goal context--")
for idx, obs in enumerate(observations):
    observations[idx] = obs[:10]
    if idx % 10000 == 0:
        print('--', idx, '--')

print("--done removing goal context--")

print('obs length:', observations.shape[0])
print(observations.iloc[-1])

df["observations"] = observations

print("--save--")
df.to_pickle(save_dir_path + save_data_path + '.pkl')
df.to_csv(save_dir_path + save_data_path + '.csv')
print("--save done--")

