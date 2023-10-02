import pandas as pd
import collections
from pathlib import Path
# from scripts.data_collection.State_class_rdc import*

root_path = Path(__file__).absolute().parent

move = "S10-S11"
name = "S10 to S11"
dir_path = f"{root_path}\datasets\\no_goal"
data_path = f"\\{move}\\{name}.pkl"

save_dir_path = f"{root_path}\datasets\\classifier"
save_data_path = f"\\{move}\\{name}"

df = pd.read_pickle(dir_path + data_path)

df_dict = collections.defaultdict(list)

print("--load done--")
print('df length:', df.shape[0])

observations = df["observations"]

print("--start dataset conversion--")
for obs in observations:
    for idx,member in enumerate(obs):
        df_dict["state[" + str(idx) + "]"].append(obs[idx])
    df_dict['skill'].append(move)

df = pd.DataFrame.from_dict(df_dict)
print("--end dataset conversion--")

print("--saving--")
df.to_pickle(save_dir_path + save_data_path + '.pkl')
df.to_csv(save_dir_path + save_data_path + '.csv')
print("--save done--")
