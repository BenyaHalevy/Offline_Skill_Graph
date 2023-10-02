import pandas as pd
from scripts.data_collection.State_class_rdc import*
from pathlib import Path

root_path = Path(__file__).absolute().parent

# goal_state = s2

# dir_path1 = f"{root_path}\datasets\goal\door_slide\\"
# dir_path2 = f"{root_path}\datasets\goal\S10-S11\\"
# dir_path3 = f"{root_path}\datasets\goal\S11-S12\\"
dir_path1 = f"{root_path}\datasets\goal\merged\\1st_sequence\\1st_step\\"
dir_path2 = f"{root_path}\datasets\goal\S12-S13\\"
dir_path3 = f"{root_path}\datasets\goal\grab_cube\\"

df1 = pd.read_pickle(dir_path1 + "1st_sequence.pkl")
df2 = pd.read_pickle(dir_path2 + "S12 to S13.pkl")
df3 = pd.read_pickle(dir_path3 + "grab_cube.pkl")

print("--load done--")
print('df1 length:', df1.shape[0])
print('df2 length:', df2.shape[0])
print('df3 length:', df3.shape[0])

# ''' stitch trajectories '''
# obs3 = df1['observations']
# rewards3 = df1['rewards']
# for i in range(len(obs3)):
#     obs3[i][7:] = np.array(list(goal_state.pos) + list(goal_state.orn))
#     rewards3[i] = 0.0
# df3 = df1.copy()
# df3['observations'] = obs3
# df3['rewards'] = rewards3
# print('--df3 created--', df3.shape[0])


print('--1st merge start--', df1.shape[0])

for i in range(len(df2)):
    df1.loc[df1.shape[0]] = df2.loc[i]
    if i % 1000 == 0:
        print(i)

print('--1st merge done--', df1.shape[0])
print('--2nd merge start--')

for i in range(len(df3)):
    df1.loc[df1.shape[0]] = df3.loc[i]
    if i % 1000 == 0:
        print(i)

print('--2nd merge done--', df1.shape[0])

df1.to_pickle('datasets\goal\merged\\1st_sequence\\1st_sequence.pkl')
df1.to_csv('datasets\goal\merged\\1st_sequence\\1st_sequence.csv')

