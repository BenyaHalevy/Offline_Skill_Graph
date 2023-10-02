import pandas as pd
from scripts.data_collection.State_class_rdc import*
from pathlib import Path

root_path = Path(__file__).absolute().parent

# goal_state = s2
old = False
if old:
    dir_path1 = f"{root_path}\datasets\goal\door_slide\\"
    dir_path2 = f"{root_path}\datasets\goal\S10-S11\\"
    dir_path3 = f"{root_path}\datasets\goal\S11-S12\\"
    dir_path4 = f"{root_path}\datasets\goal\S12-S13\\"
    dir_path5 = f"{root_path}\datasets\goal\grab_cube\\"
    dir_path6 = f"{root_path}\datasets\goal\pick_cube\\"
    dir_path7 = f"{root_path}\datasets\goal\S16-S17\\"
    dir_path8 = f"{root_path}\datasets\goal\place_cube\\"

    df1 = pd.read_pickle(dir_path1 + "door_slide.pkl")
    df2 = pd.read_pickle(dir_path2 + "S10 to S11.pkl")
    df3 = pd.read_pickle(dir_path3 + "S11 to S12.pkl")
    df4 = pd.read_pickle(dir_path4 + "S12 to S13.pkl")
    df5 = pd.read_pickle(dir_path5 + "grab_cube.pkl")
    df6 = pd.read_pickle(dir_path6 + "pick_cube.pkl")
    df7 = pd.read_pickle(dir_path7 + "S16-S17.pkl")
    df8 = pd.read_pickle(dir_path8 + "place_cube.pkl")

    print("--load done--")
    print('df1 length:', df1.shape[0])
    print('df2 length:', df2.shape[0])
    print('df3 length:', df3.shape[0])
    print('df4 length:', df4.shape[0])
    print('df5 length:', df5.shape[0])
    print('df6 length:', df6.shape[0])
    print('df7 length:', df7.shape[0])
    print('df8 length:', df8.shape[0])

    df_list = [df1, df2, df3, df4, df5, df6, df7, df8]
else:
    path1 = f"{root_path}\datasets\\no_goal\S14-S16\\S14-S16_1\S14 to S16.pkl"
    path2 = f"{root_path}\datasets\\no_goal\S14-S16\\S14-S16_2\S14 to S16.pkl"

    df1 = pd.read_pickle(path1)
    df2 = pd.read_pickle(path2)
    print("--load done--")
    print('df1 length:', df1.shape[0])
    print('df2 length:', df2.shape[0])
    df_list = [df1, df2]

print("--start merge--")
merged = pd.concat(df_list)
merged.reset_index(inplace=True)
print("--merge done--")
print('merged length:', merged.shape[0])

print("--save--")
merged.to_pickle('datasets\\no_goal\S14-S16\S14 to S16.pkl')
merged.to_csv('datasets\\no_goal\S14-S16\\S14 to S16.csv')
print("--save done--")
