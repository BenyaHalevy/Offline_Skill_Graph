import pandas as pd
from pathlib import Path

root_path = Path(__file__).absolute().parent
main_path = f"{root_path}\datasets\\classifier"
sub1 = "\S11-S12\S11 to S12.pkl"
sub2 = "\S12-S13\S12 to S13.pkl"
sub3 = "\S13-S11\S13 to S11.pkl"
sub4 = "\S10-S11\S10 to S11.pkl"
sub5 = "\S6-S11\S6 to S11.pkl"
sub6 = "\S12-S6\S12 to S6.pkl"

df1 = pd.read_pickle(main_path + sub1)
df2 = pd.read_pickle(main_path + sub2)
df3 = pd.read_pickle(main_path + sub3)
df4 = pd.read_pickle(main_path + sub4)
df5 = pd.read_pickle(main_path + sub5)
df6 = pd.read_pickle(main_path + sub6)


print("--load done--")
print('df1 length:', df1.shape[0])
print('df2 length:', df2.shape[0])
print('df3 length:', df3.shape[0])
print('df4 length:', df4.shape[0])
print('df5 length:', df5.shape[0])
print('df6 length:', df6.shape[0])
df_list = [df1, df2, df3, df4, df5, df6]

print("--start merge--")
merged = pd.concat(df_list)
merged.reset_index(inplace=True)
merged.drop(['index'], axis=1, inplace=True)
print("--merge done--")
print('merged length:', merged.shape[0])

print("--save--")
merged.to_pickle('datasets\\classifier\\merged\\merged.pkl')
merged.to_csv('datasets\\classifier\\merged\\merged.csv', index=False)
print("--save done--")
