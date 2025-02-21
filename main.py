# creating main file for projectgit
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv("descriptions.csv", on_bad_lines="skip")
df1.info()
print(df1["app_id"].unique())
df1 = df1.dropna(ignore_index=True)
df1 = df1.drop_duplicates (ignore_index=True)
df1.info()

df2 = pd.read_csv("genres.csv")
print(df2["app_id"].unique())
df2.info()
df2 = df2.dropna(ignore_index=True)
df2 = df2.drop_duplicates (ignore_index=True)
df2.info()
#df1["app_id"]=df1["app_id"].astype(int)

#df = df1.join(df2,on="app_id")
df = pd.concat([df1,df2],axis=1)
print(df)
df.info()


df.to_csv("cleaned_dataset.csv")
print(df)

