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

print(df)

print(f'What null vals are there?\n{df.loc[df.isna().any(axis=1)]}')
df = df.dropna(ignore_index=True) 
print(f'What null vals are there?\n{df.loc[df.isna().any(axis=1)]}')


print(f'Unique vals in genre col\n{df["genre"].unique()}')
print(f'Unique vals in extensive col\n{df["extensive"].unique()}')

df["genre"] = df["genre"].replace("Экшены","Action")
df["genre"] = df["genre"].replace("Бесплатные","Free to Play")
df["genre"] = df["genre"].replace("Стратегии","Strategy")
df["genre"] = df["genre"].replace("Ação","Action")
df["genre"] = df["genre"].replace("Приключенческие игры","Adventure")
df["genre"] = df["genre"].replace("Ролевые игры","Roleplay")
df["genre"] = df["genre"].replace("Rol","Roleplay")
df["genre"] = df["genre"].replace("Akční","Action")
df["genre"] = df["genre"].replace("Dobrodružné","Adventure")
df["genre"] = df["genre"].replace("动作","Action")
df["genre"] = df["genre"].replace("策略","Strategy")
df["genre"] = df["genre"].replace("角色扮演","Roleplay")
df["genre"] = df["genre"].replace("Acción","Action")
df["genre"] = df["genre"].replace("Aventure","Adventure")
df["genre"] = df["genre"].replace("Симуляторы","Simulation")
df["genre"] = df["genre"].replace("Гонки","Racing")
df["genre"] = df["genre"].replace("Спортивные игры","Sports")
df["genre"] = df["genre"].replace("Aventura","Adventure")
df["genre"] = df["genre"].replace("Многопользовательские игры","Multiplayer")
df["genre"] = df["genre"].replace("Stratégie","Strategy")
df["genre"] = df["genre"].replace("Carreras","Racing")
df["genre"] = df["genre"].replace("Deportes","Sports")
df["genre"] = df["genre"].replace("Niezależne","Indie")
df["genre"] = df["genre"].replace("Strategie","Strategy")
df["genre"] = df["genre"].replace("模拟","Simulation")
df["genre"] = df["genre"].replace("アクション","Action")
df["genre"] = df["genre"].replace("アドベンチャー","Adventure")
df["genre"] = df["genre"].replace("インディー","Indie")
df["genre"] = df["genre"].replace("Simulationen","Simulation")
df["genre"] = df["genre"].replace("Rollenspiel","Roleplay")
df["genre"] = df["genre"].replace("冒险","Adventure")
df["genre"] = df["genre"].replace("Eventyr","Adventure")
df["genre"] = df["genre"].replace("Strategi","Strategy")
df["genre"] = df["genre"].replace("Казуальные игры","Casual")
df["genre"] = df["genre"].replace("Avventura","Adventure")
df["genre"] = df["genre"].replace("Azione","Action")
df["genre"] = df["genre"].replace("Actie","Action")
df["genre"] = df["genre"].replace("Пригоди","Adventure")
df["genre"] = df["genre"].replace("Estrategia","Strategy")
df["genre"] = df["genre"].replace("Roolipelit","Roleplay")
df["genre"] = df["genre"].replace("Seikkailu","Adventure")
df["genre"] = df["genre"].replace("Strategia","Strategy")
df["genre"] = df["genre"].replace("Ранний доступ","Early Access")
df["genre"] = df["genre"].replace("Akcja","Action")
df["genre"] = df["genre"].replace("Инди","Indie")
df["genre"] = df["genre"].replace("独立","Indie")
df["genre"] = df["genre"].replace("Massively Multiplayer","Multiplayer")
df["genre"] = df["genre"].replace("Free To Play","Free to Play")
df["genre"] = df["genre"].replace("Abenteuer","Adventure")

print(f'Unique vals in genre col\n{df["genre"].unique()}')

df.to_csv("cleaned_dataset.csv")

