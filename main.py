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

import csv
import re

df1 = pd.read_csv("descriptions.csv", on_bad_lines="skip") #skip glitched lines2
df1.info()
#print(df1["app_id"].unique()) #view the unique values for debugging the different languages
df1 = df1.dropna(ignore_index=True)
df1 = df1.drop_duplicates (ignore_index=True)
df1 = df1.drop(columns= "summary")
df1 = df1.drop(columns= "about")
df1["extensive"] = df1["extensive"].str.replace('<br />', ' ', regex=True)
df1["extensive"] = df1["extensive"].str.replace(r'<p \>', ' ', regex=True)
df1["extensive"] = df1["extensive"].str.replace(r'\\\\', ' ', regex=True)
#CLEANING out extra line breaks so that columns are properly organized and not improperly split because of line breaks
df1["extensive"] = df1["extensive"].str.replace(r'\s+', ' ', regex=True).str.strip() #replace empty spaces with 1 space to solve further formatting issues

df1['app_id'] = pd.to_numeric(df1['app_id'], errors='coerce')
# drop rows where app id is number and replace with null value
df1 = df1.dropna(subset=['app_id']) 

df1 = df1[df1['extensive'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii') == x)] #drop non ascii values

df1.info()

df2 = pd.read_csv("genres.csv")
#print(df2["app_id"].unique())
df2.info()
df2 = df2.dropna(ignore_index=True)
df2 = df2.drop_duplicates (ignore_index=True)
df2 = df2.drop(columns= "app_id")
df2.info()

df = pd.concat([df1,df2],axis=1)
print(df)

df.info()

print(f'What null vals are there?\n{df.loc[df.isna().any(axis=1)]}')
df = df.dropna(ignore_index=True) 
print(f'What null vals are there?\n{df.loc[df.isna().any(axis=1)]}')

print(f'Initial unique vals in genre col\n{df["genre"].unique()}')
print(f'Initial unique vals in extensive col\n{df["extensive"].unique()}')

df["genre"] = df["genre"].replace("Экшены","Action")
df["genre"] = df["genre"].replace("Бесплатные","Free to Play")
df["genre"] = df["genre"].replace("Стратегии","Strategy")
df["genre"] = df["genre"].replace("Ação","Action")
df["genre"] = df["genre"].replace("Приключенческие игры","Adventure")
df["genre"] = df["genre"].replace("Ролевые игры","RPG")
df["genre"] = df["genre"].replace("Rol","RPG")
df["genre"] = df["genre"].replace("Akční","Action")
df["genre"] = df["genre"].replace("Dobrodružné","Adventure")
df["genre"] = df["genre"].replace("动作","Action")
df["genre"] = df["genre"].replace("策略","Strategy")
df["genre"] = df["genre"].replace("角色扮演","RPG")
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
df["genre"] = df["genre"].replace("Rollenspiel","RPG")
df["genre"] = df["genre"].replace("冒险","Adventure")
df["genre"] = df["genre"].replace("Eventyr","Adventure")
df["genre"] = df["genre"].replace("Strategi","Strategy")
df["genre"] = df["genre"].replace("Казуальные игры","Casual")
df["genre"] = df["genre"].replace("Avventura","Adventure")
df["genre"] = df["genre"].replace("Azione","Action")
df["genre"] = df["genre"].replace("Actie","Action")
df["genre"] = df["genre"].replace("Пригоди","Adventure")
df["genre"] = df["genre"].replace("Estrategia","Strategy")
df["genre"] = df["genre"].replace("Roolipelit","RPG")
df["genre"] = df["genre"].replace("Seikkailu","Adventure")
df["genre"] = df["genre"].replace("Strategia","Strategy")
df["genre"] = df["genre"].replace("Ранний доступ","Early Access")
df["genre"] = df["genre"].replace("Akcja","Action")
df["genre"] = df["genre"].replace("Инди","Indie")
df["genre"] = df["genre"].replace("独立","Indie")
df["genre"] = df["genre"].replace("Massively Multiplayer","Multiplayer")
df["genre"] = df["genre"].replace("Free To Play","Free to Play")
df["genre"] = df["genre"].replace("Abenteuer","Adventure")
df["genre"] = df["genre"].replace("Indépendant","Indie")


print(f'Final unique vals in genre col\n{df["genre"].unique()}')

#one hot encode the genre column
df = pd.get_dummies(df,columns=["genre"])

df = df.drop(columns="genre_Accounting")
df = df.drop(columns="genre_Animation & Modeling")
df = df.drop(columns="genre_Audio Production")
df = df.drop(columns="genre_Design & Illustration")
df = df.drop(columns="genre_Early Access")
df = df.drop(columns="genre_Free to Play")
df = df.drop(columns="genre_Game Development")
df = df.drop(columns="genre_Gore")
df = df.drop(columns="genre_Movie")
df = df.drop(columns="genre_Nudity")
df = df.drop(columns="genre_Photo Editing")
df = df.drop(columns="genre_Sexual Content")
df = df.drop(columns="genre_Software Training")
df = df.drop(columns="genre_Utilities")
df = df.drop(columns="genre_Video Production")
df = df.drop(columns="genre_Violent")
df = df.drop(columns="genre_Web Publishing")
df = df.drop(columns="genre_Education")







#one hot encode the letters in description

def every_letter(extensive_text):
    extensive_text = extensive_text.lower() #remove error with duplicate letters from uppercase letters
    return {char: 1 for char in set(extensive_text) if char.isalpha()} #create dictionary with each character in the set of the text within extensive if all letters are in alphabet

# Apply the function and expand the result into separate columns
char_df = df['extensive'].apply(every_letter).apply(pd.Series).fillna(0).astype(int) #fill null values with 0 as type integer for every letter in extensive once applied the the check for each alphabetical letter

df = pd.concat([df, char_df], axis=1) #combining character encoding with dataframe

print(f'These are the duplicates:\n{df.loc[df.duplicated()]}') #empty no duplicates in df

df = df.drop(columns="app_id")

df.info()

df.replace('\\N', np.nan, inplace=True) #replace null values

df = df.dropna(ignore_index = True)

df = df.drop(columns="extensive") #drop extensive, so we can convert to tensors with float values

df.to_csv("cleaned_dataset.csv")

df.info()

#converting data to torch tensors
data = torch.tensor(df.values.astype("float"),dtype=torch.float)

training_inputs = []
training_outputs = []

testing_inputs = []
testing_outputs = []

class MyDataset(Dataset): 
    def __init__(self,data):
        #initializing 
        self.length = len(data)
        self.data = data
    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        return self.data[index]
    
my_dataset = MyDataset(df)
dataloader = DataLoader(my_dataset,batch_size=500,shuffle=True) 

#class that inherits from Pytorch
class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myRNN,self).__init__()
        self.hidden_size = hidden_size
        

    def forward(self,input):
        #goes thro layers however many want

model = myRNN()
#pred = model()
loss_Fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1.0) 
epochs = 10000

for e in range(epochs):
    for x, y in dataloader:
        hidden = model.initHidden()
        for i in range(len(x[:,0])):
            pred, hidden = model(x[:,i],hidden)
        loss = lossFn(pred,y)
        back, step, zerograd