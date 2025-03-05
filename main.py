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

df1 = pd.read_csv("descriptions.csv", on_bad_lines="skip") #skip glitched lines
df1.info()
print(df1["app_id"].unique()) #view the unique values for debugging the different languages
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
print(df2["app_id"].unique())
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

#Replace extensive special characters

df["extensive"] = df["extensive"].replace(r"[`(){}[\]|_\b\\]", "", regex = True) #fixing the characters


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

#one hot encode the letters as numbers loop through text

char_to_num = {}

extensive_set = set()

for string in df['extensive']: #for each string int he extensive column add it to the set
    string = string.lower()
    for char in string:
        extensive_set.add(char)
    

for i, char in enumerate(set(extensive_set)): #loops through index of each unique element in extensive text
    char_to_num[char] = i #sets dictionary value mapped to its index
print(char_to_num)

def every_letter(extensive_text):
    if isinstance(extensive_text, pd.Series):
        extensive_text = extensive_text.iloc[0]  # fix error with series from method not being interpreted
    extensive_text = extensive_text.lower() #remove error with duplicate letters from uppercase letters
    num_list = [] #new list to store numbers for each character
    for char in extensive_text:
        num_list.append(char_to_num[char])  #going thorugh each character from text to append value mapped in dictionary to list 
    return num_list

print(f'These are the duplicates:\n{df.loc[df.duplicated()]}') #empty no duplicates in df

df = df.drop(columns="app_id")

#df = df.drop(columns="extensive") #dropping extensive too to support tensor conversion

df.replace('\\N', np.nan, inplace=True) #replace null values

df = df.dropna(ignore_index = True)

df.to_csv("cleaned_dataset.csv")

df.info()

class MyDataset(Dataset): 
    def __init__(self,data):
        #initializing 
        self.length = len(data)
        self.data = data
    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        letter = every_letter(self.data.iloc[index, 0]) #taking string from index on extensive column and getting values
        item_tensor = torch.tensor(letter)
        return nn.functional.one_hot(item_tensor, num_classes=59)
        # return self.data[index]

df.info()
from torch.nn.utils.rnn import pad_sequence #padding because of error that dataloader has different variable lengths

def padding_batch(batch):
    return pad_sequence(batch, batch_first=True)
    
testing_dataset = MyDataset(df[11836:]) #20 percent for testing
testing_dataloader = DataLoader(testing_dataset,batch_size=500,shuffle=True, collate_fn=padding_batch)
trained_dataset = MyDataset(df[:11836]) #80 percent for training
trained_dataloader = DataLoader(trained_dataset,batch_size=500,shuffle=True, collate_fn=padding_batch) 

#proof that the tensors in the dataloaders are all properly created.. be able to loop through both dataloaders
for value in testing_dataloader:
    print(value)

for value in trained_dataloader:
    print(value)

"""Graph Visualization"""

#intialize  for graph creation
sample_size = 100 

stored_testing_char = []
stored_testing_count = []

stored_training_char = []
stored_training_count = []


for i in range(sample_size):
    stored_testing_char.extend(every_letter(df.iloc[11836 + i, 0])) #setting the list of each testing character for each amount in sample size
    stored_training_char.extend(every_letter(df.iloc[i, 0])) #list of training characters for each letter up to sample size amount

plt.figure(figsize=(14, 4))
plt.hist(stored_testing_char, bins=len(char_to_num), color='pink', alpha=0.7, edgecolor='black') #set histogram to testing characters
plt.xlabel('Index of Character')
plt.ylabel('Occurrences of Characters')
plt.title('Character Occurrence in Testing Dataset')
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(stored_training_char, bins=len(char_to_num), color='red', alpha=0.7, edgecolor='black') #histogram of training characters
plt.xlabel('Index of Character')
plt.ylabel('Occurrences of Characters')
plt.title('Character Occurrence in Training Dataset')
plt.show()

#NOTHING MOREFOR MILESTONE 2
quit()


#class that inherits from Pytorch
class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myRNN,self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,input):
        #goes thro layers 
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


#in,out,hidden size
model = myRNN(25,10,100) 
#pred = model()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1.0) 
epochs = 10000


training_loss_lst = []

"""Training loop"""
for batch in dataloader:
    for x, y in range(epochs): 
        hidden = model.initHidden()
        for i in range(len(x[:,0])):
            pred, hidden = model(x[:,i],hidden)
        training_loss = loss_fn(pred,y)
       
        print(f'Training loss: {math.sqrt(training_loss.item())}') #print the sqrt of training loss to see accurate loss comparison
        training_loss_lst.append(math.sqrt(training_loss.item()))

        training_loss.backward() #calculates slope to guide optimizer
        optimizer.step() #updating weights
        optimizer.zero_grad() #resets optimizer for epochs
    

plt.plot(training_loss_lst)
plt.show()


"""Testing"""
pred = model(testing_dataset)
testing_loss = loss_fn(pred,testing_outputs)
#print the sqrt of testing loss to see accurate loss comparison
print(f'Testing loss: {math.sqrt(testing_loss.item())}')





