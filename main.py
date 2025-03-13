# creating main file for projectgit
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
#from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import re
from torch.nn.utils.rnn import pad_sequence #padding because of error that dataloader has different variable lengths


import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


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
#print(f'Initial unique vals in extensive col\n{df["extensive"].unique()}')

#clearing "genre" col of bad strings: translating into English 
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

#performed class distribution analysis for "genre" and dropped extra genre cols not benefitting model
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
        item_tensor = torch.tensor(letter, dtype=torch.long)
        
        genre = df.iloc[index,1:11]
        genre = torch.tensor(genre,dtype = torch.float32)
        #print(genre)

        character = nn.functional.one_hot(item_tensor, num_classes=59)
        #print(character)
    
        return character, genre
        
df.info()

def padding_batch(batch):
    return pad_sequence(batch, batch_first=True) #could padding see if it changes batches to be weirder
    
training_dataset = MyDataset(df[:11836]) #80 percent for training
training_dataloader = DataLoader(training_dataset,batch_size=1,shuffle=True) 

testing_dataset = MyDataset(df[11836:]) #20 percent for testing
testing_dataloader = DataLoader(testing_dataset,batch_size=1,shuffle=True)

#proof that the tensors in the dataloaders are all properly created.. be able to loop through both dataloaders

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

#using training_char data and graphing when each character occurs
plt.figure(figsize=(12, 6))
plt.hist(stored_training_char, bins=len(char_to_num), color='red', alpha=0.7, edgecolor='black') #histogram of training characters
plt.xlabel('Index of Character')
plt.ylabel('Occurrences of Characters')
plt.title('Character Occurrence in Training Dataset')
plt.show()

#using testing_char data and graphing when each character occurs
plt.figure(figsize=(14, 4))
plt.hist(stored_testing_char, bins=len(char_to_num), color='pink', alpha=0.7, edgecolor='black') #set histogram to testing characters
plt.xlabel('Index of Character')
plt.ylabel('Occurrences of Characters')
plt.title('Character Occurrence in Testing Dataset')
plt.show()


#class that inherits from Pytorch
class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(myRNN,self).__init__()
        self.hidden_size = 50
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        #(input,hidden,output)
        self.i2o = nn.Linear(59 + self.hidden_size,100) #can inc 60
        self.i2h = nn.Linear(59 + self.hidden_size,self.hidden_size)
        self.i2h2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.i2h3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.o2o = nn.Linear(100 + self.hidden_size,10) 
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        output = self.i2o(combined)
        hidden = self.i2h(combined)
        hidden = self.i2h2(hidden)
        hidden = self.i2h3(hidden)
        hidden = self.activation(hidden)
        #hidden = self.dropout(hidden)
        out_combined = torch.cat((output,hidden),dim=1)
        output = self.o2o(out_combined)     
        output = self.softmax(output)
        output = self.dropout(output)    
        return output,hidden   

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)
        

#in,out,hidden size
rnn = myRNN(59,10,9) 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr = 0.001) 
epochs = 2
#in,hidden_size, out
rnn = myRNN(59,50,10) 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr = 0.001)
epochs = 5

rnn.train(True)
#training_loss_lst = []

"""Training loop"""
for e in range(epochs): 
    for value, genre in training_dataloader:
        hidden = rnn.initHidden()
        for i in range(len(value[:,0])):
            pred, hidden = rnn(value[:,i],hidden)
            # print(pred.shape)
            # print(hidden.shape)
            # print(genre.shape)
        training_loss = loss_fn(pred,genre)
        print(f'Training loss: {training_loss.item()}') #print the sqrt of training loss to see accurate loss comparison
        training_loss.backward() #calculates slope to guide optimizer
        optimizer.step() #updating weights
        optimizer.zero_grad() #resets optimizer for epochs



#         """training code: tested when implementing LSTM will revist and refine moving forward"""
#         # batch_size = value.shape[0]
        """training code: tested when implementing LSTM will revist and refine moving forward"""
        # batch_size = value.shape[0]
        # h0,c0 = rnn.initHidden(batch_size)
        # pred,_ = rnn(value,(h0,c0))
        # training_loss = loss_fn(pred, genre)
        # print(f'Training loss: {training_loss.item()}')
    
#plt.plot(training_loss_lst)
#plt.show()

tested_values = 0
correct_pred = 0
genres = ["Action", "Adventure", "Casual", "Indie", "Multiplayer", "RPG", "Racing", "Simulation", "Sports", "Strategy"]
rnn.eval()

"""Testing loop"""
for value, genre in testing_dataloader:

    if tested_values == 200:
        break
    hidden = rnn.initHidden()
    for i in range(value.shape[1]):
        pred, hidden = rnn(value[:,i],hidden)
    print(pred)
    print(genre)

    max_index = torch.argmax(pred).item()
    true_index = torch.argmax(genre).item()

    print(max_index == true_index)

    if max_index == true_index:
        correct_pred += 1
    tested_values += 1

    print(f"Correct: {correct_pred} / Total: {tested_values}")
    print(f"Predicted genre: {genres[max_index]}, Correct genre: {genres[true_index]}")

    testing_loss = loss_fn(pred,genre)
    print(f'Testing loss: {testing_loss.item()}')


    """testing code: tested when implementing LSTM will revist and refine moving forward"""
    # batch_size = value.shape[0]
    # h0,c0 = rnn.initHidden(batch_size)
    # pred,_ = rnn(value,(h0,c0))
    # testing_loss = loss_fn(pred, genre)
    # print(f'Testing loss: {testing_loss.item()}')

    #sum value checking 0 and 1 for testing the genres

    #split it up by words (word to vec tools to make into numbers similaries between words as numbers) as stretch goal geeks 4 geeks word embedding
