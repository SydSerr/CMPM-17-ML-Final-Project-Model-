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
        
        row = df.iloc[index]
        genre_cols = df.columns[1:11]

        genre = [genre for genre in genre_cols if row[genre]==1]

        character = nn.functional.one_hot(item_tensor, num_classes=59)
        #genre = torch.tensor(self.data.iloc[index, 1:].values.astype(np.float32))
        return character, genre
        

df.info()

def padding_batch(batch):
    return pad_sequence(batch, batch_first=True)
    
training_dataset = MyDataset(df[:11836]) #80 percent for training
training_dataloader = DataLoader(training_dataset,batch_size=500,shuffle=True, collate_fn=padding_batch) 

testing_dataset = MyDataset(df[11836:]) #20 percent for testing
testing_dataloader = DataLoader(testing_dataset,batch_size=500,shuffle=True, collate_fn=padding_batch)

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
        self.hidden_size = 10
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        #input,hidden,output
        self.i2o = nn.Linear(59 + self.hidden_size,9) 
        self.i2h = nn.Linear(59+ self.hidden_size,self.hidden_size)  
        self.o2o = nn.Linear(9 + self.hidden_size,9) 
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.Tanh()
        #self.lstm = nn.LSTM(59,self.hidden_size,3,batch_first=True)

    def forward(self,input,hidden):

        combined = torch.cat((input,hidden),1)
        output = self.i2o(combined)
        hidden = self.i2h(combined)
        hidden = self.activation(hidden)
        out_combined = torch.cat((output,hidden),dim=1)
        output = self.o2o(out_combined)
        output = self.softmax(output)        
        return output,hidden
    
        #goes thro layers 
        #lstm_out, hidden = self.lstm(input,hidden)
        #lstm_out_last = lstm_out[:,-1,:]
        #"""Check"""
        # combined = torch.cat((lstm_out_last,input[:,-1,:]),dim=1)
        # output = self.i2o(combined)
        # output = self.activation(output)
        # hidden = self.i2h(combined)
        # hidden = self.activation(hidden)
        # """check"""
        # out_combined = torch.cat((output,lstm_out_last),dim=1)
        # output = self.o2o(out_combined)
        # output = self.activation(output)
        # output = self.softmax(output)
        # # combined = torch.cat((input,hidden),1)
        # # output = self.i2o(combined)""""

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)
        ## need clarification with lstm
        # h0 = torch.zeros(1, batch_size, self.hidden_size)
        # c0 = torch.zeros(1, batch_size, self.hidden_size)
        # return h0,c0


#in,out,hidden size
rnn = myRNN(59,10,9) 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr = 0.01) 
epochs = 10000

#training_loss_lst = []

"""Training loop: get checked"""

for e in range(epochs): 
    for value, genre in training_dataloader:
        for i in range(len(value[:,0])):
            pred, hidden = rnn(value[:,i],hidden)
        training_loss = loss_fn(pred,genre)
       
        print(f'Training loss: {training_loss.item()})') #print the sqrt of training loss to see accurate loss comparison

        training_loss.backward() #calculates slope to guide optimizer
        optimizer.step() #updating weights
        optimizer.zero_grad() #resets optimizer for epochs

        # batch_size = value.shape[0]
        # h0,c0 = rnn.initHidden(batch_size)
        # pred,_ = rnn(value,(h0,c0))
        # training_loss = loss_fn(pred, genre)
        # print(f'Training loss: {training_loss.item()}')
    
#plt.plot(training_loss_lst)
#plt.show()

"""Testing loop"""
for value, genre in testing_dataloader:
    hidden = rnn.initHidden()
    for i in range(value.shape[1]):
        pred, hidden = rnn(value[:,i],hidden)
    testing_loss = loss_fn(pred,genre)
    print(f'Testing loss: {testing_loss.item()}')

    # batch_size = value.shape[0]
    # h0,c0 = rnn.initHidden(batch_size)
    # pred,_ = rnn(value,(h0,c0))
    # testing_loss = loss_fn(pred, genre)

    # print(f'Testing loss: {testing_loss.item()}')

