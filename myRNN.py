import torch.nn as nn
import torch


#class that inherits from Pytorch
class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size):
        super(myRNN,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        #initializing layers of model
        self.i2o = nn.Linear(59 + self.hidden_size,90) 
        self.i2h = nn.Linear(59 + self.hidden_size,self.hidden_size)
        self.i2h2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.i2h3 = nn.Linear(self.hidden_size,self.hidden_size)
        self.o2o = nn.Linear(90 + self.hidden_size,10) 
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    #function to pass input and hidden through each layer of rnn to return another ouput and hidden
    def forward(self,input,hidden): 
        combined = torch.cat((input,hidden),1)
        output = self.i2o(combined)
        hidden = self.i2h(combined)
        hidden = self.i2h2(hidden)
        hidden = self.i2h3(hidden)
        hidden = self.activation(hidden)
        out_combined = torch.cat((output,hidden),dim=1)
        output = self.o2o(out_combined)
        output = self.softmax(output)
        output = self.dropout(output)    
        return output,hidden   

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)
        