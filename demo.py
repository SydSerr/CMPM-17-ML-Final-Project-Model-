from myRNN import myRNN
import torch

model = myRNN(59,150,10)
model.load_state_dict(torch.load("demo.pth", weights_only = True))

#user_input = input("Enter game description:")