import torch.nn as nn
import torch.optim as optim
import torch
from data_processing import training_dataloader,testing_dataloader
from myRNN import myRNN

rnn = myRNN(59,150,10) #in,hidden_size, out
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr = 0.001)
epochs = 3

rnn.train(True)

"""Training loop"""
if __name__ == '__main__':
    for e in range(epochs): 
        for value, genre in training_dataloader:
            hidden = rnn.initHidden()
            for i in range(len(value[:,0])):
                pred, hidden = rnn(value[:,i],hidden)
            training_loss = loss_fn(pred,genre)
            print(f'Training loss: {training_loss.item()}') 
            training_loss.backward() #calculates slope to guide optimizer
            optimizer.step() #updating weights
            optimizer.zero_grad() #resets optimizer for epochs


    rnn.eval()

    tested_values = 0
    correct_pred = 0
    genres = ["Action", "Adventure", "Casual", "Indie", "Multiplayer", "RPG", "Racing", "Simulation", "Sports", "Strategy"]

    """Testing loop"""
    for value, genre in testing_dataloader:

        if tested_values == 100:
            break
        hidden = rnn.initHidden()
        for i in range(value.shape[1]):
            pred, hidden = rnn(value[:,i],hidden)

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

#save model into new file
torch.save(rnn.state_dict(),"demo.pth")
