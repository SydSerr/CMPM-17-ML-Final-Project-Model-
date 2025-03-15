from myRNN import myRNN
import torch
from main import every_letter
import numpy as np

model = myRNN(59,150,10)
model.load_state_dict(torch.load("demo.pth", weights_only = True))

model.eval()

genres = ["Action", "Adventure", "Casual", "Indie", "Multiplayer", "RPG", "Racing", "Simulation", "Sports", "Strategy"]

def pred_genre(user_description):
    input_processed = every_letter(user_description)
    input_tensor = torch.tensor(input_processed, dtype = torch.long).unsqueeze(0)
    print(input_tensor.shape)
    hidden = model.initHidden()
 
    output, _ = model(input_tensor, hidden)
    pred_index = torch.argmax(output, dim=1).item()

    return genres[pred_index]

if __name__ == "__main__":
    user_input = str(input("Enter game description!: "))
    pred_genre = pred_genre(user_input)
    print(f"Predicted Genre: {pred_genre}")

