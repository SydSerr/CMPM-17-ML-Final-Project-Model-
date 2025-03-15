from myRNN import myRNN
import torch
from main import every_letter
import numpy as np
import re

model = myRNN(59,150,10)
model.load_state_dict(torch.load("demo.pth", weights_only = True))

model.eval()

genres = ["Action", "Adventure", "Casual", "Indie", "Multiplayer", "RPG", "Racing", "Simulation", "Sports", "Strategy"]

def pred_genre(user_description):
    input_processed = every_letter(user_description)
    
    # Ensure input shape is (1, seq_len, input_size)
    input_tensor = torch.tensor(input_processed, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_len, input_size)
    
    print(f"Input tensor shape: {input_tensor.shape}")  # Debugging step to check shape

    hidden = model.initHidden(batch_size=1)  # Shape: (1, 1, hidden_size)
    print(f"Hidden state shape: {hidden.shape}")  # Debugging step to check shape

    output, hidden = model(input_tensor, hidden)  # Forward pass
    print(f"Output shape: {output.shape}")  # Debugging step to check shape

    pred_index = torch.argmax(output, dim=1).item()
    print(f"Predicted index: {pred_index}")
    print(f"Output shape after argmax: {output.shape}")
    print(f"Hidden shape after forward pass: {hidden.shape}")

    return genres[pred_index]

if __name__ == "__main__":
    user_input = str(input("Enter game description!: "))
    user_input = re.sub(r"[`(){}[\]|_\b\\‘’“”]", "", user_input)
    print(f"Predicted Genre: {pred_genre(user_input)}")

