from myRNN import myRNN
import torch
import torch.nn.functional 
from data_processing import every_letter
import re

# Load model
model = myRNN(59, 150, 10)
model.load_state_dict(torch.load("demo.pth"))  
model.eval()

genres = ["Action", "Adventure", "Casual", "Indie", "Multiplayer", "RPG", "Racing", "Simulation", "Sports", "Strategy"]

def pred_genre(user_description):
    input_processed = every_letter(user_description)
    # one-hot encode character indices 
    input_tensor = torch.tensor(input_processed, dtype=torch.long)
    input_tensor = torch.nn.functional.one_hot(input_tensor, num_classes=59).float().unsqueeze(0) 

    # Initialize hidden state
    hidden = model.initHidden()

    # Pass input sequentially
    for i in range(input_tensor.shape[1]):  # Iterates through each char
        output, hidden = model(input_tensor[:, i], hidden)
    
    # Gets predicted index
    pred_index = torch.argmax(output, dim=1).item()  

    return genres[pred_index] 

#user prompted to enter description and outputs the model prediction
if __name__ == "__main__":
    user_input = input("Enter game description!: ")
    user_input = re.sub(r"[`(){}[\]|_\b\\]", "", user_input)
    predicted_genre = pred_genre(user_input)  
    print(f"Predicted Genre: {predicted_genre}")