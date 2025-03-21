from flask import Flask, request, jsonify
import torch
import numpy as np
from model import LSTMModel

app = Flask(__name__)

# Model params must match training
input_size = 10
hidden_size = 64
num_layers = 2
output_size = 1

# Initialize and load model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device("cpu")))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Input should be a list or nested list depending on timesteps
    input_sequence = np.array(data["sequence"], dtype=np.float32)  # e.g. shape (1, 20, 10)
    input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # shape becomes (1, seq_len, input_size)

    with torch.no_grad():
        prediction = model(input_tensor)
    
    return jsonify({"prediction": prediction.squeeze().tolist()})

if __name__ == "__main__":
    app.run(debug=True)
