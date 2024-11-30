import torch
from flask import Flask, jsonify
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
from datetime import datetime

app = Flask(__name__)

# Load the trained model
class BitcoinLSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(BitcoinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = BitcoinLSTM()
model.load_state_dict(torch.load('bitcoin_lstm_model.pth'))
model.eval()

scaler = MinMaxScaler()

end_date = datetime.today().strftime('%Y-%m-%d')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        btc_data = yf.download('BTC-USD', start='2024-01-01', end=end_date)
        btc_close = btc_data[['Close']].values

        scaled_data = scaler.fit_transform(btc_close)

        last_30_days = scaled_data[-30:]
        last_30_days = torch.tensor(last_30_days, dtype=torch.float32).reshape(1, -1, 1)

        with torch.no_grad():
            prediction = model(last_30_days)
            prediction = scaler.inverse_transform(prediction.numpy())

        return jsonify({
            'prediction': float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
