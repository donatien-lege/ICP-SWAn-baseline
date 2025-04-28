import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(
        self,
        dim_in: int = 180,
        length: int = 40,
        dim_out: int = 3,
        cells: int = 3,
        dropout: float = 0.1,
    ):
        super(GRUNet, self).__init__()

        self.preprocess = nn.Linear(dim_in, dim_in)
        self.hidden_layer_size = length
        self.lstm = nn.GRU(
            dim_in,
            hidden_size=length,
            num_layers=cells,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.hidden = nn.Linear(2 * length, 2 * length)
        self.linear = nn.Linear(2 * length, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.relu(x)
        lstm_out, self.hidden_cell = self.lstm(x)
        x = self.hidden(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.linear(lstm_out)
        return y
    
    
class GRUQuantile(nn.Module):
    def __init__(self, dim_in=180, hls=40, dim_out=4, numl=2, drop=0.05):
        super(GRUQuantile, self).__init__()

        self.preprocess = nn.Linear(dim_in, dim_in)
        self.gru = nn.GRU(
            dim_in,
            hidden_size=hls,
            num_layers=numl,
            batch_first=True,
            bidirectional=True,
        )
        self.hidden = nn.Linear(2 * hls, 2 * hls)
        self.dropout = nn.Dropout(drop)
        self.linear = nn.Linear(2 * hls, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.relu(x)
        gru_out, self.hidden_cell = self.gru(x)
        gru_out = self.hidden(gru_out)
        gru_out = self.relu(gru_out)
        gru_out = self.dropout(gru_out)
        y = self.linear(gru_out)
        return y
    

