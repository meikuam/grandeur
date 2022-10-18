import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        self.clf = nn.Linear(hidden_size, 10)
    
    def forward(self, input, length=None):
        # input: (batch_size, hidden_size, seq_len)
        output, _ = self.rnn(input.transpose(-1, -2))
        # output: (batch_size, seq_len, hidden_size)
        
        # Now we want to take the last hidden state of each instance in batch
        # BUT we don't want to take `padding` hidden state
        # We will use `torch.gather` and `length` to dio that
        
        # learn more about gather
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        
        last_hidden = torch.gather(
            output,
            dim=1,
            index=length.sub(1).view(-1,1,1).expand(-1,-1,self.hidden_size)
            # index=length.sub(1).unsqueeze(-1).repeat(1,1,self.hidden_size)
        )
        # (batch, 1, hidden_size)
        
        logits = self.clf(last_hidden.squeeze(dim=1))
        
        return logits