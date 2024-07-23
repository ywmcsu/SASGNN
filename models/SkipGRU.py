import torch
import torch.nn as nn

class RecurrentSkipLayer(nn.Module):
    def __init__(self, input_size, hidden_size, skip_period):
        super(RecurrentSkipLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.skip_period = skip_period
        self.output_linear = nn.Linear(hidden_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.skip_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        output = []
        h_t = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        h_history = []

        for t in range(seq_len):
            x_t = x[:, t, :].unsqueeze(1)
            _, h_t = self.gru(x_t, h_t)

            if t >= self.skip_period:
                skip_hidden = h_history[-self.skip_period]
                h_t = h_t + self.skip_linear(skip_hidden).unsqueeze(0)

            output.append(h_t.squeeze(0))
            h_history.append(h_t.squeeze(0).detach())

        output = torch.stack(output, dim=1)
        output = self.output_linear(output).transpose(1, 2)
        return output