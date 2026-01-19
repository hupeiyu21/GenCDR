from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, dropout=0.0):
        super(MLP, self).__init__()
        self.mlp_blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.in_dropout = nn.Dropout(p=dropout)
        self.out_projection = nn.Linear(hidden_sizes[-1], latent_size)
        hidden_sizes = [input_size] + hidden_sizes
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.LayerNorm(output_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
            # add residual connections
            self.residuals.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=output_size,
                    kernel_size=input_size,
                    bias=False,
                    stride=input_size,
                )
            )

    def forward(self, x):
        x = self.in_dropout(x)
        for i in range(len(self.mlp_blocks)):
            res = self.residuals[i](x.unsqueeze(1)).squeeze()
            x = self.mlp_blocks[i](x)
            x = x + res
        return self.out_projection(x)