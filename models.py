import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_Net(nn.Module):
    """ FC-type net, strongly inspired by https://www.kaggle.com/chriszou/titanic-with-pytorch-nn-solution"""
    def __init__(self, emb_dims, n_cont, lin_layer_sizes, output_size, dropout_rate=0.2):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        self.n_embs = sum([y for x, y in emb_dims])
        self.n_cont = n_cont
        self.output_size = output_size

        # Linear Layers
        first_lin_layer = nn.Linear(self.n_embs + self.n_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + 
                                        [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)])
    
        for lin_layer in self.lin_layers:
             nn.init.kaiming_normal_(lin_layer.weight.data)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)
    
    def forward(self, cont_data, cat_data):
        if self.n_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)

            if self.n_cont != 0:
                x = torch.cat([x, cont_data], 1) 
            
        else:
            x = cont_data

        for lin_layer in self.lin_layers:
            x = torch.relu(lin_layer(x))
            x = self.dropout(x)

        x = self.output_layer(x)
        
        if self.output_size == 1:
            x = torch.sigmoid(x)
        else:
            x = F.log_softmax(x, dim=1)
        return x