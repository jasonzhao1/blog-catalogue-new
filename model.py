import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, relu_first = True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x  # Return raw scores (logits)


def train(model, x, y_true, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x[train_idx])
    loss = F.binary_cross_entropy_with_logits(out, y_true[train_idx])  # Use BCEWithLogits for multi-label
    loss.backward()
    optimizer.step()
    return loss.item()



@torch.no_grad()
def test(model, x, y, split_idx, evaluator):
    model.eval()
    out = model(x)
    y_pred = (torch.sigmoid(out) > 0.5).long()  # Apply sigmoid and threshold

    # Adjust the evaluator to handle multi-label predictions
    train_results = evaluator.eval(y_true=y[split_idx['train']], y_pred=y_pred[split_idx['train']])
    train_acc = train_results['f1_score_macro']

    valid_results = evaluator.eval(y_true=y[split_idx['valid']], y_pred=y_pred[split_idx['valid']])
    valid_acc = valid_results['f1_score_macro']

    test_results = evaluator.eval(y_true=y[split_idx['test']], y_pred=y_pred[split_idx['test']])
    test_acc = test_results['f1_score_macro']

    return (train_acc, valid_acc, test_acc), out

