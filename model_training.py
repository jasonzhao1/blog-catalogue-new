from logging import Logger

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from evaluator import Evaluator
from model import MLP, train, test

RUNS = 5
LEARNING_RATE = 0.01
EPOCHS = 1000


# function to train the model and save it to the output
def get_model(embeddings, labels):
    # Assuming you have the following data:
    # embeddings: The spectral embeddings of your graph nodes.
    # labels: The one-hot encoded labels of your nodes.

    # Convert embeddings to PyTorch tensors
    X = torch.tensor(embeddings, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32)  # Keep labels as one-hot encoded

    # Split data into train, validation, and test sets
    train_idx, temp_idx = train_test_split(range(len(X)), test_size=0.4, random_state=42)
    valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    # Initialize the model and evaluator
    model = MLP(in_channels=X.shape[1], hidden_channels=64, out_channels=39, num_layers=3, dropout=0.5)
    evaluator = Evaluator()

    # Training loop with multiple runs
    num_runs = RUNS
    for run in range(num_runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best_valid_acc = 0
        best_out = None

        for epoch in range(EPOCHS):
            loss = train(model, X, Y, train_idx, optimizer)
            (train_acc, valid_acc, test_acc), out = test(model, X, Y, split_idx, evaluator)

            # # Optionally: Save the model or its outputs with the best validation accuracy
            # if valid_acc > best_valid_acc:
            #     best_valid_acc = valid_acc
            #     best_out = out.cpu().exp()

            print(f'epoch{epoch}', train_acc)
            # print(
            #     f"Run: {run + 1}/{num_runs}, Epoch: {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}")

        # Save the best output from this run
        # torch.save(best_out, f'best_out_run_{run}.pt')

    # After all runs, you can evaluate the model's performance on the test set or analyze the saved outputs.
