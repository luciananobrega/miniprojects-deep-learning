import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data, datasets
from torchtext.vocab import GloVe

torch.manual_seed(42)

class RNNModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, len_x):
        embedded = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, len_x, enforce_sorted=False)
        packed_output, (output, cell) = self.rnn(packed)
        return self.fc(output.squeeze(0)).view(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, len_x):
        embedded = self.embedding(x)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, len_x, enforce_sorted=False)
        packed_output, (output, cell) = self.lstm(packed)

        return self.fc(output.squeeze(0)).view(-1)


def compute_binary_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            text, text_lengths = batch_data.text
            labels = batch_data.label - 1
            logits = model(text, text_lengths)
            predicted_labels = (torch.sigmoid(logits) > 0.5).long()  # 0 and 1
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels.long()).sum()
        return correct_pred.float()/num_examples * 100


def train_model(model, optimizer):
    for b_id, batch in enumerate(train_iter):
        text, text_lengths = batch.text
        labels = batch.label - 1
        predictions = model(text, text_lengths)
        loss = criterion(predictions, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'    Train: {b_id + 1:03d}/{len(train_iter):03d} | '
              f'Loss: {loss:.4f}')


if __name__ == '__main__':
    state_dimensions = [20, 50, 100, 200, 500]
    batch_size = 64
    embedding_dim = 300  # GloVe embedding dimension
    output_dim = 1  # Binary classification

    # set up fields
    TEXT = data.Field(lower=True, include_lengths=True)
    LABEL = data.Field(sequential=False)
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    train, _ = train.split(split_ratio=0.01)  # Adjust the split ratio as needed
    test, _ = test.split(split_ratio=0.01)  # Adjust the split ratio as needed

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=embedding_dim))
    LABEL.build_vocab(train)

    # create iterators for batching
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=batch_size)

    # Define model, loss, and optimizer
    input_dim = len(TEXT.vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for hidden_dim in state_dimensions:
        print('== State dimensions: {} == '.format(hidden_dim))
        rnn_model = RNNModel(input_dim, embedding_dim, hidden_dim, output_dim)
        rnn_model = rnn_model.to(device)
        rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

        lstm_model = LSTMModel(input_dim, embedding_dim, hidden_dim, output_dim)
        lstm_model = lstm_model.to(device)
        lstm_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        num_epochs = 3
        for epoch in range(num_epochs):
            print('=== Epoch {}'.format(epoch))
            print('RNN Model')
            train_model(rnn_model, rnn_optimizer)
            with torch.set_grad_enabled(False):
                print(f'    training accuracy: '
                      f'{compute_binary_accuracy(rnn_model, train_iter, device):.2f}%'
                      f' | valid accuracy: '
                      f'{compute_binary_accuracy(rnn_model, test_iter, device):.2f}%')

            print('LSTM Model')
            train_model(lstm_model, lstm_optimizer)
            with torch.set_grad_enabled(False):
                print(f'    training accuracy: '
                      f'{compute_binary_accuracy(lstm_model, train_iter, device):.2f}%'
                      f' | valid accuracy: '
                      f'{compute_binary_accuracy(lstm_model, test_iter, device):.2f}%')
