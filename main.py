import torch
import torch.optim as optim
import torch.nn as nn
from dataset import Shakespeare
from model import RNN, LSTM
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt





def plot_metrics(train_losses, val_losses, model_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 10)) 
    plt.plot(epochs, train_losses, 'g', label='Training Loss')
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    f = f"{model_name}_loss.png"
    plt.savefig(f)    



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Shakespeare('./shakespeare_train.txt')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    input_size = len(dataset.char_to_idx)
    hidden_size = 128
    output_size = input_size
    num_layers = 4
    
    rnn = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    lstm = LSTM(input_size, hidden_size, output_size, num_layers).to(device)
    init_lr = 0.005
    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.Adam(rnn.parameters(),lr=init_lr)
    optimizer_lstm = optim.Adam(lstm.parameters(),lr=init_lr)
    
    num_epochs = 20

    best_val_loss_rnn = float('inf') 
    best_val_loss_lstm = float('inf') 
    
    metrics_rnn = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
    metrics_lstm = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}

    for epoch in range(num_epochs):
        train_loss_rnn, train_acc_rnn = train(rnn, train_loader, device, criterion, optimizer_rnn)
        val_loss_rnn, val_acc_rnn = validate(rnn, val_loader, device, criterion)
        metrics_rnn['train_losses'].append(train_loss_rnn)
        metrics_rnn['val_losses'].append(val_loss_rnn)
        metrics_rnn['train_accs'].append(train_acc_rnn)
        metrics_rnn['val_accs'].append(val_acc_rnn)
        print(f'Epoch {epoch+1}, Train Loss RNN: {train_loss_rnn}, Val Loss RNN: {val_loss_rnn}, Train Acc RNN: {train_acc_rnn}%, Val Acc RNN: {val_acc_rnn}%')

      
    
    for epoch in range(num_epochs):
        train_loss_lstm, train_acc_lstm = train(lstm, train_loader, device, criterion, optimizer_lstm)
        val_loss_lstm, val_acc_lstm = validate(lstm, val_loader, device, criterion)
        metrics_lstm['train_losses'].append(train_loss_lstm)
        metrics_lstm['val_losses'].append(val_loss_lstm)
        metrics_lstm['train_accs'].append(train_acc_lstm)
        metrics_lstm['val_accs'].append(val_acc_lstm)
        print(f'Epoch {epoch+1}, Train Loss LSTM: {train_loss_lstm}, Val Loss LSTM: {val_loss_lstm}, Train Acc LSTM: {train_acc_lstm}%, Val Acc LSTM: {val_acc_lstm}%')
    
    plot_metrics(metrics_rnn['train_losses'], metrics_rnn['val_losses'], 'RNN')
    plot_metrics(metrics_lstm['train_losses'], metrics_lstm['val_losses'], 'LSTM')
    
    
def train(model, data_loader, device, criterion, optimizer):
    model.train()
    total_loss, total_acc = 0, 0
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device).view(-1)
        hidden = model.init_hidden(x.size(0))
        optimizer.zero_grad()
        output, _ = model(x, hidden)
        loss = criterion(output, y)
        acc = accuracy(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(data_loader), total_acc / len(data_loader)

def validate(model, data_loader, device, criterion):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device).view(-1)
            hidden = model.init_hidden(x.size(0))
            output, _ = model(x, hidden)
            loss = criterion(output, y)
            acc = accuracy(output, y)
            total_loss += loss.item()
            total_acc += acc.item()
    return total_loss / len(data_loader), total_acc / len(data_loader)
    
def accuracy(predictions, labels):
    _, predicted = torch.max(predictions, dim=1)
    return ((predicted == labels).float().mean()) * 100  

if __name__ == '__main__':
    main()
