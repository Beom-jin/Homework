import torch
from model import Vanilla_RNN, Vanilla_LSTM
from dataset import Shakespeare

def generate(model, seed_characters, temperature, idx_to_char, char_to_idx, length=100):
    model.eval()
    input = torch.tensor([char_to_idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).cuda()
    hidden = model.init_hidden(1)
    predicted_chars = seed_characters
    vocab_size = len(idx_to_char)
    
    for _ in range(length):
        output, hidden = model(input, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Ensure the generated index is within the valid range
        if top_i.item() >= vocab_size:
            top_i = torch.tensor([vocab_size - 1]).cuda()
        
        predicted_char = idx_to_char[top_i.item()]
        predicted_chars += predicted_char
        input = torch.tensor([[top_i]], dtype=torch.long).cuda()
    
    return predicted_chars

def main():
    # Load the dataset to get character mappings
    dataset = Shakespeare('/home/idsl/sangbeom/homework/shakespeare_train.txt')
    idx_to_char = dataset.idx_to_char
    char_to_idx = dataset.char_to_idx
    
    input_size = len(char_to_idx)  # Should match the vocab size used during training
    hidden_size = 128
    output_size = input_size
    num_layers = 4

    # Load the model with the best validation performance
    model = Vanilla_RNN(input_size, hidden_size, output_size, num_layers).cuda()
    model.load_state_dict(torch.load('/home/idsl/sangbeom/homework/model_path_rnn.pth'))
    
    # Seed characters for generating samples
    seed_characters_list = ['The ', 'And ', 'But ', 'She ', 'He ']
    temperature = 0.7
    generated_samples = []

    for seed_characters in seed_characters_list:
        sample = generate(model, seed_characters, temperature, idx_to_char, char_to_idx, length=100)
        generated_samples.append(sample)
        print(f'Seed: "{seed_characters}"')
        print(sample)
        print()

if __name__ == '__main__':
    main()
