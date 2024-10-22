import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import pretty_midi
import gdown
from midi2seq import piano2seq
import math

# Define the model
class MusicTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, hidden_dim, vocab_size, max_seq_len=512):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Positional encoding
        self.positional_encoding = self.get_positional_encoding(max_seq_len, d_model)
        
    def get_positional_encoding(self, max_seq_len, d_model):
        # Create a positional encoding matrix following the sin-cos functions from the original paper
        positional_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        positional_encoding = positional_encoding.unsqueeze(0)  # Add batch dimension
        return positional_encoding

    def forward(self, x, src_mask=None, src_padding_mask=None):
        x = self.embedding(x)
        
        # Add positional encoding to the input embedding
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)
        
        x = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_padding_mask)
        x = self.fc_out(x)
        return x

# Generate square subsequent mask to prevent attending to future tokens
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# Create padding mask to ignore padding tokens in attention
def create_padding_mask(x, pad_token=0):
    return (x == pad_token)

# Preprocess MIDI files and convert to sequences, truncate to max_seq_len
def preprocess_midi(midi_files, max_seq_len=512):
    sequences = []
    labels = []
    for midi_file in midi_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            seq = piano2seq(midi_data)
            
            # Ensure that seq is a 1D array (event sequence)
            if len(seq.shape) == 1:
                seq = torch.tensor(seq, dtype=torch.long)
            
            # Truncate sequence if necessary
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]

            # Prepare input and label sequences (shifted by one step)
            input_seq = seq[:-1]
            label_seq = seq[1:]

            sequences.append(input_seq)
            labels.append(label_seq)

            print(f"Processed file: {midi_file}, Sequence length: {len(seq)}")
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
    return sequences, labels

# Pad sequences to the same length and convert to DataLoader
def get_data_loader(batch_size=32, midi_files=None, max_seq_len=512):
    sequences, labels = preprocess_midi(midi_files, max_seq_len)
    
    # Pad sequences to the same length
    tensor_data = pad_sequence(sequences, batch_first=True)
    tensor_labels = pad_sequence(labels, batch_first=True)

    # Create DataLoader
    dataset = TensorDataset(tensor_data, tensor_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# Train the model
def train_model(model, data_loader, epochs=10, lr=0.0001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"Starting epoch {epoch+1}/{epochs}")

        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Generate mask and padding mask
            seq_len = inputs.size(1)
            src_mask = generate_square_subsequent_mask(seq_len).to(device)
            src_padding_mask = create_padding_mask(inputs).to(device)

            outputs = model(inputs, src_mask=src_mask, src_padding_mask=src_padding_mask)

            # Reshape outputs and labels for CrossEntropyLoss
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))  # Flatten the tensors for loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print batch training details
            print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(data_loader)}, "
                  f"Batch Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs} completed with average loss: {avg_epoch_loss:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), 'music_transformer.pth')
    print("Model saved to music_transformer.pth")

# Load MIDI files from directory
def load_midi_files(directory, limit=50):
    midi_files = []
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                midi_path = os.path.join(root, file)
                midi_files.append(midi_path)
                count += 1
                if count >= limit:
                    break
        if count >= limit:
            break
    return midi_files

# Composer class with the model and compose method
class Composer:
    def __init__(self, d_model=256, num_heads=8, num_layers=6, hidden_dim=512, vocab_size=382, max_seq_len=512, load_trained=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MusicTransformer(d_model, num_heads, num_layers, hidden_dim, vocab_size, max_seq_len).to(self.device)

        if load_trained:
            self.download_and_load_weights()

    def download_and_load_weights(self):
        # Path to download pretrained model
        weights_path = './music_transformer.pth'

        # Download weights if the file does not exist locally
        if not os.path.exists(weights_path):
            print("Downloading trained weights...")
            url = "https://drive.google.com/uc?id=1fogAkT5P4dNOpzFjw355qDeILU5lfRoC"
            gdown.download(url, weights_path, quiet=False)

        # Load the weights
        print("Loading trained weights...")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        print("Weights loaded successfully.")

    def train(self, batch):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        self.model.train()

        inputs, labels = batch.to(self.device), batch.to(self.device)

        optimizer.zero_grad()
        outputs = self.model(inputs)

        # Reshape outputs and labels for CrossEntropyLoss
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))  # Flatten the tensors for loss
        loss.backward()
        optimizer.step()

        print(f"Batch Loss: {loss.item():.4f}")

    def compose(self, seq_len, temperature=1.0):
        # Randomly generate an initial input
        generated_sequence = [torch.randint(0, self.model.embedding.num_embeddings, (1,), dtype=torch.long).item()]
        self.model.eval()

        with torch.no_grad():
            for _ in range(seq_len):
                input_tensor = torch.tensor([generated_sequence[-1]], dtype=torch.long).to(self.device)
                input_tensor = input_tensor.unsqueeze(0)

                output = self.model(input_tensor)

                output = output.squeeze() / temperature
                output = output.exp()

                # Sample the next token from the output distribution
                next_token = torch.multinomial(output, 1).item()

                # Ensure next_token is within vocab_size range
                next_token = max(0, min(next_token, self.model.embedding.num_embeddings - 1))


                # Append the generated token to the sequence
                generated_sequence.append(next_token)
                
        return np.array(generated_sequence)

    def save_to_midi(self, sequence, output_file='output.midi'):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # 0 for piano
        start = 0
        for pitch in sequence:  # Directly iterate over the sequence
            pitch = int(pitch) % 128  # Ensure the pitch is in the range 0 to 127
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=start, end=start+0.5)
            instrument.notes.append(note)
            start += 0.5  # Adjust note length as needed
        midi.instruments.append(instrument)
        midi.write(output_file)
        print(f"MIDI file saved as {output_file}")

# Main execution
if __name__ == "__main__":
    load_trained = True  # Modify this to decide whether to load pretrained weights
    composer = Composer(d_model=256, num_heads=8, num_layers=6, hidden_dim=512, vocab_size=382, max_seq_len=512, load_trained=load_trained)

    if not load_trained:
        # If pretrained model is not loaded, proceed with training
        dataset_dir = './maestro-v1.0.0'
        midi_files = load_midi_files(dataset_dir, limit=50)
        data_loader = get_data_loader(batch_size=32, midi_files=midi_files, max_seq_len=512)
        train_model(composer.model, data_loader, epochs=10, lr=0.0001, device=composer.device)
    else:
        # Once pretrained model is loaded, generate music and save as a MIDI file
        print("Model loaded. Generating music...")
        midi_sequence = composer.compose(seq_len=100, temperature=1.5)
        print(f"Generated MIDI sequence: {midi_sequence}")
        composer.save_to_midi(midi_sequence, output_file='generated_music.midi')
