from hw1 import Composer
import numpy as np
import torch
from midi2seq import process_midi_seq, seq2piano
from torch.utils.data import DataLoader, TensorDataset

def main():
    # Define batch size and training epochs
    bsz = 32
    epoch = 1

    # Choose the device, use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MIDI sequence data and convert to Tensor
    piano_seq = torch.from_numpy(process_midi_seq(datadir='.')).to(device)  # Load data onto the device
    loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz, num_workers=0)  # Set num_workers=0 to avoid multithreading issues

    # Initialize the Composer model, no need to call .to(device) here
    cps = Composer()

    # Move the model to the specified device
    cps.model = cps.model.to(device)

    # Train the model
    for i in range(epoch):
        for x in loader:
            cps.train(x[0].to(device).long())  # Ensure the input data is on the correct device

    # Generate music
    midi = cps.compose(seq_len=600)
    midi = seq2piano(midi)
    midi.write('piano1.midi')
    print("Generated piano1.midi.")

if __name__ == "__main__":
    main()
