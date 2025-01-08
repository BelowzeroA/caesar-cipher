import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import string

from caesar.utils import encrypt_char, apply_caps

################################################################################
# HYPERPARAMETERS
################################################################################
EMBED_DIM = 32
HIDDEN_DIM = 64

################################################################################
# BUILD VOCAB
################################################################################
# For simplicity, here’s a small example set of characters:
ALL_CHARS = string.ascii_lowercase
char2idx = {ch: i for i, ch in enumerate(ALL_CHARS)}
idx2char = {i: ch for ch, i in char2idx.items()}
VOCAB_SIZE = len(char2idx)


class CaesarEncryptionDataset(Dataset):
    """
    Dataset class for Caesar cipher encryption data.

    Each sample consists of a character, a shift value, and the encrypted character.
    """

    def __init__(self):
        self.samples = []
        for shift in range(1, 26):
            for char in ALL_CHARS:
                encrypted_char = encrypt_char(char, shift)
                self.samples.append((char, shift, encrypted_char))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (x_enc, shift, y_enc), where:
                - x_enc (torch.Tensor): Encoded input character as a tensor.
                - shift (torch.Tensor): Shift value as a tensor.
                - y_enc (torch.Tensor): Encoded encrypted character as a tensor.
        """
        char, shift, encrypted_char = self.samples[idx]
        x_enc = torch.tensor(char2idx[char], dtype=torch.long)
        y_enc = torch.tensor(char2idx[encrypted_char], dtype=torch.long)
        shift = torch.tensor(shift, dtype=torch.long)

        return x_enc, shift, y_enc


class CaesarEncryptModel(nn.Module):
    """
    Neural network model for Caesar cipher encryption.
    This model predicts the encrypted character given an input character and a shift value.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        """
        Initializes the CaesarEncryptModel.

        Args:
            vocab_size (int): Size of the character vocabulary.
            embed_dim (int): Dimension of the embedding layers.
            hidden_dim (int): Dimension of the hidden layer in the MLP.
        """
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, embed_dim)
        self.shift_embed = nn.Embedding(26, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_chars, x_shifts):
        """
        Forward pass for the model.

        Args:
            x_chars (torch.Tensor): Tensor of character indices with shape [batch, seq_len].
            x_shifts (torch.Tensor): Tensor of shift values with shape [batch].

        Returns:
            torch.Tensor: Logits over the vocabulary for each character with shape [batch, seq_len, vocab_size].
        """
        char_emb = self.char_embed(x_chars)
        shift_emb = self.shift_embed(x_shifts)

        if len(char_emb.size()) == 3:
            shift_emb = shift_emb.unsqueeze(1).expand(-1, x_chars.size(1), -1)

        combined = torch.cat([char_emb, shift_emb], dim=-1)
        h = self.fc1(combined)
        h = self.relu(h)
        logits = self.fc2(h)

        return logits


class CaesarEncrypter:
    """
    Wrapper class for training, loading, and generating text with the CaesarEncryptModel.
    """

    def __init__(self):
        self.model = CaesarEncryptModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

    def load_model(self, path_to_model: str, device: torch.device):
        """
        Loads a pretrained model from a file.

        Args:
            path_to_model (str): Path to the model checkpoint.
            device (torch.device): Device to load the model onto.
        """
        checkpoint = torch.load(path_to_model, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)

    def train(self, train_loader, num_epochs, lr, device, validate_callback, save_path):
        """
        Trains the CaesarEncryptModel.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            device (torch.device): Device to train the model on.
            validate_callback (callable): Callback function for validation after each epoch.
            save_path (str): Path to save the trained model.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(device)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss, total_count = 0.0, 0
            for x_chars, x_shifts, y_enc in train_loader:
                x_chars = x_chars.to(device)
                x_shifts = x_shifts.to(device)
                y_enc = y_enc.to(device)

                optimizer.zero_grad()
                logits = self.model(x_chars, x_shifts)
                batch_size, vocab_size = logits.size()

                loss = criterion(logits, y_enc)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_size
                total_count += batch_size

            avg_train_loss = total_loss / total_count
            print(f"[Epoch {epoch+1}/{num_epochs}] train loss: {avg_train_loss:.4f}")

            if validate_callback:
                validate_callback(self)

            torch.save(self.model.state_dict(), save_path)

    def generate(self, plaintext: str, shift: int, device: torch.device):
        """
        Generates the ciphertext for a given plaintext using the model.

        Args:
            plaintext (str): Input plaintext string.
            shift (int): Shift value for Caesar cipher.
            device (torch.device): Device to perform generation on.

        Returns:
            str: Encrypted ciphertext string.
        """
        caps = [i for i, _ in enumerate(plaintext) if plaintext[i].isupper()]
        to_convert = {i: ch.lower() for i, ch in enumerate(plaintext) if ch.isalpha()}
        x_enc = [char2idx[ch] for ch in to_convert.values()]
        x_tensor = torch.tensor([x_enc], dtype=torch.long, device=device)

        shift_val = min(max(shift, 0), 25)
        shift_tensor = torch.tensor([shift_val], dtype=torch.long, device=device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor, shift_tensor)

        pred_ids = logits.argmax(dim=-1).squeeze(0)
        ciphertext = "".join(idx2char[idx.item()] for idx in pred_ids)

        result = []
        convert_char_counter = 0
        for i, source_char in enumerate(plaintext):
            if i in to_convert:
                result.append(ciphertext[convert_char_counter])
                convert_char_counter += 1
            else:
                result.append(source_char)

        result = "".join(result)
        result = apply_caps(result, caps)

        return result
