import torch
from torch.utils.data import DataLoader

from caesar.utils import caesar_encrypt
from caesar.encrypt_modeling import CaesarEncryptionDataset, CaesarEncrypter

BATCH_SIZE = 32
NUM_EPOCHS = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 5e-3
VALID_SHIFT = 21
SAVE_PATH = "models/caesar_encrypter.pt"
VALID_TEXT = "Hello, the world of encryption!"


def validate_callback(encrypter: CaesarEncrypter):
    hard_encrypted = caesar_encrypt(VALID_TEXT, VALID_SHIFT)
    soft_encrypted = encrypter.generate(
        VALID_TEXT,
        VALID_SHIFT,
        DEVICE
    )
    if hard_encrypted == soft_encrypted:
        print("Validation passed!")
    else:
        print(f"Expected: {hard_encrypted}, got: {soft_encrypted}")


def main():
    # Create Dataset objects
    train_dataset = CaesarEncryptionDataset()

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Initialize model
    encrypter = CaesarEncrypter()

    # Train
    encrypter.train(
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        device=DEVICE,
        validate_callback=validate_callback,
        save_path=SAVE_PATH
    )


if __name__ == "__main__":
    main()