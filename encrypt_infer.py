import torch

from caesar.utils import caesar_encrypt
from caesar.encrypt_modeling import CaesarEncrypter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SHIFT = 25
MODEL_PATH = "models/caesar_encrypter.pt"
VALID_TEXT = "Caesar is like: Veni, vidi, vici! abcdefghijklmnopqrstuvwxyz"


def main():

    # Initialize model
    encrypter = CaesarEncrypter()
    encrypter.load_model(MODEL_PATH, DEVICE)

    hard_encrypted = caesar_encrypt(VALID_TEXT, SHIFT)
    soft_encrypted = encrypter.generate(
        VALID_TEXT,
        SHIFT,
        DEVICE
    )

    print(f"{VALID_TEXT} => {soft_encrypted}")

    if hard_encrypted == soft_encrypted:
        print("Encryption is correct!")
    else:
        print(f"Expected: {hard_encrypted}, got: {soft_encrypted}")


if __name__ == "__main__":
    main()