from caesar.utils import caesar_encrypt
from caesar.decrypt_modeling import CaesarDecrypter

SHIFT = 11
MODEL_PATH = "Belowzero/caesar-decrypt"
VALID_TEXT = "Caesar is like: Veni, vidi, vici!"


def main():
    # Initialize model
    decrypter = CaesarDecrypter(MODEL_PATH)

    encrypted_text = caesar_encrypt(VALID_TEXT, SHIFT)
    decrypted_text = decrypter.generate(encrypted_text)

    print(f"{encrypted_text} => {decrypted_text}")

    if VALID_TEXT == decrypted_text:
        print("Decryption is correct!")
    else:
        print(f"Expected: {VALID_TEXT}, got: {decrypted_text}")


if __name__ == "__main__":
    main()