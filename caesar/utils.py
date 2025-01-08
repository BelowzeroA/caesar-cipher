
def encrypt_char(char, shift):
    start = ord('A') if char.isupper() else ord('a')
    # Shift character
    offset = (ord(char) - start + shift) % 26
    new_char = chr(start + offset)
    return new_char


def caesar_encrypt(plaintext, shift):
    """
    Encrypts a plaintext string using a Caesar cipher with the given shift.
    """
    encrypted = []
    for char in plaintext:
        if 'a' <= char <= 'z':
            encrypted.append(encrypt_char(char, shift))
        elif 'A' <= char <= 'Z':
            encrypted.append(encrypt_char(char, shift))
        else:
            encrypted.append(char)  # Keep non-alphabet characters unchanged
    return ''.join(encrypted)


def apply_caps(s: str, caps: list[int]) -> str:
    # Convert the string to a list to allow modifications
    result = list(s)
    for index in caps:
        # Ensure the index is valid and within the bounds of the string
        if 0 <= index < len(s):
            result[index] = result[index].upper()
    # Join the list back into a string
    return ''.join(result)