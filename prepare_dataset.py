import json
import csv
import random

from tqdm import tqdm

from caesar.utils import caesar_encrypt

DATA_DIR = "./data"
raw_corpus_file = f"{DATA_DIR}/c4-train.00000-of-01024.json"
corpus_file = f"{DATA_DIR}/corpus.txt"
dataset_file = f"{DATA_DIR}/decrypt_train.csv"


def clean_and_split_text(
    text_file: str,
    min_length: int = 3,
    max_length: int = 30
) -> list[str]:
    """
    Reads a text file, cleans it by removing unwanted characters, and splits
    the text into a list of sentences. Sentences shorter or longer than specified
    bounds are filtered out.

    :param text_file: Path to the file containing the raw text.
    :param min_length: Minimum length of a sentence to include (in words).
    :param max_length: Maximum length of a sentence to include (in words).
    :return: List of cleaned sentences.
    """
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n")

    cleaned_sentences = []
    lines = lines[:1000000]
    forbidden_symbols = ["&apos", "▲", "�"]
    for sentence in lines:
        sentence = sentence.strip()
        words_num = len(sentence.split())
        if min_length > words_num or words_num > max_length:
            continue
        if any(fs for fs in forbidden_symbols if fs in sentence):
            continue
        cleaned_sentences.append(sentence)

    return cleaned_sentences


def convert_corpus(source_filename: str, target_filename: str) -> None:
    """
    Converts a JSONL corpus file into a plain text file by extracting the text field.

    :param source_filename: Path to the source JSONL file.
    :param target_filename: Path to the output text file.
    """
    with open(source_filename, 'r', encoding='utf-8') as file:
        json_raw = file.read()
    json_lines = json_raw.split("\n")
    result = []
    for line in json_lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        result.append(d.get("text", ""))

    with open(target_filename, 'w', encoding='utf-8') as file:
        for line in result:
            print(str(line).strip(), file=file)


def generate_dataset(
    sentences: list[str],
    train_ratio: float,
    output_csv: str,
) -> None:
    """
    Generates a dataset of plaintext and Caesar cipher pairs and writes it to a CSV file.

    :param sentences: List of plaintext sentences.
    :param train_ratio: Proportion of data to assign to the training set.
    :param output_csv: Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["plaintext", "ciphertext", "shift", "split"])
        writer.writeheader()

        for sentence in tqdm(sentences):
            # Using all possible shifts (1-25)
            for shift in range(1, 26):
                encrypted_sentence = caesar_encrypt(sentence, shift)

                # Decide train or eval split
                split_type = "train" if random.random() < train_ratio else "eval"

                row = {
                    "plaintext": sentence,
                    "ciphertext": encrypted_sentence,
                    "shift": shift,
                    "split": split_type
                }
                try:
                    writer.writerow(row)
                except Exception as e:
                    print(f"Error writing row: {e}")
                    continue

    print(f"Dataset saved to {output_csv}")


def main():
    """
    Main function to convert a corpus, clean and split it into sentences,
    and generate a Caesar cipher dataset.
    """
    # 1. Convert corpus to text file
    convert_corpus(raw_corpus_file, corpus_file)

    # 2. Clean and split text into sentences
    sentences = clean_and_split_text(corpus_file, min_length=5)
    print(f"Total sentences after cleaning: {len(sentences)}")

    # 3. Generate the dataset
    generate_dataset(
        sentences,
        train_ratio=0.99,
        output_csv=dataset_file,
    )


if __name__ == "__main__":
    main()
