
import re
import pandas as pd

from caesar.decrypt_modeling import CaesarDecrypter

model_path = "Belowzero/caesar-decrypt"
data_path = "./data/decrypt_eval.csv"


def compute_sample_accuracy(ground_truth: str, predicted: str) -> float:
    """
    Computes the fraction of overlapping words (case-insensitive) between the
    ground truth and predicted text, ignoring any non-alphabetic characters.
      overlap_accuracy = (# overlapping words) / (# words in ground truth)
    """
    # Convert to lowercase
    gt_lower = ground_truth.lower()
    pred_lower = predicted.lower()

    # Tokenize on any non-alphabetic char
    gt_tokens = re.findall(r"[a-z]+", gt_lower)
    pred_tokens = re.findall(r"[a-z]+", pred_lower)

    if len(gt_tokens) == 0:
        # Edge case: if ground truth has 0 words, 100% if predicted is also empty
        return 1.0 if len(pred_tokens) == 0 else 0.0

    # Convert predicted tokens to a set for simple membership check
    pred_set = set(pred_tokens)
    overlap_count = sum(1 for token in gt_tokens if token in pred_set)

    return overlap_count / len(gt_tokens)


def main():
    decrypter = CaesarDecrypter(model_path)

    df = pd.read_csv(data_path)
    # df = df.sample(n=100)

    # ------------------------------------------------------------------------
    # 3. Batched inference
    # ------------------------------------------------------------------------
    # Collect all ciphertext for batch inference
    ciphertexts = list(df["ciphertext"])

    # Decrypt them in batches
    decrypted_list = decrypter.generate_batch(ciphertexts, batch_size=32)

    # ------------------------------------------------------------------------
    # 4. Accuracy calculation
    # ------------------------------------------------------------------------
    accuracies = []
    for sample, predicted_text in zip(df["plaintext"], decrypted_list):
        # Calculate accuracy against the ground-truth plaintext
        sample_accuracy = compute_sample_accuracy(sample, predicted_text)
        accuracies.append(sample_accuracy)

    # ------------------------------------------------------------------------
    # 5. Print final results
    # ------------------------------------------------------------------------
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f"Number of eval samples: {len(df)}")
    print(f"Average accuracy: {mean_accuracy:.4f}")


if __name__ == "__main__":
    main()
