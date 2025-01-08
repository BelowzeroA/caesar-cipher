
# Neural Caesar Cipher

### Task Description

The objective of this project is to implement Caesar cipher encryption/decryption using a neural networks. 
This task basically falls into two independent subtasks - encryption and decryption, for which I explored different approaches and techniques

## Encryption task

Encryption task explores using a neural network to learn the mapping from plaintext to encrypted text based on the shift value 

### Approach

Instead of following a traditional neural network pipeline, my project leverages a simplified approach:
- A synthetic dataset is created for training, consisting of all possible shifts (1–25) for lowercase alphabetical characters.
- A neural network model predicts encrypted characters based on an input character and the shift value.
- The model overfits on the synthetic dataset, as generalization is not a concern for this use case.

Key components:
- **Dataset**: Synthetic dataset generated using the Caesar cipher logic, covering all shifts for the alphabet a-z.
- **Model**: A simple neural network with embedding layers for input characters and shifts, followed by fully connected layers to predict encrypted characters.
- **Training**: The model is trained on the synthetic dataset without standard practices like dataset splitting, metrics evaluation, or text corpus loading.

### Disclaimer

My approach intentionally omits:
- Dataset splitting (train/validation/test) and loading from external text corpora.
- Evaluation metrics for model performance - it quickly hits 100% in any metric.
- Generalization considerations (overfit is not a bad thing here), as the model's goal is to learn a straightforward mapping.

These choices are justified given the simplicity of the task and the synthetic nature of the dataset.

## Decryption task


### Approach
1. **Dataset Preparation**:
   - A raw corpus of text is cleaned and split into sentences.
   - Each sentence is encrypted using all possible Caesar cipher shifts (1 to 25).
   - A dataset of plaintext and ciphertext pairs is generated with a train/eval split.

2. **Model Fine-Tuning**:
   - The Flan-T5 model (`google/flan-t5-base`) is fine-tuned on the generated dataset using the Hugging Face `transformers` library.
   - Training optimizes the model to map ciphertext inputs to plaintext outputs.

3. **Evaluation**:
   - The fine-tuned model is evaluated on a separate test set.
   - Performance is measured using a custom accuracy metric based on overlapping words.

4. **Inference**:
   - The trained model is used to decrypt ciphertexts either individually or in batches.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/BelowzeroA/caesar-cipher.git
   cd caesar-cipher
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Training the encryption model

1. Run the training script:
   ```bash
   python encrypt_train.py
   ```

2. After training, the model checkpoint will be saved at the specified `SAVE_PATH`.

## Inference of the encryption model

To run inference:
1. Ensure you have a trained model checkpoint saved at `models/caesar_encrypter.pt`.
2. Run the inference script:
   ```bash
   python encrypt_infer.py
   ```
3. The script will print the original text and its encrypted version using both a standard Caesar cipher and the neural network model. It also checks for correctness.


## Preparing a dataset for the decryption model

1. Download the [C4 corpus file](https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz?download=true) and unpack it to `./data` directory, making sure you have this file `./data/c4-train.00000-of-01024.json` (it's 800MB in size)

2. Run the script:
   ```bash
   python prepare_dataset.py
   ```

3. After the script is done, the train dataset is at `data/decrypt_train.csv`.

## Training the decryption model

1. Run the training script:
   ```bash
   python decrypt_train.py
   ```

2. After training, the model checkpoint will be saved at `./models/flan-t5-caesar-decrypt`.

## Evaluation of the decryption model

To run evaluation:
1. If you want to evaluate my uploaded checkpoint, go ahead to 2. If you want to evaluate the freshly made model, replace 

```python 
model_path = "Belowzero/caesar-decrypt" 
``` 
with 
```python 
model_path = "./models/flan-t5-caesar-decrypt"
```

2. Run the evaluation script:
   ```bash
   python decrypt_eval.py
   ```
3. The script will print the accuracy score.
