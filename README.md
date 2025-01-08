
# Neural Caesar Encryption

## Task Description

The objective of this project is to implement Caesar cipher encryption using a neural network. This task explores using a neural network to learn the mapping from plaintext to encrypted text based on the shift value.

## Approach

Instead of following a traditional neural network pipeline, my project leverages a simplified approach:
- A synthetic dataset is created for training, consisting of all possible shifts (1–25) for lowercase alphabetical characters.
- A neural network model predicts encrypted characters based on an input character and the shift value.
- The model overfits on the synthetic dataset, as generalization is not a concern for this use case.

Key components:
- **Dataset**: Synthetic dataset generated using the Caesar cipher logic, covering all shifts for the alphabet.
- **Model**: A simple neural network with embedding layers for input characters and shifts, followed by fully connected layers to predict encrypted characters.
- **Training**: The model is trained on the synthetic dataset without standard practices like dataset splitting, metrics evaluation, or text corpus loading.

### Disclaimer

My approach intentionally omits:
- Dataset splitting (train/validation/test) and loading from external text corpora.
- Evaluation metrics for model performance.
- Generalization considerations, as the model's goal is to learn a straightforward mapping.

These choices are justified given the simplicity of the task and the synthetic nature of the dataset.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Training

To train the model:
1. Run the training script:
   ```bash
   python encrypt_train.py
   ```

2. After training, the model checkpoint will be saved at the specified `SAVE_PATH`.

## Inference

To run inference:
1. Ensure you have a trained model checkpoint saved at `models/caesar_encrypter.pt`.
2. Run the inference script:
   ```bash
   python encrypt_infer.py
   ```
3. The script will print the original text and its encrypted version using both a standard Caesar cipher and the neural network model. It also checks for correctness.
