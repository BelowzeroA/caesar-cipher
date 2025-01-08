import pandas as pd
import os
from datasets import Dataset

from caesar.decrypt_modeling import CaesarDecrypter

os.environ["WANDB_DISABLED"] = "true"

# Load the dataset
data_path = "data/decrypt_train.csv"
base_model = "google/flan-t5-base"
output_dir = "./models/flan-t5"


def main():
    df = pd.read_csv(data_path)

    # Split the dataset into train and eval based on the 'split' column
    df_train = df[df['split'] == 'train']
    df_eval = df[df['split'] == 'eval']

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)

    decrypter = CaesarDecrypter(base_model)

    # Train the model
    decrypter.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
