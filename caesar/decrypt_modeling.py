import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

MAX_SEQ_LENGTH = 128
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CaesarDecrypter:
    """
    A class to train and utilize a T5 model for decrypting Caesar cipher text.
    """

    def __init__(self, path_to_model: str):
        """
        Initializes the CaesarDecrypter with a pre-trained T5 model and tokenizer.

        Args:
            path_to_model (str): Path to the pre-trained T5 model directory.
        """
        self.model = T5ForConditionalGeneration.from_pretrained(path_to_model)
        self.tokenizer = T5Tokenizer.from_pretrained(path_to_model)
        self.model.to(DEVICE)

    def preprocess_function(self, examples: dict) -> dict:
        """
        Preprocesses a dataset by tokenizing the ciphertext and plaintext.

        Args:
            examples (dict): A dictionary containing 'ciphertext' and 'plaintext' keys.

        Returns:
            dict: Tokenized inputs and labels for the model.
        """
        inputs = examples['ciphertext']
        targets = examples['plaintext']
        model_inputs = self.tokenizer(inputs, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length")
        labels = self.tokenizer(targets, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, train_dataset, eval_dataset, output_dir: str) -> None:
        """
        Trains the T5 model using the provided datasets.

        Args:
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            output_dir (str): The directory to save the trained model.
        """
        # Tokenize the datasets
        train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)

        # Data collator for padding during training
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=LR,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=3,
            save_steps=1000,
            predict_with_generate=True,
            fp16=True,
            logging_dir="./logs",
            logging_steps=1000,
        )

        # Initialize the Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()

        # Save the fine-tuned model
        trainer.save_model(output_dir)

    def generate(self, encrypted_text: str) -> str:
        """
        Decrypts a single Caesar cipher text.

        Args:
            encrypted_text (str): The encrypted text.

        Returns:
            str: The decrypted text.
        """
        input_ids = self.tokenizer.encode(encrypted_text, return_tensors="pt").to(DEVICE)

        outputs = self.model.generate(
            input_ids,
            max_length=MAX_SEQ_LENGTH,
            do_sample=False,
            num_beams=3,
            early_stopping=True
        )

        decrypted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decrypted_text

    def generate_batch(self, ciphertexts: list[str], batch_size: int = 16) -> list[str]:
        """
        Decrypts a batch of Caesar cipher texts.

        Args:
            ciphertexts (list[str]): A list of encrypted texts.
            batch_size (int): The batch size for decryption. Default is 16.

        Returns:
            list[str]: A list of decrypted texts corresponding to the input ciphertexts.
        """
        decrypted_texts = []
        for i in tqdm(range(0, len(ciphertexts), batch_size)):
            batch_slice = ciphertexts[i: i + batch_size]

            inputs = self.tokenizer(batch_slice, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=MAX_SEQ_LENGTH,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=True
                )

            decrypted_texts.extend(self.tokenizer.decode(output_ids, skip_special_tokens=True) for output_ids in outputs)

        return decrypted_texts
