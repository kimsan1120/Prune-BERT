from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer

class TokenizeDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, task_name="sst2", batch_size=32, dataset_name="glue"):
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.batch_size = batch_size
        self.dataset = load_dataset(dataset_name, task_name)
        self.dataset = self.dataset.map(self._tokenize_function, batched=True)
        self.dataset = self.dataset.map(lambda x: {"labels": x["label"]}, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    def glue_data_loader(self):
        train_loader = DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(self.dataset["validation"], batch_size=self.batch_size, shuffle=False)
        return train_loader, eval_loader

