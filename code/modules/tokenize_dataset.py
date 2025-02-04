import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer

class TokenizeDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, task_name="sst2", batch_size=32, dataset_name="glue"):
        """
        GLUE 데이터셋을 로드하고 토큰화하는 클래스.

        Args:
            tokenizer (PreTrainedTokenizer): 사용할 토크나이저 (예: BertTokenizer).
            task_name (str, optional): GLUE 태스크 이름 (예: "sst2"). Defaults to "sst2".
            batch_size (int, optional): 배치 크기. Defaults to 32.
            dataset_name (str, optional): 데이터셋 이름. Defaults to "glue".
        """
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.batch_size = batch_size

        # 데이터 로드
        self.dataset = load_dataset(dataset_name, task_name)
        
        # 토큰화
        self.dataset = self.dataset.map(self._tokenize_function, batched=True)
        
        # 필요한 열 선택
        self.dataset = self.dataset.map(lambda x: {"labels": x["label"]}, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def _tokenize_function(self, examples):
        """ 입력 문장을 토큰화 """
        return self.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    def glue_data_loader(self):
        """Train 및 Eval DataLoader 생성"""
        train_loader = DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(self.dataset["validation"], batch_size=self.batch_size, shuffle=False)
        return train_loader, eval_loader

