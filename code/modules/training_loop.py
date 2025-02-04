import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

class TrainingLoop:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks if callbacks is not None else []
        self.running_loss = 0.0

    def _trigger_event(self, event_name, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, event_name):
                getattr(callback, event_name)(*args, **kwargs)

    def forward_pass(self, model, batch, device):


        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.train()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        self.running_loss += loss.item()
        return loss

    def backward_pass(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def evaluation(self, model, eval_loader, device, compute_metrics):
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():

            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_logits, all_labels)
        return metrics

    def train(self, model, train_loader, eval_loader, optimizer, device, compute_metrics, num_epochs):
        for batch in train_loader:
            print("Batch Keys:", batch.keys())  # 🔍 batch의 키 확인
            break  # 한 번만 출력
        self._trigger_event("on_train_begin")
        
        for epoch in range(num_epochs):
            model.train()
            self.running_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
                loss = self.forward_pass(model, batch, device)
                self.backward_pass(loss, optimizer)

            epoch_loss = self.running_loss / len(train_loader)
            print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

            metrics = self.evaluation(model, eval_loader, device, compute_metrics)
            print(f"Epoch {epoch + 1} Metrics: {metrics}")
        
        self._trigger_event("on_train_end")

