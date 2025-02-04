import torch
import time

class ResourceLoggingCallback:
    def __init__(self, log_interval=1):
        self.log_interval = log_interval
        self.start_time = None

    def on_train_begin(self):
        """훈련 시작 시 호출되는 메서드."""
        self.start_time = time.time()
        print("Training started (via ResourceLoggingCallback)...")
        self.log_resource_usage("Start")

    def on_epoch_end(self, epoch):
        """각 epoch이 끝날 때 호출."""
        print(f"Epoch {epoch + 1} completed.")
        self.log_resource_usage(f"Epoch {epoch + 1}")

    def on_train_end(self):
        """훈련 종료 시 호출되는 메서드."""
        total_time = time.time() - self.start_time
        print(f"Training completed in: {total_time:.2f} seconds (via ResourceLoggingCallback)")
        self.log_resource_usage("End")

    def log_resource_usage(self, event):
        """현재 GPU 및 CPU 자원 사용량을 출력."""
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        cached_memory = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0
        
        print(f"[{event}] Allocated GPU Memory: {allocated_memory:.2f} MB")
        print(f"[{event}] Cached GPU Memory: {cached_memory:.2f} MB")

        if not torch.cuda.is_available():
            print("[INFO] GPU not available, using CPU.")