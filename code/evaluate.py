from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from modules.training_loop import TrainingLoop
from modules.tokenize_dataset import TokenizeDataset
from modules.activation_tracker import LayerTracker, NeuronTracker
from modules.resource_logging_callback import ResourceLoggingCallback
from modules.set_seed import set_seed
from modules.threshold import Threshold
from copy import deepcopy
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_score
import json
from tqdm import tqdm
from matplotlib import pyplot as plt   
import matplotlib
import numpy as np
import seaborn as sns

set_seed(42)

def create_optimizer(model, config):
    optimizer_type = config["optimizer"]["type"]
    lr = config["optimizer"]["lr"]
    weight_decay = config["optimizer"].get("weight_decay", 0.0)

    if optimizer_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        momentum = config["optimizer"].get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
def compute_metrics(logits, labels):
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    p = precision_score(labels, predictions, average="micro")
    return {"accuracy": acc, "f1": f1, "precision": p}


def apply_pruning_to_model(model, neuron_tracker):
    device = next(model.parameters()).device  # 모델이 있는 디바이스 가져오기
    neurons_pruned = []
    for i in range(4):  # 총 4개 Layer
        layer_name = f"bert.encoder.layer.{i}.intermediate.dense"

        if layer_name in neuron_tracker.pruning_masks:
            pruning_mask = torch.tensor(neuron_tracker.pruning_masks[layer_name], dtype=torch.float32).to(device)  # 모델 디바이스로 이동
            active_neurons = int(pruning_mask.sum().item())  # 살아남은 뉴런 개수

            # 최소 1개 뉴런 유지 (모두 제거되는 경우 방지)
            if active_neurons == 0:
                active_neurons = 1
                pruning_mask[0] = 1  # 첫 번째 뉴런 유지

            # 1️⃣ intermediate.dense 업데이트
            layer = dict(model.named_modules())[layer_name]  # 기존 Layer 가져오기
            with torch.no_grad():
                pruned_weight = layer.weight[pruning_mask.bool(), :].to(device)  # CUDA로 이동
                pruned_bias = layer.bias[pruning_mask.bool()].to(device) if layer.bias is not None else None  # Bias도 CUDA로 이동
                
                # 새로운 Pruned Linear Layer 생성
                new_layer = nn.Linear(layer.in_features, active_neurons, bias=(layer.bias is not None)).to(device)  # CUDA로 이동
                new_layer.weight = nn.Parameter(pruned_weight)
                if pruned_bias is not None:
                    new_layer.bias = nn.Parameter(pruned_bias)

            # Pruned Layer를 모델에 적용
            parts = layer_name.split(".")
            submodule_name = parts[-1]
            parent_module = model
            for part in parts[:-1]:  
                parent_module = getattr(parent_module, part)
            setattr(parent_module, submodule_name, new_layer)

            print(f"Layer {i}: {layer.out_features} to {active_neurons} neurons")

            # 2️⃣ output.dense 업데이트 (intermediate.dense와 연결된 레이어)
            parts[4] = 'output'  # `intermediate` → `output`으로 변경
            layer_name = ".".join(parts)  # 새로운 레이어 이름 생성
            
            if layer_name in dict(model.named_modules()):  # `output.dense`가 존재하는 경우
                output_layer = dict(model.named_modules())[layer_name]
                
                with torch.no_grad():
                    original_weight = output_layer.weight.detach().cpu().numpy()
                    original_bias = output_layer.bias.detach().cpu().numpy() if output_layer.bias is not None else None

                    # output.dense의 입력 뉴런 개수 변경
                    pruned_weight = torch.tensor(original_weight[:, :active_neurons], dtype=torch.float32).to(device)  # CUDA로 이동
                    pruned_bias = torch.tensor(original_bias, dtype=torch.float32).to(device) if original_bias is not None else None  # CUDA 이동

                    # 새로운 Pruned Linear Layer 생성
                    new_output_layer = nn.Linear(active_neurons, output_layer.out_features, bias=(output_layer.bias is not None)).to(device)
                    new_output_layer.weight = nn.Parameter(pruned_weight)
                    if pruned_bias is not None:
                        new_output_layer.bias = nn.Parameter(pruned_bias)

                # Pruned Layer를 모델에 적용
                parts = layer_name.split(".")
                submodule_name = parts[-1]
                parent_module = model
                for part in parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, submodule_name, new_output_layer)
                neurons_pruned.append(2048 - active_neurons)
    return neurons_pruned
                # print(f"Layer {i}: intermediate dense to output dense: ({layer.out_features} {active_neurons} → {output_layer.out_features}) neurons")

            
            
            

def fine_tune_and_evaluate(config, callbacks=None):
    tokenizer = BertTokenizer.from_pretrained(config["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 다양한 Threshold 조합 테스트
    threshold_configs = {
        # "base": [0, 0, 0, 0],  # 기본값 (Pruning 없음)
        "[0.8,0.4, 0.4, 0.4]": [0.8, 0.4, 0.4, 0.4],
        "[0.4, 0.8, 0.4, 0.4]": [0.4, 0.8, 0.4, 0.4],
        "[0.4, 0.4, 0.8, 0.4]": [0.4, 0.4, 0.8, 0.4],
        "[0.4, 0.4, 0.4, 0.8]": [0.4, 0.4, 0.4, 0.8],
        "[0.8, 0.8, 0.8, 0.8]": [0.8, 0.8, 0.8, 0.8],
        "[0, 0, 0.6, 0]": [0, 0, 0.6, 0],
        "[0, 0, 0.4, 0]": [0, 0, 0.4, 0],
        "[0, 0, 0.2, 0]": [0, 0, 0.2, 0],
        "[0, 0.6, 0.8, 0]": [0, 0.6, 0.8, 0],
        "[0, 0.4, 0.6, 0]": [0, 0.4, 0.6, 0],
        "[0, 0.2, 0.4, 0]": [0, 0.2, 0.4, 0],
        # "0th": [0.8, 0, 0, 0],
        # "1st": [0, 0.8, 0, 0],
        # "2nd": [0, 0, 0.8, 0],
        # "3rd": [0, 0, 0, 0.8],

        # # 균등한 pruning (모든 Layer 동일)
        # "uniform": [0.12, 0.13, 0.14, 0.15],  

        # # 초반 pruning 증가 (초기 Layer를 더 많이 Pruning)
        # "early_more": [0.15, 0.14, 0.13, 0.12],  
        
        # # 후반 pruning 증가 (후반 Layer를 더 많이 Pruning)
        # "late_more": [0.12, 0.15, 0.22, 0.28],  

        # # 적응형 pruning (Layer별 변화량 고려)
        # "adaptive_1": [0.14, 0.12, 0.16, 0.20],  
        # "adaptive_2": [0.13, 0.14, 0.20, 0.25],  

        # # 가장 강력한 pruning (90th percentile 근처)
        # "aggressive": [0.10, 0.15, 0.25, 0.34],  

        # # **Linear Scaling (선형 증가)**
        # "linear_1": [0.10, 0.15, 0.20, 0.25],  # Layer별 점진적 증가
        # "linear_2": [0.08, 0.12, 0.16, 0.20],  

        # # **Focused Pruning (특정 Layer 집중)**
        # "high_start": [0.25, 0.18, 0.12, 0.10],  # 초반 Layer 강한 Pruning
        # "high_end": [0.08, 0.10, 0.15, 0.25],  # 후반 Layer 강한 Pruning

        # # **Exponential Pruning (지수적 감소)**
        # "exp_1": [0.20, 0.18, 0.12, 0.08],  # 초반 Pruning 강하게
        # "exp_2": [0.05, 0.10, 0.20, 0.30],  # 후반 Pruning 강하게 (지수적 증가)

        # # **백분위수 기반 pruning**
        # "percentile_85": [0.740, 0.769, 0.790, 0.686],  # 85th percentile
        # "percentile_90": [0.804, 0.847, 0.856, 0.731],  # 90th percentile
        # "percentile_95": [0.914, 0.972, 0.973, 0.811],  # 95th percentile

        # # **더 강력한 Pruning 추가**
        # "ultra_aggressive": [0.20, 0.30, 0.40, 0.50],  # 95th percentile 이상에서 pruning
        # "extreme": [0.25, 0.40, 0.55, 0.70],  # 거의 모든 뉴런 제거
        # "hard_exp": [0.10, 0.20, 0.40, 0.80],  # 지수적 pruning (후반부에서 거의 삭제됨)
        # "percentile_98": [1.10, 1.15, 1.20, 1.25],  # 98th Percentile 기반 pruning
    }

    results = {}  # 성능 결과 저장

    for config_name, threshold in threshold_configs.items():
        print(f"\nRunning with pruning threshold: {config_name} ({threshold})")

        # 원본 모델 로드
        model = BertForSequenceClassification.from_pretrained(
            config["model_path"],
            num_labels=2,
            output_hidden_states=True,
            output_attentions=True,
        )
        model.to(device)

        # print("Model Architecture:", model)
        trainer = TrainingLoop(callbacks=callbacks)
        activation_tracker = LayerTracker()
        activation_tracker.register_all_layers(model)
        neuron_tracker = NeuronTracker()
        neuron_tracker.register_all_layers(model)

        glue_dataset = TokenizeDataset(tokenizer, config["task_name"], config["batch_size"], dataset_name="glue")
        train_loader, eval_loader = glue_dataset.glue_data_loader()

        # print("Running 1st Forward Pass for Neuron Pruning...")
        model.eval()

        # 1st Forward Pass (현재 활성화 값 저장)
        for batch in train_loader:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
                _ = model(**batch)
            break  # 1회만 실행

        # print("Running 2nd Forward Pass to store previous activations...")
        for batch in train_loader:
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
                _ = model(**batch)
            break  # 1회만 실행

        # Pruning Mask 생성
        # Pruning Mask 생성 및 Layer별 백분위수 분석
        pruning_mask = []
        layerwise_changes = {}  # Layer별 변화량 저장

        for i in range(4):
            layer_name = f"bert.encoder.layer.{i}.intermediate.dense"
            if layer_name not in activation_tracker.activations:
                # print(f"⚠ Warning: {layer_name} activation is missing!")
                continue

            abs_change = neuron_tracker.calculate_neuron_changes(layer_name)
            if abs_change is None:
                # print(f"⚠ Warning: {layer_name} has no activation data. Skipping pruning.")
                pruning_mask.append(None)
                continue

            # Layer별 변화량 저장
            layerwise_changes[layer_name] = abs_change.flatten()

            # Pruning Mask 생성
            pruning_mask.append(neuron_tracker.generate_pruning_mask(layer_name, threshold[i]))
            # print(f"Layer {i}: Pruning Mask 생성 완료!")

        # # Layer별 백분위수 계산
        # percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]

        # # print("\n**Layer-wise Absolute Change Percentiles:**")
        # for layer_name, changes in layerwise_changes.items():
        #     if len(changes) > 0:
        #         changes = np.array(changes)  # NumPy 배열 변환
        #         percentile_values = np.percentile(changes, percentiles)

        #         # print(f"\n📌 Layer: {layer_name}")
        #         for p, v in zip(percentiles, percentile_values):
        #             # print(f"{p}th percentile: {v:.6f}")

        #     pruning_mask.append(neuron_tracker.generate_pruning_mask(layer_name, threshold[i]))
        #     # print(f"Layer {i}: Pruning Mask 생성 완료!")

        # Pruning 적용
        neuron_pruned = apply_pruning_to_model(model, neuron_tracker)

        # print("Running Fine-tuning...")
        # print(model)
        model.train()
        optimizer = create_optimizer(model, config)
        trainer._trigger_event("on_train_begin")

        train_loss = 0
        for epoch in range(config["num_epochs"]):
            trainer.running_loss = 0.0
            for batch in tqdm(train_loader, desc="Training", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = trainer.forward_pass(model, batch, device)
                trainer.backward_pass(loss, optimizer)

            epoch_loss = trainer.running_loss / len(train_loader)
            print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
            metrics = trainer.evaluation(model, eval_loader, device, compute_metrics)
            print(f"Epoch {epoch + 1} Metrics: {metrics}")

        trainer._trigger_event("on_train_end")

        # 결과 저장
        results[config_name] = {
            
            "threshold": threshold,
            # "train_loss": epoch_loss,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "neurons_pruned": neuron_pruned, 
            # "precision": metrics["precision"],
        }
        
    #   method, threshold, l0, l1, l2, l3, training time, allocated gpu mem, cached gpu memory, accuracy, f1-score
        

    # 모든 결과 정렬 후 출력
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    print("\n**Final Results (Sorted by Accuracy)**:")
    for name, res in sorted_results:
        print(f"{name}: Accuracy={res['accuracy']:.4f}, F1={res['f1']:.4f}, Pruning={res['threshold']}, Neurons Pruned={res['neurons_pruned']}")

    return activation_tracker, neuron_tracker

if __name__ == "__main__":
    with open("../config/evaluate_config.json", "r") as f:
        config = json.load(f)
    
    activation_tracker, neuron_tracker = fine_tune_and_evaluate(config, callbacks=[ResourceLoggingCallback()])
