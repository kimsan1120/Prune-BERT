import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
class Threshold:
    matplotlib.use("Agg")

    def __init__(self, data):
        self.data = data  # 원본 데이터
        self.modified_data = None  # Threshold 적용 후 데이터
        

    def analyze_statistics(self):
        print(f"Mean: {np.mean(self.data):.4f}")
        print(f"Median: {np.median(self.data):.4f}")
        print(f"Standard Deviation: {np.std(self.data):.4f}")
        print(f"Min: {np.min(self.data):.4f}")
        print(f"Max: {np.max(self.data):.4f}")
        print(f"25th Percentile: {np.percentile(self.data, 25):.4f}")
        print(f"75th Percentile: {np.percentile(self.data, 75):.4f}")

    def plot_distribution(self):
        plt.figure(figsize=(12, 6))

        # 히스토그램
        plt.subplot(1, 2, 1)
        plt.hist(self.data.flatten(), bins=50, color="blue", alpha=0.7)
        plt.title("Histogram", fontsize=14)
        plt.xlabel("Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)

        # 박스플롯
        plt.subplot(1, 2, 2)
        plt.boxplot(self.data.flatten(), vert=True, patch_artist=True, showmeans=True)
        plt.title("Boxplot", fontsize=14)
        plt.ylabel("Value", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./eval_threshold/plot_distribution.png")
        plt.close()

    def apply_threshold(self, threshold):
        self.modified_data = np.copy(self.data)
        self.modified_data[self.modified_data < threshold] = 0
        print(f"Applied threshold: {threshold}")
        return self.modified_data

    def visualize_heatmap(self, layer_name="Layer", normalized=False):
        if self.modified_data is None:
            print("No threshold applied. Visualizing original data.")
            data_to_plot = self.data
        else:
            data_to_plot = self.modified_data

        if normalized:
            data_to_plot = (data_to_plot - np.min(data_to_plot)) / (np.max(data_to_plot) - np.min(data_to_plot) + 1e-8)
            print("Data normalized for heatmap.")

        plt.figure(figsize=(10, 10))
        sns.heatmap(data_to_plot, cmap="viridis", cbar=True)
        plt.title(f"Heatmap of {layer_name}", fontsize=16)
        plt.xlabel("Hidden Size", fontsize=14)
        plt.ylabel("Sequence Length", fontsize=14)
        plt.savefig(f"./eval_threshold/neuron_heatmap_{layer_name}.png")
        plt.close()

    def evaluate_with_threshold(self, threshold, evaluate_function):
        self.apply_threshold(threshold)
        performance = evaluate_function(self.modified_data)
        print(f"Performance with threshold {threshold}: {performance:.4f}")
        return performance

    def find_optimal_threshold(self, thresholds, evaluate_function):
        best_threshold = None
        best_performance = -np.inf

        performances = []

        for threshold in thresholds:
            performance = self.evaluate_with_threshold(threshold, evaluate_function)
            performances.append(performance)

            if performance > best_performance:
                best_performance = performance
                best_threshold = threshold

        # 성능 그래프 출력
        self.plot_performance(thresholds, performances)

        print(f"Optimal Threshold: {best_threshold}, Performance: {best_performance:.4f}")
        return best_threshold, best_performance

    def plot_performance(self, thresholds, performances):
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, performances, marker="o", linestyle="-", color="b")
        plt.title("Model Performance vs. Threshold", fontsize=16)
        plt.xlabel("Threshold", fontsize=14)
        plt.ylabel("Performance (Accuracy/F1-Score)", fontsize=14)
        plt.grid(True)
        plt.savefig(f"./eval_threshold/plot_performance.png")
        plt.close()
        
    def track_inactivate_neurons(self, data, thresholds):
        return np.all(data < thresholds, axis=0)

    def prune_neurons(self, weights, inactivate_neurons_mask):
        return weights[~inactivate_neurons_mask, :]

    def remove_neurons_from_layer(self, layer, inactivate_neurons_mask):
        weight = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

        active_neurons_mask = ~inactivate_neurons_mask
        pruned_weight = weight[active_neurons_mask, :]
        pruned_bias = bias[active_neurons_mask] if bias is not None else None

        new_layer = nn.Linear(pruned_weight.shape[1], pruned_weight.shape[0])
        new_layer.weight = nn.Parameter(torch.tensor(pruned_weight, dtype=torch.float32))
        if pruned_bias is not None:
            new_layer.bias = nn.Parameter(torch.tensor(pruned_bias, dtype=torch.float32))
        return new_layer

    def prune_model(self, model, threshold, layer_name):
        layer = dict(model.named_modules())[layer_name]

        # 활성화값 불러오기
        activations = np.load(f'../activations/{layer_name.replace(".", "_")}_activations.npy')

        # 비활성 뉴런 추적
        inactive_neurons_mask = self.track_inactivate_neurons(activations, threshold)

        # 뉴런 제거
        pruned_layer = self.remove_neurons_from_layer(layer, inactive_neurons_mask)

        # 모델 업데이트
        setattr(model, layer_name, pruned_layer)

        return model

    def get_activations(self, model, layer_name, input_data, device):
        """
        Retrieve activations from a specific layer of the model.

        Args:
            model (torch.nn.Module): The model.
            layer_name (str): Name of the layer to retrieve activations from.
            input_data (torch.Tensor): Input data for the model.
            device (torch.device): Device to run the model on.

        Returns:
            numpy.ndarray: Activations from the specified layer.
        """
        activations = []

        def hook(module, input, output):
            activations.append(output.detach().cpu().numpy())

        # 레이어 가져오기 및 Hook 등록
        layer = dict(model.named_modules())[layer_name]
        hook_handle = layer.register_forward_hook(hook)

        # 모델 실행
        model(input_data.to(device))

        # Hook 제거
        hook_handle.remove()

        # Activations 반환
        return activations[0]  # Activations 리스트의 첫 번째 요소 반환