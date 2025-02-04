import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LayerTracker:
    def __init__(self):
        self.activations = {}
        self.pruning_masks = {}

    def register_hook(self, module, name):
        """Register forward hook to capture activations"""
        def hook_function(module, input, output):
            self.activations[name] = output.detach().cpu().numpy()

        module.register_forward_hook(hook_function)

    def register_all_layers(self, model):
        """Register hooks for all relevant layers"""
        for name, module in model.named_modules():
            if "output.dense" in name or "intermediate.dense" in name or "pooler.dense" in name:
                self.register_hook(module, name)

    def clear(self):
        self.activations = {}

    def get_activations(self, name):
        return self.activations.get(name, None)

class NeuronTracker:
    def __init__(self):
        """Track neuron changes between forward passes"""
        self.activations = {}
        self.previous_activations = {}

    def register_hook(self, module, name):
        """Register hook to capture neuron activations"""
        def neuron_hook_function(module, input, output):
            if name in self.activations:
                self.previous_activations[name] = self.activations[name]
            self.activations[name] = output.detach().cpu().numpy()

        module.register_forward_hook(neuron_hook_function)

    def register_all_layers(self, model):
        """Register hooks for all layers in the model"""
        for name, module in model.named_modules():
            if "output.dense" in name or "intermediate.dense" in name or "pooler.dense" in name:
                self.register_hook(module, name)
    def generate_pruning_mask(self, layer_name, threshold):
        """ 뉴런 Pruning Mask 생성 (hidden_size 크기로 맞춤) """
        abs_change = self.calculate_neuron_changes(layer_name)
        if abs_change is None:
            return None

        # ✅ pruning_masks가 없으면 초기화
        if not hasattr(self, 'pruning_masks'):
            self.pruning_masks = {}

        # ✅ 올바른 크기로 Mask 생성 (hidden_size,)
        mask = (np.mean(abs_change, axis=0) >= threshold).astype(float)  

        self.pruning_masks[layer_name] = mask  # Mask 저장
        return mask             
    def calculate_neuron_changes(self, name):
        """ 뉴런의 변화량을 계산 (Absolute 기준) """
        if name not in self.previous_activations or name not in self.activations:
            return None
        prev = self.previous_activations[name]
        curr = self.activations[name]
        abs_change = np.abs(curr - prev)  # (batch_size, seq_len, hidden_size)
        return np.mean(abs_change, axis=0)  # (hidden_size,)


    def clear(self):
        """Clear all stored activations"""
        self.activations = {}
        self.previous_activations = {}

    def get_neuron_change(self, name):
        """Retrieve stored neuron activations"""
        return self.activations.get(name, None)

    def visualize_heatmap(self, relative_change, layer_name):
        """Visualize the relative change in neuron activations as a heatmap"""
        plt.figure(figsize=(10, 10))
        sns.heatmap(relative_change, cmap="viridis", cbar=True)
        plt.title(f"Relative Change in Neuron Activations for Layer: {layer_name}")
        plt.xlabel("Hidden Size")
        plt.ylabel("Sequence Length")
        plt.savefig(f"neuron_heatmap_{layer_name}.png")
        plt.close()
