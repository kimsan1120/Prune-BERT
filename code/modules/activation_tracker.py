import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LayerTracker:
    def __init__(self):
        self.activations = {}
        self.pruning_masks = {}
        self.activation_scores = {}

    def register_hook(self, module, name):
        def hook_function(module, input, output):
            if isinstance(output, tuple):
                context_layer, attention_weights = output
                self.activations[name] = context_layer.detach().cpu().numpy()
                self.activation_scores[name] = attention_weights.detach().cpu().numpy()
            elif isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu().numpy()
        module.register_forward_hook(hook_function)
        
    def register_all_layers(self, model):
        for name, module in model.named_modules():
            if "output.dense" in name or "intermediate.dense" in name or "pooler.dense" in name or "attention.self" in name:
                self.register_hook(module, name)

    def clear(self):
        self.activations = {}
        
    def get_activations(self, name):
        return self.activations.get(name, None)

class NeuronTracker:
    def __init__(self):
        self.activations = {}
        self.previous_activations = {}
        self.activation_scores = {}
        self.pruning_masks = {}

    def register_hook(self, module, name):
        def hook_function(module, input, output):
            if isinstance(output, tuple):
                context_layer, attention_weights = output
                self.activations[name] = context_layer.detach().cpu().numpy()
                self.activation_scores[name] = attention_weights.detach().cpu().numpy()
            elif isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu().numpy()
        module.register_forward_hook(hook_function)
        
    def register_all_layers(self, model):
        for name, module in model.named_modules():
            if "output.dense" in name or "intermediate.dense" in name or "pooler.dense" in name or "attention.self" in name:
                self.register_hook(module, name)
    def generate_pruning_mask(self, layer_name, threshold):
        abs_change = self.calculate_neuron_changes(layer_name)
        if abs_change is None:
            return None

        if not hasattr(self, 'pruning_masks'):
            self.pruning_masks = {}
        mask = (np.mean(abs_change, axis=0) >= threshold).astype(float)
        self.pruning_masks[layer_name] = mask  
        return mask            

    def calculate_neuron_changes(self, name):
        if name not in self.previous_activations:
            return None
        if name not in self.activations:
            return None
        prev = self.previous_activations[name]
        curr = self.activations[name]
        abs_change = np.abs(curr - prev)   
        return np.mean(abs_change, axis=0)  
    def clear(self):
        self.activations = {}
        self.previous_activations = {}
    def get_neuron_change(self, name):
        return self.activations.get(name, None)
