# BERT Fine-Tuning and Pruning

## Overview
This repository provides a framework for fine-tuning and pruning a BERT model for sequence classification tasks. It includes dynamic neuron pruning based on activation tracking, optimization with different strategies, and resource monitoring during training. Below is a detailed breakdown of the main script and how it orchestrates pruning.

## Features
- Fine-tuning BERT for sequence classification tasks
- Dynamic pruning of attention and intermediate layers
- Activation tracking for neuron importance evaluation
- Customizable optimizer selection (AdamW, SGD)
- Resource logging and monitoring
- Dataset tokenization and data loading

## Performance Improvements
- **Model Size Reduction**: Reduced by **17.7%** compared to BERT-Small.
- **Performance Gain**: Improved **1.5%** on the GLUE SST-2 task (averaged over 10 different seeds).
- **Inference Cost Reduction**:
  - **GPU Memory Allocation** decreased by **12%**.
  - **Inference Time** decreased by **8%**.

## Detailed Code Analysis

### 1. Random Seed Setup
```python
set_seed(42)
```
Ensures experiment reproducibility across different runs.

### 2. Optimizer Creation
```python
def create_optimizer(model, config):
    # Supports AdamW or SGD with optional momentum & weight_decay
    ...
```
- Reads optimizer settings (type, lr, weight_decay, momentum) from JSON config.
- Returns the corresponding PyTorch optimizer.

### 3. Compute Metrics
```python
def compute_metrics(logits, labels):
    # Returns accuracy, F1-score, precision
    ...
```
- Leverages **scikit-learn** metrics to evaluate classification performance.
- Ideal for GLUE tasks (e.g., SST-2) with two-class classification.

### 4. Pruning Utilities
#### (1) `update_output_dense_layer`
```python
def update_output_dense_layer(model, layer_name, new_input_dim, device):
    # Updates the layer's weights and biases to match a pruned input dimension
    ...
```
- Locates a named layer (e.g., `bert.encoder.layer.X.attention.output.dense`) in the model.
- Modifies its weight & bias tensors to reflect the pruned dimension.
- Ensures dimension alignment after pruning.

#### (2) `apply_pruning_to_model`
```python
def apply_pruning_to_model(model_config, model, neuron_tracker):
    # Iterates over attention & intermediate layers
    # Adjusts hidden sizes and modifies them in-place
    ...
```
- Core pruning logic for **4 layers** of BERT-Small.
- **Attention Pruning**: `query`, `key`, `value` are pruned based on a pruning mask.
- **Intermediate Layer Pruning**: Reduces FFN (Fully Connected) dimensions.
- Updates the model’s `num_attention_heads` and the relevant dense layers.

### 5. Main Fine-Tuning & Evaluation
```python
def fine_tune_and_evaluate(model_config, config, callbacks=None):
    # 1) Load BERT, Tokenizer, & Dataset
    # 2) Run forward pass twice for activation tracking
    # 3) Compute absolute activation changes -> generate pruning masks
    # 4) Apply pruning & measure pruned model size
    # 5) Train & evaluate the pruned model
    ...
```
1. **Load model & tokenizer**: From a specified `model_path`.
2. **Register hooks** via `LayerTracker` and `NeuronTracker`.
3. **Forward propagation (twice)**: Gathers previous & current activations.
4. **Compute activation changes**: `abs_change = |current - previous|`.
5. **Generate pruning masks** for attention & intermediate layers.
6. **Apply pruning** with `apply_pruning_to_model`.
7. **Train & Evaluate**:
   - Uses `TrainingLoop` for multiple epochs.
   - Tracks metrics (accuracy, F1, precision) on a dev/test set.

### 6. Activation Trackers
```python
class LayerTracker:
    # Tracks activations of each layer

class NeuronTracker:
    # Stores previous & current activations to compute absolute changes
```
1. **Hook registration**:
   - Automatically hooks any layer named `output.dense`, `intermediate.dense`, `pooler.dense`, or `attention.self`.
   - Captures outputs (or `(context_layer, attention_weights)` if it’s attention).
2. **Pruning mask creation**:
   - `generate_pruning_mask` uses a threshold to convert activation changes into a binary mask.
   - Zero → prune, One → keep.

## How the Pruning Works
1. **Register Hooks**: Activation trackers attach a forward hook to relevant BERT layers.
2. **First Forward Pass**: Save `previous_activations`.
3. **Second Forward Pass**: Save `current_activations`.
4. **Calculate Absolute Changes**: `|current - previous|`.
5. **Prune**:
   - **Attention**: Prune `value` dimension, align `query` & `key` to that dimension.
   - **Intermediate**: Prune neurons in the feed-forward block.
6. **Update Model Structure**: Adjust linear layers and attention heads.

## Usage
1. **Install dependencies**:
```bash
pip install -r requirements.txt
```
2. **Configure** your settings:
   - `config/evaluate_config.json` (learning rate, batch size, epochs)
   - `config/bert_small_config.json` (hidden sizes, intermediate size, etc.)
3. **Run**:
```bash
python evaluate.py
```

## Code Structure
```
.
├── config/                         # Configuration files for bert small and evaluation for GLUE sst-2 task
├── modules/                        # Custom modules
│   ├── training_loop.py            # Training loop implementation
│   ├── tokenize_dataset.py         # Dataset tokenization
│   ├── activation_tracker.py       # Layer and neuron activation tracking
│   ├── resource_logging_callback.py# Resource logging during training
│   ├── set_seed.py                 # Random seed setup
├── evaluate.py                         # Entry point for training and pruning
└── README.md                       # Project documentation
```

## Metrics Computation
```python
def compute_metrics(logits, labels):
    # typical classification metrics: accuracy, F1, precision
    ...
```

## Examples
```python
attention_threshold_configs = {
    "attn_test": [0.0, 0.0, 0.3, 0.7],  # Layer-wise pruning thresholds for attention
}
intermediate_threshold_configs = {
    "interm_test": [0.0, 0.0, 0.8, 0.8],  # Layer-wise pruning thresholds for FFN layers
}
```

**This configuration leads to the following pruned BERT architecture:**

**Pruned Model Architecture (BERT for Sequence Classification)**
```
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 512, padding_idx=0)
      (position_embeddings): Embedding(512, 512)
      (token_type_embeddings): Embedding(2, 512)
      (LayerNorm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        ├── (0-1): 2 x BertLayer (No Pruning)
        │    ├── Attention: Query, Key, Value (512 → 512)
        │    ├── FFN Intermediate: (512 → 2048)
        │    ├── FFN Output: (2048 → 512)
        │
        ├── (2): BertLayer (Moderate Pruning)
        │    ├── Attention: Query, Key, Value (512 → 384)
        │    ├── FFN Intermediate: (512 → 193)
        │    ├── FFN Output: (193 → 512)
        │
        ├── (3): BertLayer (Heavy Pruning)
        │    ├── Attention: Query, Key, Value (512 → 64)
        │    ├── FFN Intermediate: (512 → 83)
        │    ├── FFN Output: (83 → 512)
      )
    )
    (pooler): BertPooler(
      (dense): Linear(512 → 512)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(512 → 2)
)
```