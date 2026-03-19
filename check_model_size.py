import torch
import torch.nn as nn
from train import ConvTransformerModel
from data_segmentation_helpers.config import (
    SEQUENCE_LENGTH,
    CONV_EMBEDDING_DIM,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_DROPOUT,
    USE_CLS_TOKEN,
)

model = ConvTransformerModel(
    num_input_channels=18,
    num_classes=2,
    sequence_length=SEQUENCE_LENGTH,
    embed_dim=CONV_EMBEDDING_DIM,
    num_layers=TRANSFORMER_NUM_LAYERS,
    num_heads=TRANSFORMER_NUM_HEADS,
    ffn_dim=TRANSFORMER_FFN_DIM,
    dropout=TRANSFORMER_DROPOUT,
    use_cls_token=USE_CLS_TOKEN,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

conv_params = sum(p.numel() for p in model.conv_tower.parameters())
print(f"ConvTower Parameters: {conv_params:,}")

transformer_params = sum(p.numel() for p in model.transformer.parameters())
print(f"Transformer Parameters: {transformer_params:,}")

fc_params = sum(p.numel() for p in model.fc.parameters())
print(f"FC Head Parameters: {fc_params:,}")
