import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from .config import DATASET_ROOT, DEVICE


def load_dataset(dataset_name):
    dataset = Planetoid(
        root=DATASET_ROOT,
        name=dataset_name,
        split="public",
        transform=NormalizeFeatures(),
    )

    data = dataset[0].to(DEVICE)

    return dataset, data


def print_dataset_statistics(dataset_name, dataset, data):
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    print(f"Nodes          : {data.num_nodes}")
    print(f"Edges          : {data.edge_index.size(1)}")
    print(f"Node features  : {dataset.num_node_features}")
    print(f"Classes        : {dataset.num_classes}")
    print(f"Train nodes    : {int(data.train_mask.sum())}")
    print(f"Validation     : {int(data.val_mask.sum())}")
    print(f"Test nodes     : {int(data.test_mask.sum())}")
    print()