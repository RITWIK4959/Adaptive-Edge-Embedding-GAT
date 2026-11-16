import random
import time

import numpy as np
import torch

from .config import (
    SEED,
    DEVICE,
    DATASETS,
    CPU_NUM_THREADS,
    HIDDEN_CHANNELS,
    NUM_HEADS,
    EDGE_DIM,
    ATTENTION_DROPOUT,
    CLASSIFIER_DROPOUT,
    EDGE_ENCODER_HIDDEN,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
)

from .data_loader import (
    load_dataset,
    print_dataset_statistics,
)

from .edge_encoder import (
    static_edge_attributes,
)

from .models import (
    GATModel,
    GATv2Model,
    GATv2StaticEdge,
    GATv2LearnedEdge,
    AEGAT,
)

from .train import (
    train_model,
)

from .evaluate import (
    save_results,
)


EXPERIMENTS = (
    "GAT",
    "GATv2",
    "GATv2 + Static Edge",
    "GATv2 + Learned Edge",
    "AE-GAT",
)


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def build_model(
    model_name,
    num_features,
    num_classes,
):

    common = dict(
        in_channels=num_features,
        num_classes=num_classes,
        hidden_channels=HIDDEN_CHANNELS,
        heads=NUM_HEADS,
        attention_dropout=ATTENTION_DROPOUT,
        classifier_dropout=CLASSIFIER_DROPOUT,
        edge_dim=EDGE_DIM,
        edge_encoder_hidden=EDGE_ENCODER_HIDDEN,
    )

    if model_name == "GAT":

        return GATModel(
            **common
        )

    if model_name == "GATv2":

        return GATv2Model(
            **common
        )

    if (
        model_name
        == "GATv2 + Static Edge"
    ):

        return GATv2StaticEdge(
            **common
        )

    if (
        model_name
        == "GATv2 + Learned Edge"
    ):

        return GATv2LearnedEdge(
            **common
        )

    if model_name == "AE-GAT":

        return AEGAT(
            **common
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


class StaticEdgeWrapper(
    torch.nn.Module
):

    def __init__(
        self,
        model,
        edge_attr,
    ):
        super().__init__()

        self.model = model

        self.register_buffer(
            "edge_attr",
            edge_attr,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        return self.model(
            x,
            edge_index,
            edge_attr=self.edge_attr,
        )


def create_static_edge_features(
    data
):

    # Project original node features
    # into the same low-dimensional
    # space used by the GAT backbone.

    generator = (
        torch.Generator(
            device=data.x.device
        )
    )

    generator.manual_seed(
        SEED
    )

    weight = torch.randn(
        HIDDEN_CHANNELS * NUM_HEADS,
        data.num_features,
        generator=generator,
        device=data.x.device,
        dtype=data.x.dtype,
    )

    weight /= np.sqrt(
        data.num_features
    )

    projected_x = (
        data.x @ weight.T
    )

    edge_attr = (
        static_edge_attributes(
            projected_x,
            data.edge_index,
            EDGE_DIM,
        )
    )

    return edge_attr


def run_static_edge_model(
    data,
    num_features,
    num_classes,
):

    model = build_model(
        "GATv2 + Static Edge",
        num_features,
        num_classes,
    )

    edge_attr = (
        create_static_edge_features(
            data
        )
    )

    model = StaticEdgeWrapper(
        model,
        edge_attr,
    )

    return train_model(
        model,
        data,
        evaluate_test=True,
        epochs=EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def run_experiment(
    model_name,
    data,
    num_features,
    num_classes,
):

    if (
        model_name
        == "GATv2 + Static Edge"
    ):

        return run_static_edge_model(
            data,
            num_features,
            num_classes,
        )

    model = build_model(
        model_name,
        num_features,
        num_classes,
    )

    return train_model(
        model,
        data,
        evaluate_test=True,
        epochs=EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def run_dataset(
    dataset_name
):

    dataset, data = (
        load_dataset(
            dataset_name
        )
    )

    print_dataset_statistics(
        dataset_name,
        dataset,
        data,
    )

    num_features = (
        dataset.num_node_features
    )

    num_classes = (
        dataset.num_classes
    )

    dataset_results = []

    for model_name in (
        EXPERIMENTS
    ):

        print()
        print("=" * 70)
        print(
            f"{dataset_name} | "
            f"{model_name}"
        )
        print("=" * 70)

        start_time = (
            time.time()
        )

        result = (
            run_experiment(
                model_name,
                data,
                num_features,
                num_classes,
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        print()

        print(
            "Best validation "
            f"accuracy: "
            f"{result['validation_accuracy']:.4f}"
        )

        print(
            "Test accuracy: "
            f"{result['test_accuracy']:.4f}"
        )

        print(
            "Macro Precision: "
            f"{result['macro_precision']:.4f}"
        )

        print(
            "Macro Recall: "
            f"{result['macro_recall']:.4f}"
        )

        print(
            "Macro F1: "
            f"{result['macro_f1']:.4f}"
        )

        print(
            "Weighted F1: "
            f"{result['weighted_f1']:.4f}"
        )

        print(
            f"Parameters: "
            f"{result['parameters']:,}"
        )

        print(
            "Training time: "
            f"{result['training_time']:.2f}s"
        )

        print(
            "Best epoch: "
            f"{result['best_epoch']}"
        )

        print(
            f"Wall time: "
            f"{elapsed:.2f}s"
        )

        dataset_results.append(
            {
                "Dataset":
                    dataset_name,

                "Model":
                    model_name,

                "Validation Accuracy":
                    result[
                        "validation_accuracy"
                    ],

                "Validation Macro F1":
                    result[
                        "validation_macro_f1"
                    ],

                "Test Accuracy":
                    result[
                        "test_accuracy"
                    ],

                "Macro Precision":
                    result[
                        "macro_precision"
                    ],

                "Macro Recall":
                    result[
                        "macro_recall"
                    ],

                "Macro F1":
                    result[
                        "macro_f1"
                    ],

                "Weighted F1":
                    result[
                        "weighted_f1"
                    ],

                "Parameters":
                    result[
                        "parameters"
                    ],

                "Training Time":
                    result[
                        "training_time"
                    ],

                "Best Epoch":
                    result[
                        "best_epoch"
                    ],
            }
        )

    return dataset_results


def print_final_results(
    all_results
):

    print()

    print("=" * 120)
    print(
        "FINAL RESEARCH METRICS"
    )
    print("=" * 120)

    header = (
        f"{'Dataset':<12}"
        f"{'Model':<30}"
        f"{'Val':>8}"
        f"{'Test':>8}"
        f"{'Prec':>8}"
        f"{'Recall':>8}"
        f"{'F1':>8}"
        f"{'Params':>12}"
    )

    print(header)

    print("-" * 120)

    for row in all_results:

        print(
            f"{row['Dataset']:<12}"
            f"{row['Model']:<30}"
            f"{row['Validation Accuracy']:>8.4f}"
            f"{row['Test Accuracy']:>8.4f}"
            f"{row['Macro Precision']:>8.4f}"
            f"{row['Macro Recall']:>8.4f}"
            f"{row['Macro F1']:>8.4f}"
            f"{row['Parameters']:>12,}"
        )


def print_best_models(
    all_results
):

    print()

    print("=" * 70)
    print(
        "BEST MODEL PER DATASET"
    )
    print("=" * 70)

    for dataset_name in (
        DATASETS
    ):

        rows = [
            row
            for row in all_results
            if row["Dataset"]
            == dataset_name
        ]

        if not rows:
            continue

        best = max(
            rows,
            key=lambda row:
                row["Test Accuracy"],
        )

        print(
            f"{dataset_name:<12}"
            f"{best['Model']:<30}"
            f"{best['Test Accuracy']:.4f}"
        )


def print_ae_gat_results(
    all_results
):

    print()

    print("=" * 70)
    print(
        "AE-GAT RESULTS"
    )
    print("=" * 70)

    for dataset_name in (
        DATASETS
    ):

        rows = [
            row
            for row in all_results
            if (
                row["Dataset"]
                == dataset_name
                and row["Model"]
                == "AE-GAT"
            )
        ]

        if not rows:
            continue

        row = rows[0]

        print(
            f"{dataset_name:<12}"
            f"Test Accuracy: "
            f"{row['Test Accuracy']:.4f} | "
            f"Macro F1: "
            f"{row['Macro F1']:.4f}"
        )


def main():

    set_seed(
        SEED
    )

    torch.set_num_threads(
        CPU_NUM_THREADS
    )

    print()

    print("=" * 70)
    print(
        "ADAPTIVE EDGE-EMBEDDING GAT"
    )
    print("=" * 70)

    print(
        f"Device  : {DEVICE}"
    )

    print(
        f"Seed    : {SEED}"
    )

    print(
        f"Datasets: "
        f"{', '.join(DATASETS)}"
    )

    print(
        f"Models  : "
        f"{len(EXPERIMENTS)}"
    )

    print(
        f"Hidden  : "
        f"{HIDDEN_CHANNELS}"
    )

    print(
        f"Heads   : "
        f"{NUM_HEADS}"
    )

    print(
        f"Dropout : "
        f"{ATTENTION_DROPOUT}"
    )

    print("=" * 70)

    all_results = []

    for dataset_name in (
        DATASETS
    ):

        dataset_results = (
            run_dataset(
                dataset_name
            )
        )

        all_results.extend(
            dataset_results
        )

    print_final_results(
        all_results
    )

    print_best_models(
        all_results
    )

    print_ae_gat_results(
        all_results
    )

    print()

    print("=" * 70)
    print(
        "SAVING RESULTS"
    )
    print("=" * 70)

    save_results(
        all_results
    )

    print()

    print("=" * 70)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()