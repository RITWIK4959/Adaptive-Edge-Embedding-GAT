import copy
import time

import torch
from torch import nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .config import (
    DEVICE,
    LABEL_SMOOTHING,
    GRADIENT_CLIP,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    MIN_LEARNING_RATE,
)


def classification_metrics(
    y_true,
    y_pred,
):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    data,
    mask,
):
    model.eval()

    logits = model(
        data.x,
        data.edge_index,
    )

    predictions = (
        logits[mask]
        .argmax(dim=-1)
        .cpu()
        .numpy()
    )

    labels = (
        data.y[mask]
        .cpu()
        .numpy()
    )

    return classification_metrics(
        labels,
        predictions,
    )


def train_model(
    model,
    data,
    evaluate_test=True,
    epochs=200,
    patience=40,
    learning_rate=0.005,
    weight_decay=5e-4,
):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
        )
    )

    best_val_accuracy = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
        )

        loss = criterion(
            logits[data.train_mask],
            data.y[data.train_mask],
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        train_pred = (
            logits[data.train_mask]
            .argmax(dim=-1)
        )

        train_accuracy = (
            (
                train_pred
                == data.y[
                    data.train_mask
                ]
            )
            .float()
            .mean()
            .item()
        )

        val_metrics = evaluate(
            model,
            data,
            data.val_mask,
        )

        val_accuracy = (
            val_metrics["accuracy"]
        )

        val_f1 = (
            val_metrics["macro_f1"]
        )

        scheduler.step(
            val_accuracy
        )

        improved = (
            val_accuracy
            > best_val_accuracy
            or (
                abs(
                    val_accuracy
                    - best_val_accuracy
                ) < 1e-12
                and val_f1
                > best_val_f1
            )
        )

        if improved:

            best_val_accuracy = (
                val_accuracy
            )

            best_val_f1 = val_f1

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Train {train_accuracy:.4f} | "
                f"Val {val_accuracy:.4f} | "
                f"Val F1 {val_f1:.4f} | "
                f"LR {current_lr:.6f}"
            )

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    training_time = (
        time.time()
        - start_time
    )

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    best_val_metrics = evaluate(
        model,
        data,
        data.val_mask,
    )

    if evaluate_test:

        test_metrics = evaluate(
            model,
            data,
            data.test_mask,
        )

    else:

        test_metrics = {
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "validation_accuracy":
            best_val_metrics[
                "accuracy"
            ],

        "validation_macro_f1":
            best_val_metrics[
                "macro_f1"
            ],

        "test_accuracy":
            test_metrics[
                "accuracy"
            ],

        "macro_precision":
            test_metrics[
                "macro_precision"
            ],

        "macro_recall":
            test_metrics[
                "macro_recall"
            ],

        "macro_f1":
            test_metrics[
                "macro_f1"
            ],

        "weighted_f1":
            test_metrics[
                "weighted_f1"
            ],

        "parameters":
            parameters,

        "training_time":
            training_time,

        "best_epoch":
            best_epoch,
    }