from pathlib import Path
import json

import pandas as pd


RESULTS_DIR = Path("results")


def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        results
    )

    csv_path = (
        RESULTS_DIR
        / "all_results.csv"
    )

    json_path = (
        RESULTS_DIR
        / "all_results.json"
    )

    excel_path = (
        RESULTS_DIR
        / "GAT_Research_Results.xlsx"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="All Results",
            index=False,
        )

        for dataset_name in (
            df["Dataset"]
            .unique()
        ):

            dataset_df = df[
                df["Dataset"]
                == dataset_name
            ]

            dataset_df.to_excel(
                writer,
                sheet_name=dataset_name[
                    :31
                ],
                index=False,
            )

        metric_columns = [
            "Validation Accuracy",
            "Test Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted F1",
        ]

        model_average = (
            df.groupby("Model")[
                metric_columns
            ]
            .mean()
            .reset_index()
        )

        model_average.to_excel(
            writer,
            sheet_name="Model Average",
            index=False,
        )

        dataset_average = (
            df.groupby("Dataset")[
                metric_columns
            ]
            .mean()
            .reset_index()
        )

        dataset_average.to_excel(
            writer,
            sheet_name="Dataset Average",
            index=False,
        )

        best_per_dataset = (
            df.loc[
                df.groupby(
                    "Dataset"
                )[
                    "Test Accuracy"
                ].idxmax()
            ]
            .reset_index(
                drop=True
            )
        )

        best_per_dataset.to_excel(
            writer,
            sheet_name="Best Per Dataset",
            index=False,
        )

    print()
    print("=" * 70)
    print("RESULT FILES")
    print("=" * 70)

    print(
        f"Excel : {excel_path}"
    )

    print(
        f"CSV   : {csv_path}"
    )

    print(
        f"JSON  : {json_path}"
    )