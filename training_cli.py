"""Standalone training CLI for ASTINA.

This entry-point is meant to run *outside* the Streamlit UI, on long-lived
compute (Cloud Run Job, Vertex AI Custom Job, Kubernetes Job, your laptop,
...). It reads a CSV/Parquet/Excel file, fits a ``CombinedAnomalyDetector``,
runs Stratified K-Fold CV (optional), and writes the artefacts to either
the local filesystem or a GCS bucket (controlled by the same env vars as
``cloud_storage``).

Usage
-----
    python training_cli.py --data data/claims.csv \\
        --output-prefix models/fraud_detector \\
        --graph-method star \\
        --fraud-label-column fraud_label \\
        --cv-folds 5

Exit code is 0 on success and non-zero on error, suitable for Cloud Run
Jobs / Vertex AI failure handling.
"""
import argparse
import json
import os
import sys
import time
import traceback
from logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger("graphnet.train_cli")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ASTINA standalone training (Cloud Run Job / Vertex AI).",
    )
    p.add_argument("--data", required=True,
                   help="Path to the training data (CSV/Parquet/Excel).")
    p.add_argument("--output-prefix", default="models/fraud_detector",
                   help="Local prefix for the saved model artefacts.")
    p.add_argument("--graph-method", choices=["star", "knn", "heterogeneous"],
                   default="star")
    p.add_argument("--graph-k", type=int, default=5,
                   help="k for k-NN graph (only used when --graph-method=knn).")
    p.add_argument("--fraud-label-column", default="fraud_label",
                   help="Column name for the binary label (set to empty "
                        "string to train purely unsupervised).")
    p.add_argument("--cv-folds", type=int, default=0,
                   help="If >1, run Stratified K-Fold CV with this many "
                        "folds before fitting on the full data.")
    p.add_argument("--cv-refit", action="store_true",
                   help="Refit the detector on the full data after CV.")
    p.add_argument("--epochs-ae", type=int, default=100)
    p.add_argument("--epochs-gnn", type=int, default=200)
    p.add_argument("--gnn-hidden", type=int, default=64)
    p.add_argument("--gnn-heads", type=int, default=4)
    p.add_argument("--gnn-dropout", type=float, default=0.2)
    p.add_argument("--use-soft-labels", action="store_true",
                   help="Enable DevNet-style soft labels for the GNN.")
    p.add_argument("--devnet-weight", type=float, default=0.5)
    p.add_argument("--devnet-margin", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _load_dataframe(path: str):
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet",):
        return pd.read_parquet(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _select_feature_columns(df, fraud_label_column: str):
    """Drop obvious non-feature columns: labels, free-text, ids.

    Heuristic — kept conservative so we never accidentally include
    text columns in the numeric pipeline.
    """
    import pandas as pd
    drop = set()
    if fraud_label_column and fraud_label_column in df.columns:
        drop.add(fraud_label_column)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            drop.add(col)
    return [c for c in df.columns if c not in drop]


def main(argv=None) -> int:
    args = _parse_args(argv)
    t0 = time.time()

    try:
        import torch
        from model import CombinedAnomalyDetector, create_claim_graph
    except (Exception, OSError) as e:
        logger.error("Failed to import model: %s", e)
        return 2

    logger.info(
        "ASTINA training CLI | data=%s | output=%s | graph=%s | cv=%d | seed=%d",
        args.data, args.output_prefix, args.graph_method, args.cv_folds, args.seed,
    )

    try:
        df = _load_dataframe(args.data)
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return 3
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    feature_columns = _select_feature_columns(df, args.fraud_label_column)
    if not feature_columns:
        logger.error("No numeric feature columns found.")
        return 4
    logger.info("Feature columns: %s", feature_columns[:20])

    X = df[feature_columns].values
    labels = None
    if args.fraud_label_column and args.fraud_label_column in df.columns:
        labels = df[args.fraud_label_column].values
        logger.info("Label distribution: %s",
                    dict(zip(*__import__("numpy").unique(labels, return_counts=True))))

    # Build detector
    gnn_params = {
        "hidden_channels": args.gnn_hidden,
        "num_heads": args.gnn_heads,
        "dropout": args.gnn_dropout,
        "epochs": args.epochs_gnn,
        "use_soft_labels": bool(args.use_soft_labels),
        "devnet_weight": args.devnet_weight,
        "devnet_margin": args.devnet_margin,
    }
    detector = CombinedAnomalyDetector(
        isolation_forest_params={"contamination": 0.05, "n_estimators": 100},
        autoencoder_params={
            "encoding_dim": 32,
            "hidden_dims": [64, 48],
            "epochs": args.epochs_ae,
        },
        gnn_params=gnn_params,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional CV
    if args.cv_folds and labels is not None and len(set(labels)) > 1:
        try:
            edge_index = None
            if args.graph_method == "star":
                _, edge_index = create_claim_graph(df, feature_columns, method="star")
            cv_result = detector.cross_validate(
                X, labels, edge_index=edge_index,
                n_splits=int(args.cv_folds), device=device, refit=bool(args.cv_refit),
            )
            logger.info("CV summary: %s",
                        json.dumps(cv_result["summary"], indent=2, default=str))
        except Exception as e:
            logger.warning("CV step failed (continuing): %s", e)
            traceback.print_exc()

    # Final fit on full data
    try:
        detector.fit(X, labels=labels, device=device)
    except Exception as e:
        logger.error("Final fit failed: %s", e)
        traceback.print_exc()
        return 5

    # Build graph for GNN training (mirrors the Streamlit page)
    if "gnn" in detector.algorithms:
        try:
            graph_result = create_claim_graph(
                df, feature_columns, method=args.graph_method,
                **({"k": args.graph_k} if args.graph_method == "knn" else {}),
            )
            if isinstance(graph_result, tuple) and len(graph_result) == 3:
                node_features, edge_index, _ = graph_result
            else:
                node_features, edge_index = graph_result
            train_labels = labels if labels is not None else (
                detector.predict_anomaly_probability(X, device=device)[0] > 0.5
            ).astype(int)
            detector._train_gnn(node_features, edge_index, train_labels, device)
        except Exception as e:
            logger.warning("GNN training step failed: %s", e)
            traceback.print_exc()

    # Persist schema
    try:
        dtypes_map = {c: str(df[c].dtype) for c in feature_columns if c in df.columns}
    except Exception:
        dtypes_map = {}
    detector.training_metadata = {
        **(detector.training_metadata or {}),
        "feature_columns": list(feature_columns),
        "feature_dtypes": dtypes_map,
        "graph_method": args.graph_method,
    }

    artefacts = detector.save_models(args.output_prefix)
    logger.info("Saved %d artefacts to %s", len(artefacts), args.output_prefix)

    # R8: structured event so Cloud Logging log-based metrics fire.
    try:
        import metrics
        metrics.record_event("training_cli.completed",
                              n_artefacts=len(artefacts),
                              output_prefix=args.output_prefix,
                              graph_method=args.graph_method)
    except Exception:
        pass

    logger.info("Done in %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
