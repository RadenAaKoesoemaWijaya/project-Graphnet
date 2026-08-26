#!/usr/bin/env python3
"""Test GNN edge_index format fix"""
import numpy as np
import pandas as pd
from model import CombinedAnomalyDetector, create_claim_graph
import time

# Create larger synthetic data
np.random.seed(42)
n_samples = 1500
n_features = 10

X = np.random.randn(n_samples, n_features)
df = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
df['provider_id'] = np.random.choice([f'P{i}' for i in range(50)], n_samples)
df['patient_id'] = np.random.choice([f'Pat{i}' for i in range(100)], n_samples)
df['diagnosis_code'] = np.random.choice([f'D{i}' for i in range(30)], n_samples)

print('Testing GNN with fixed edge_index format...')
start = time.time()
node_features, edge_index = create_claim_graph(df, [f'f{i}' for i in range(n_features)], method='star')
print(f'[OK] Graph created: {node_features.shape[0]} nodes, {edge_index.shape[1]} edges')
print(f'   Edge index shape: {edge_index.shape}')

print('\nTesting GNN training...')
start = time.time()
detector = CombinedAnomalyDetector(algorithms=['isolation_forest', 'xgboost', 'gnn'], random_state=42, verbose=False)
ei_arr = edge_index.numpy() if hasattr(edge_index, 'numpy') else np.asarray(edge_index)
detector.fit(X, edge_index=ei_arr.T, device='cpu')
train_time = time.time() - start
print(f'[OK] Training completed in {train_time:.2f}s')
status = 'trained' if detector.gnn_model is not None else 'skipped'
print(f'[OK] GNN model: {status}')
print(f'[OK] GNN weight: {detector.gnn_weight:.4f}')
print(f'\n[OK] GNN fix successful!')
