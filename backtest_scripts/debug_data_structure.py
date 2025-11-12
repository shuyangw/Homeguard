import pandas as pd
from pathlib import Path

# Load TQQQ
df = pd.read_parquet('data/leveraged_etfs/TQQQ_1d.parquet')
print(f'Original index type: {type(df.index)}')
print(f'Original index dtype: {df.index.dtype}')
print(f'Original index: {df.index[:3]}')
print(f'Original columns: {df.columns.tolist()}')

# Flatten columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

print(f'\nAfter flatten:')
print(f'Index type: {type(df.index)}')
print(f'Index dtype: {df.index.dtype}')
print(f'Index: {df.index[:3]}')
print(f'Columns: {df.columns.tolist()}')

# Try filtering
train_start = pd.Timestamp('2015-11-16')
train_end = pd.Timestamp('2017-11-16')
print(f'\nFiltering from {train_start} to {train_end}')
filtered = df[(df.index >= train_start) & (df.index <= train_end)]
print(f'Filtered rows: {len(filtered)}')
print(f'First row: {filtered.index[0]}')
