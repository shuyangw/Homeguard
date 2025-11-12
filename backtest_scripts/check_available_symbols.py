"""Check what symbols are available in the data directory."""

from pathlib import Path

DATA_DIR = Path('data/leveraged_etfs')

# Current optimal 20 symbols
CURRENT_SYMBOLS = [
    'FAZ', 'USD', 'UDOW', 'UYG', 'SOXL', 'TECL',
    'UPRO', 'SVXY', 'TQQQ', 'SSO', 'DFEN', 'WEBL',
    'UCO', 'FAS', 'TNA', 'LABU', 'SPXU', 'QLD', 'SQQQ', 'NAIL'
]

# Get all available symbols
files = list(DATA_DIR.glob('*_1d.parquet'))
symbols = sorted([f.stem.replace('_1d', '') for f in files if 'BACKUP' not in f.name])

# Remove SPY and VIX (used for market data, not trading)
symbols = [s for s in symbols if s not in ['SPY', '^VIX']]

# Find unused symbols
unused = [s for s in symbols if s not in CURRENT_SYMBOLS]

print("="*80)
print("AVAILABLE SYMBOLS ANALYSIS")
print("="*80)
print(f"\nTotal symbols available: {len(symbols)}")
print(f"Currently using: {len(CURRENT_SYMBOLS)}")
print(f"Available but unused: {len(unused)}")

print(f"\n{'='*80}")
print("CURRENTLY USED SYMBOLS (20)")
print("="*80)
for i, s in enumerate(CURRENT_SYMBOLS, 1):
    print(f"{i:2d}. {s}")

print(f"\n{'='*80}")
print(f"UNUSED SYMBOLS ({len(unused)})")
print("="*80)
if len(unused) > 0:
    for i, s in enumerate(unused, 1):
        print(f"{i:2d}. {s}")
else:
    print("All available symbols are being used!")

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)
print(f"Coverage: {len(CURRENT_SYMBOLS)}/{len(symbols)} symbols ({len(CURRENT_SYMBOLS)/len(symbols)*100:.1f}%)")
print(f"Unused potential: {len(unused)} symbols")
