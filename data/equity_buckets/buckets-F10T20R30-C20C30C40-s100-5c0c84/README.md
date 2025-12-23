# buckets-F10T20R30-C20C30C40-s100-5c0c84

## Configuration

```
Abstraction: buckets-F10T20R30-C20C30C40-s100-5c0c84
Type: equity_bucketing
Created: 2025-12-20T02:57:34.217031
Buckets: FLOP=10, TURN=20, RIVER=30
Board Clusters: FLOP=20, TURN=30, RIVER=40
MC Samples: 100
Seed: 42
Aliases: fast_test, test
```

## Usage

```python
from src.abstraction.equity_bucketing import EquityBucketing

bucketing = EquityBucketing.load('data/equity_buckets/buckets-F10T20R30-C20C30C40-s100-5c0c84/abstraction.pkl')
```
