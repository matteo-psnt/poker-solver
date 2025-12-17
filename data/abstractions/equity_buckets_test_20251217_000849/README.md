# equity_buckets_test

## Configuration

```
Abstraction: equity_buckets_test
Type: equity_bucketing
Created: 2025-12-17T00:08:49.896172
Buckets: FLOP=10, TURN=20, RIVER=30
Board Clusters: FLOP=20, TURN=30, RIVER=40
MC Samples: 100
Computation Time: 0.9 minutes
Workers: 12
File Size: 86.2 KB
```

## Usage

```python
from src.abstraction.equity_bucketing import EquityBucketing

bucketing = EquityBucketing.load('data/abstractions/equity_buckets_test_20251217_000849/bucketing.pkl')
```
