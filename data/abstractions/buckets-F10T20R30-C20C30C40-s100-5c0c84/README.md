# buckets-F10T20R30-C20C30C40-s100-5c0c84

## Configuration

```
Abstraction: buckets-F10T20R30-C20C30C40-s100-5c0c84
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

bucketing = EquityBucketing.load('data/abstractions/buckets-F10T20R30-C20C30C40-s100-5c0c84_20251217_000849/abstraction.pkl')
```
