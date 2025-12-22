# buckets-F50T100R200-C200C500C1000-s1000-2a0e3c

## Configuration

```
Abstraction: buckets-F50T100R200-C200C500C1000-s1000-2a0e3c
Type: equity_bucketing
Created: 2025-12-17T04:23:37.767119
Buckets: FLOP=50, TURN=100, RIVER=200
Board Clusters: FLOP=200, TURN=500, RIVER=1000
MC Samples: 1000
Computation Time: 52.7 minutes
Workers: 12
File Size: 1552.9 KB
```

## Usage

```python
from src.abstraction.equity_bucketing import EquityBucketing

bucketing = EquityBucketing.load('data/abstractions/buckets-F50T100R200-C200C500C1000-s1000-2a0e3c_20251217_042337/abstraction.pkl')
```
