# C4, post hoc — each fresh calibration against the micro-benchmark cells

⚠️ **Written after the results were seen.** The registration listed this as a reported item (*"every cell's signed error against profile A, and against each fresh fit"*) and the registered analysis computed only profile A's half. Same files, the registration's own arithmetic (`scripts/c4_prepare.py`), no verdict.

Model against measured: above zero the model is too slow.

## Decode steps of batches that share a cached prefix

| b | summed context | measured | A | s0 | s1 | s2 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 11,552 | 81.3 ms | +2% | +5% | +5% | +5% |
| 12 | 17,328 | 102.7 ms | +2% | +5% | +4% | +4% |
| 16 | 23,104 | 122.2 ms | +3% | +7% | +6% | +6% |
| 32 | 46,208 | 207.0 ms | +3% | +7% | +5% | +5% |
| 64 | 92,416 | 342.3 ms | +12% | +17% | +15% | +15% |
| 96 | 138,624 | 494.1 ms | +13% | +18% | +16% | +16% |
| 128 | 184,832 | 634.1 ms | +15% | +20% | +18% | +18% |

The fitted cost of context, in ms per 1,000 tokens of the batch's summed context: A 3.50 · s0 3.83 · s1 3.76 · s2 3.76.

## What one arriving request costs the sequences already decoding

| b | measured extra gap | A | s0 | s1 | s2 |
|---:|---:|---:|---:|---:|---:|
| 8 | 179.3 ms | +3% | +1% | +6% | +5% |
| 32 | 154.4 ms | +19% | +17% | +23% | +22% |
| 64 | 171.7 ms | +7% | +5% | +10% | +10% |
