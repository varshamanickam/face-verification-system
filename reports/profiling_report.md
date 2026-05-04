## Batch Size Sensitivity

To understand how runtime changes with workload size, profiling was repeated with different numbers of input pairs.

| Batch Size (pairs) | Mean End-to-End Latency per Pair (ms) |
|-------------------|----------------------------------------|
| 1                 | 3436.26                                |
| 8                 | 797.22                                 |
| 32                | 583.71                                 |

### Observations

The single-pair run had much higher latency because it includes more one-time setup and warmup overhead relative to the amount of work being measured. As the number of profiled pairs increased, the mean end-to-end latency per pair became more stable.

The 8-pair and 32-pair runs show that the system is still CPU-bound, with most runtime coming from the embedding stage. This makes sense because embedding generation includes image loading, face detection, face selection, and ArcFace inference. Scoring remains negligible across all batch sizes.