## Batch Size Sensitivity

To understand how runtime changes with workload size, profiling was repeated with different numbers of input pairs.

| Batch Size (pairs) | Mean End-to-End Latency per Pair (ms) |
|-------------------|----------------------------------------|
| 1                 | 3436.26                                |
| 8                 | 797.22                                 |
| 32                | 583.71                                 |

### Observations

The single pair run has much higher latency because it includes the initial setup and model loading overhead, but that cost is only being used for one pair. As a result, the average time per pair is much higher in that case.

As the number of pairs increases, the average latency per pair becomes more stable. With 8 and 32 pairs, the system has already done most of the setup work, so each additional pair takes less time on average.

Even with this change, the system is still clearly CPU-bound. Most of the runtime comes from the embedding stage, which includes image loading, face detection, face selection, and the ArcFace forward pass. Scoring remains negligible across all batch sizes.