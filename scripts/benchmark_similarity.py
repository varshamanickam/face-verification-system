import json
import os
import time

import numpy as np

from src.similarity_metrics import (cosine_similarity_loop,
                                    cosine_similarity_vector,
                                    euclidean_distance_loop,
                                    euclidean_distance_vector)

ATOL = 1e-9
RTOL = 1e-9

def benchmark_call(func, *args, repeats: int=3):
    """
    Runs func(*args) and repeats times. Returns:
     - best_time_seconds: float
     - result: np.ndarray (the ouput from the best run)
    we only keep the best run for noise reduction
    """
    best_time = float("inf")
    best_result = None

    for _ in range(repeats):
        start_time = time.perf_counter()
        result = func(*args)
        time_elapsed = time.perf_counter() - start_time

        if time_elapsed < best_time:
            best_time = time_elapsed
            best_result = result
    
    return best_time, best_result

# Writing benchmark results to outputs/bench/ 
def write_benchmark_result(payload: dict, filename: str = "similarity_benchmark.json"):
    bench_dir = os.path.join("outputs", "bench")
    os.makedirs(bench_dir, exist_ok=True)
    out_path = os.path.join(bench_dir, filename)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote benchmark results to {out_path}")


def run_one_case(num_pairs: int, dim: int, seed: int=42):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(num_pairs, dim))
    B = rng.normal(size=(num_pairs, dim))

    #baseline times for euclidean and cosine loops
    e_loop_time, e_loop = benchmark_call(euclidean_distance_loop, A, B, repeats=3)
    c_loop_time, c_loop = benchmark_call(cosine_similarity_loop, A, B, repeats=3)

    #baseline times for euclidean and cosine vectorized
    e_vec_time, e_vec = benchmark_call(euclidean_distance_vector, A, B, repeats=3)
    c_vec_time, c_vec = benchmark_call(cosine_similarity_vector, A, B, repeats=3)
   
    #correctness check on the above times
    e_check = np.allclose(e_loop, e_vec, atol=ATOL, rtol=RTOL)
    c_check = np.allclose(c_loop, c_vec, atol=ATOL, rtol=RTOL)

    e_max_diff = float(np.max(np.abs(e_loop - e_vec)))
    c_max_diff = float(np.max(np.abs(c_loop - c_vec)))

    results = {
        "num_pairs": int(num_pairs),
        "dim": int(dim),
        "seed": int(seed),
        "repeats": 3,
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "euclidean": {
            "loop_time_s": float(e_loop_time),
            "vector_time_s": float(e_vec_time),
            "speedup_x": float(e_loop_time / e_vec_time) if e_vec_time > 0 else None,
            "allclose": bool(e_check),
            "max_abs_diff": float(e_max_diff),
        },
        "cosine": {
            "loop_time_s": float(c_loop_time),
            "vector_time_s": float(c_vec_time),
            "speedup_x": float(c_loop_time / c_vec_time) if c_vec_time > 0 else None,
            "allclose": bool(c_check),
            "max_abs_diff": float(c_max_diff),
        },
    }

    #need to use uniqeu filename per case so runs don't overwrite each other
    write_benchmark_result(results, filename=f"similarity_benchmark_np{num_pairs}_d{dim}.json")

    print(f"\nTest case: num_pairs={num_pairs:,}, dims={dim}")
    print(f"Euclidean    loop = {e_loop_time:.4f}s vector={e_vec_time:.4f}s   speedup={e_loop_time/e_vec_time:.2f}x   max_diff = {e_max_diff:.3e}  euclidean_matches={e_check}")
    print(f"Cosine sim    loop = {c_loop_time:.4f}s vector={c_vec_time:.4f}s   speedup={c_loop_time/c_vec_time:.2f}x   max_diff = {c_max_diff:.3e}  cosine_matches={c_check}")

    #let us know it failed if there's a mismatch
    assert e_check, f"Euclidean mismatch: max_diff={e_max_diff}"
    assert c_check, f"Cosine mismatch: max_diff={c_max_diff}"

def main():
    run_one_case(num_pairs=10_000, dim=128, seed=42)
    run_one_case(num_pairs=10_000, dim=512, seed=42)

if __name__ == "__main__":
    main()