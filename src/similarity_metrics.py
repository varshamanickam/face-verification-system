import numpy as np

EPSILON = 1e-12

def check_valid_input(A: np.ndarray, B: bp.ndarray) -> None:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D arrays, but received {A.ndim}D and {B.ndim}D")
    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape, but received {A.shape} and {B.shape}")
    

def euclidean_distance_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    """
    check_valid_input(A, B)
    N, D = A.shape
    result = np.empty(N, dtype=np.float64)

    for i in range(N):
        dist = 0.0
        for j in range(D):
            diff = float(A[i, j]) - float(B[i, j])
            dist += diff*diff
        result[i] = np.sqrt(dist)

    return result

def euclidean_distance_vector(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    """
    check_valid_input(A, B)
    diff = A - B
    return np.sqrt(np.sum(diff*diff, axis=1))

def cosine_similarity_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    """
    check_valid_input(A, B)
    N, D = A.shape
    result = np.empty(N, dtype=np.float64)

    for i in range(N):
        dot_product = 0.0
        a2_norm = 0.0
        b2_norm = 0.0
        for j in range(D):
            a = float(A[i, j])
            b = float(B[i, j])
            dot += a*b
            a2_norm += a*a
            b2_norm += b*b

        denominator = (np.sqrt(a2_norm) * np.sqrt(b2_norm)) + EPSILON
        result[i] = dot_product/denominator

def cosine_similarity_vector(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    """
    check_valid_input(A, B)
    dot_product = np.sum(A*B, axis=1)
    a_norm = np.sqrt(np.sum(A*A, axis=1))
    b_norm = np.sqrt(np.sum(B*B, axis=1))

    return dot_product/((a_norm * b_norm) + EPSILON)
