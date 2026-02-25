import numpy as np
import pytest

from src.similarity_metrics import (cosine_similarity_loop,
                                    cosine_similarity_vector,
                                    euclidean_distance_loop,
                                    euclidean_distance_vector)


##Testing correct shape output for all four functinos
def test_euclidean_v_output_shape():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(10,5))
    B = rng.normal(size=(10,5))
    result = euclidean_distance_vector(A, B)
    assert result.shape == (10, )

def test_euclidean_l_output_shape():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(10,5))
    B = rng.normal(size=(10,5))
    result = euclidean_distance_loop(A, B)
    assert result.shape == (10, )

def test_cosine_v_output_shape():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(10,5))
    B = rng.normal(size=(10,5))
    result = cosine_similarity_vector(A, B)
    assert result.shape == (10, )

def test_cosine_l_output_shape():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(10,5))
    B = rng.normal(size=(10,5))
    result = cosine_similarity_loop(A, B)
    assert result.shape == (10, )


##Testing if errors raised when not 2D and when A.shape != B.shape
def test_non_2D_input():
    A = np.zeros(5)
    B = np.zeros(5)
    with pytest.raises(ValueError):
        cosine_similarity_vector(A, B)

def test_AB_shape_mismatch():
    A = np.zeros((5, 3))
    B = np.zeros((6, 3))
    with pytest.raises(ValueError):
        euclidean_distance_vector(A, B)

## Testing loop vs vector correctness
def test_euclidean_loop_matches_vector():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(20, 15))
    B = rng.normal(size=(20, 15))
    e_loop = euclidean_distance_loop(A, B)
    e_vector = euclidean_distance_vector(A, B)

    assert np.allclose(e_loop, e_vector, atol=1e-9, rtol=1e-9)

def test_cosine_loop_matches_vector():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(20, 15))
    B = rng.normal(size=(20, 15))
    c_loop = cosine_similarity_loop(A, B)
    c_vector = cosine_similarity_vector(A, B)

    assert np.allclose(c_loop, c_vector, atol=1e-9, rtol=1e-9)


##Testing cosine math

#angle between two identical vectors is 0 so cosine similarity must be 1
def test_cosine_identical_vecs_is_one():
    rng = np.random.default_rng(42)
    A = rng.normal(size=(20, 15))
    B = A.copy()

    result = cosine_similarity_vector(A, B)
    assert np.allclose(result, np.ones(A.shape[0]), atol=1e-9)

#checking to see if no nans or inf for case of zero vectors (if epsilon addition works as it should)
def test_cosine_zero_vecs_no_nan_or_inf():
    A = np.zeros((5, 3))
    B = np.zeros((5, 3))
    result = cosine_similarity_vector(A, B)
    assert np.all(np.isfinite(result))
