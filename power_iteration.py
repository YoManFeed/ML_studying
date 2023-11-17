import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    random_array = np.random.rand(data.shape[0])
    for _ in range(num_steps):
        random_array = data @ random_array
        random_array = random_array / np.linalg.norm(random_array)

    eigenvector = random_array
    eigenvalue = random_array.T @ data @ random_array

    return float(eigenvalue), eigenvector