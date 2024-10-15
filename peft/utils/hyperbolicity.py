import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

def euclidean_hyperbolicity(points):
    # Calculate the pairwise Euclidean distances between all points
    distances = squareform(pdist(points, 'euclidean'))
    
    n = points.shape[0]
    max_hyp = 0

    # Iterate over all combinations of four distinct points
    for combo in combinations(range(n), 4):
        # Extract the distances for the selected points
        d = [distances[combo[i], combo[j]] for i in range(4) for j in range(i+1, 4)]

        # Sort the distances to calculate hyperbolicity
        d.sort()
        hyp = (d[4] + d[5] - d[2] - d[3]) / 2  # Corresponds to the delta value in the definition of hyperbolicity
        
        # Update the maximum hyperbolicity found
        max_hyp = max(max_hyp, hyp)

    return max_hyp

def hyperbolicity_sample_euclidean(points, num_samples=5000):
    distances = squareform(pdist(points, 'euclidean'))
    diam = np.max(distances)  # Diameter of the set of points
    n = points.shape[0]
    max_hyp = 0

    for _ in range(num_samples):
        combo = np.random.choice(n, 4, replace=False)
        d = [distances[combo[i], combo[j]] for i in range(4) for j in range(i+1, 4)]
        d.sort()
        hyp = (d[4] + d[5] - d[2] - d[3]) / 2
        max_hyp = max(max_hyp, hyp)

    return 2 * max_hyp / diam  # Normalized hyperbolicity

def multiple_trials_hyperbolicity(points, num_samples=5000, num_trials=10):
    hyperbolicities = []

    for _ in range(num_trials):
        hyp = hyperbolicity_sample_euclidean(points, num_samples)
        hyperbolicities.append(hyp)

    mean_hyperbolicity, std_hyperbolicity = np.mean(hyperbolicities), np.std(hyperbolicities)
    return mean_hyperbolicity, std_hyperbolicity

def mean_hyperbolicity_per_batch(points_tensor, num_samples=50000, num_trials=10):
    np.random.seed(42)
    batch_hyperbolicities_means = []

    for batch_idx in range(points_tensor.shape[0]):
        # Extract points for the current batch
        points = points_tensor[batch_idx]
        
        # Compute the mean hyperbolicity for the current batch across multiple trials
        mean_hyperbolicity, _ = multiple_trials_hyperbolicity(points, num_samples, num_trials)
        batch_hyperbolicities_means.append(mean_hyperbolicity)
    
    return batch_hyperbolicities_means

if __name__ == "__main__":
    # Example: Generate a random set of points in a 3D space (d = 3) for demonstration
    # np.random.seed(42)  # For reproducibility
    n, d = 256, 4096  # Number of points and dimensions
    points = np.random.rand(4, n, d)  # Generate random points

    # Compute hyperbolicity
    import time
    start_time = time.time()
    mean = mean_hyperbolicity_per_batch(points, num_samples=5000, num_trials=10)
    print(mean, f'time:{time.time()-start_time:.2f}')
