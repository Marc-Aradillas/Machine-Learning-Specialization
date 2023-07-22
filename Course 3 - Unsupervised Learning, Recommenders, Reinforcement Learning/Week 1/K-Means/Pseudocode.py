"""
In pseudocode, the K-means algorithm is as follows:

  # Initialize centroids
  # K is the number of clusters
  centroids = kMeans_init_centroids(X, K)

  for iter in range(iterations):
      # Cluster assignment step: 
      # Assign each data point to the closest centroid. 
      # idx[i] corresponds to the index of the centroid 
      # assigned to example i
      idx = find_closest_centroids(X, centroids)

      # Move centroid step: 
      # Compute means based on centroid assignments
      centroids = compute_centroids(X, idx, K)
"""
