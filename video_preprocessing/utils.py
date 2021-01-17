import numpy as np


def fish_k_means(clipped_array: np.array) -> np.array:
    # Flatten array to allow k means calculation
    flat_array = clipped_array.reshape(clipped_array.shape[0] * clipped_array.shape[1]).reshape(-1, 1)

    # Initialise centroids through values of interest
    centroids = np.array([np.quantile(flat_array, 0.005),
                          np.quantile(flat_array, 0.006),
                          np.quantile(flat_array, 0.007),
                          np.median(flat_array) - 5,
                          np.median(flat_array) - 1,
                          np.median(flat_array)]).reshape(-1, 1)

    # Create a list to store which centroid is assigned to each point
    assigned_centroids = np.zeros(len(flat_array), dtype=np.int32)

    def compute_l2_distance(x, centroid):
        # Compute the difference, following by raising to power 2 and summing
        dist = ((x - centroid) ** 2).sum(axis=x.ndim - 1)

        return dist

    def get_closest_centroid(x, centroids):

        # Loop over each centroid and compute the distance from data point.
        dist = compute_l2_distance(x, centroids)

        # Get the index of the centroid with the smallest distance to the data point
        closest_centroid_index = np.argmin(dist, axis=1)

        return closest_centroid_index

    def compute_sse(data, centroids, assigned_centroids):
        # Initialise SSE
        sse = 0

        # Compute SSE
        sse = compute_l2_distance(data, centroids[assigned_centroids]).sum() / len(data)

        return sse

    sse_list = list()
    for n in range(100):
        # Get closest centroids to each data point
        assigned_centroids = get_closest_centroid(flat_array[:, None, :], centroids[None, :, :])

        # Compute new centroids
        for c in range(centroids.shape[1]):
            # Get data points belonging to each cluster
            cluster_members = flat_array[assigned_centroids == c]

            # Compute the mean of the clusters
            cluster_members = cluster_members.mean(axis=0)

            # Update the centroids
            centroids[c] = cluster_members

        # Compute SSE
        sse = compute_sse(flat_array.squeeze(), centroids.squeeze(), assigned_centroids)
        sse_list.append(sse)
        if len(sse_list) > 2 and sse == sse_list[-1]:
            break

    clustered_frame = assigned_centroids.reshape(416, 512)

    return clustered_frame != 5
