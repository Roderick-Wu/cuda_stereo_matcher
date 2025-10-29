import numpy as np
from scipy.ndimage.filters import *

def census_transform(img, win_size=5):
    """ 
    STAGE 1: FEATURE ENCODING
    ==========================
    The Census Transform creates a robust binary descriptor for each pixel
    by encoding the local intensity structure. For each pixel:
    
    1. Compare center pixel with all neighbors in a window (e.g., 5x5)
    2. Create a bit string: bit=1 if neighbor < center, bit=0 otherwise
    3. Result: A binary number that captures local texture pattern
    
    Similar to the BRISK features. 
    
    EXAMPLE: For a 3x3 window
    ```
    Intensities:        Center = 100
    [90  95  105]       [0   0   1]
    [85  100 110]  -->  [0   0   1]  -->  Bit string: 11010110 (skip middle one)
    [80  95  115]       [0   0   1]
    ```
    https://en.wikipedia.org/wiki/Census_transform
    """
    m, n = img.shape
    half_win = win_size // 2
    census = np.zeros((m, n), dtype=np.uint64)
    
    # Pad the image to handle borders (replicate edge pixels)
    img_padded = np.pad(img, half_win, mode='edge')
    
    #import pdb; pdb.set_trace()
    # Process all pixels at once for each neighbor position
    # Instead of looping over pixels, we slice the entire image
    # This is MUCH faster  
    for i in range(win_size):
        for j in range(win_size):
            if i == half_win and j == half_win:
                continue  # Skip center pixel itself
            
            # Extract ALL neighbor pixels at this offset (i, j)
            # This gets the i,j neighbor for every pixel simultaneously
            neighbor = img_padded[i:m + i, j:n + j]
            
            # Compare ALL pixels at once: creates boolean array
            # Then shift existing bits left and bitwise OR in the new comparison bit
            # We can just OR with the original image since its pixels are the center pixels
            # This builds the binary descriptors one bit at a time, for all pixels simultaneously
            # These bit operations seem to be much faster than if we stored as a string for example
            census = (census << 1) | (neighbor < img).astype(np.uint32)

    # Remarks:
    # If we consider a 3*3 window, we get an 8-bit descriptor (1 byte)
    # A 5*5 window gives a 24-bit descriptor (fits in uint32)
    # This part of the algorithm is not the bottleneck but is it optimal to have
    # descriptors that have number of bits divisible by 8 to fit into neat number of bytes?
    # Not going to look into the C code
    
    return census

def hamming_distance_tensors(a, b):
    """
    Compute Hamming distance between arrays of integers
    """
    # XOR to find differing bits
    # Remember a and b are arrays of integers so we can't just bin(a ^ b).count('1')
    xor = np.bitwise_xor(a, b)
    
    count = np.zeros_like(xor, dtype=np.int32)
    # Similar to before, count bits and shift, but for everything in the array simultaneously
    while np.any(xor):
        count += (xor & 1).astype(np.int32)
        xor >>= 1
    
    return count

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---
    
    # ========================================================================
    # STEREO CORRESPONDENCE ALGORITHM - OVERVIEW
    # 
    # 1. Census Transform: Convert pixels to robust binary descriptors
    # 2. Cost Computation: Compare descriptors at different disparities,
    #    use hamming distance for this
    # 3. Cost Aggregation: Average costs over local windows (reduce noise), 
    #    apply a guided filter
    # 4. Disparity Selection: Fit a parabola to get sub-pixel accuracy, not just
    #    winner-take-all
    # ========================================================================

    Id = np.zeros_like(Il, dtype=np.float64)
    
    # First, census transform both images
    # Creates nice binary descriptors for fast matching
    census_win = 5  # Window size for census transform, 3 seems to be good. 
    # I think originally 5 was better but with billateral filtering and better cost
    # aggregation, 3 seems to be the best now
    census_left = census_transform(Il, census_win)
    census_right = census_transform(Ir, census_win)
    
    
    
    
    # Get bounding box dimensions (only compute within this region)
    y_min, y_max = bbox[1, 0], bbox[1, 1]
    x_min, x_max = bbox[0, 0], bbox[0, 1]
    
    # Cost volume dimensions: [height x width x disparity]
    # cost_volume[y, x, d] = matching cost (hamming distance) for pixel (y,x) at disparity d
    cost_volume = np.zeros((Id.shape[0], Id.shape[1], maxd + 1), dtype=np.float32)
    
    # For each disparity level, compute how well pixels match
    # We again take a vectorized approach to speed this up
    # We iterate over disparities, and compute the cost (hamming distance for us) for all valid pixels at once
    # We then add this 2d slice to our cost volume
    for d in range(0, maxd + 1):
        # For disparity d:
        # - Left image pixel at (y, x) should match
        # - Right image pixel at (y, x-d)  [shifted LEFT by d pixels]
        
        #for y_idx, y in enumerate(range(y_min, y_max + 1)):
        # Only consider x positions where (x-d) is valid in right image
        x_start = max(x_min, d)  # Need x >= d so that x-d >= 0
        x_end = x_max
            
        if x_start > x_end:
            continue  # No valid pixels for this disparity
            
        # Get census descriptors for all pixels in this row
        x_left = np.arange(x_start, x_end + 1)
        x_right = x_left - d  # Corresponding positions in right image
            
        left_census = census_left[y_min:y_max + 1, x_left]
        right_census = census_right[y_min:y_max + 1, x_right]

        #import pdb; pdb.set_trace()
            
        # Compute matching cost = Hamming distance between descriptors
        # Low cost = descriptors are similar = good match
        hamming_dists = hamming_distance_tensors(left_census, right_census)

        # Store this 2D slice in 3D cost volume
        cost_volume[y_min:y_max+1, x_left, d] = hamming_dists
    
    # for y in range(y_min, y_max + 1):
        # for x in range(x_min, x_max + 1):
            # left_descriptor = [census_left[y, x] for d in range(0, maxd + 1)]
            # right_descriptors = [census_right[y, x - d] for d in range(0, maxd + 1)]

            # hamming_dist = hamming_distance_tensors(left_descriptor, right_descriptors)

            # cost_volume[y, x, :] = hamming_dist


    #return np.argmin(cost_volume, axis=2)
    # If we return here its not even that bad but it is noisy as hell
    # Need to apply some cost aggregation to smooth things out





    # Cost Aggregation
    # Single pixel comparisons are noisy. Aggregate costs over a window

    # Originally, we use a simply box filter
    # This is also easy to optimize with integral images
    # It's not bad at all

    # But we can do better with a Gaussian filter
    # It's also linearly separable (from lecture!) 
    # so we can optimize with 1D convolutions

    # Okay the Gaussian is actually slightly worse for some reason
    # Let's use a bilateral filter instead
    # Not lienarly separable though, so more expensive
    # This is the biggest bottleneck now

    # Coming back to this after a week or so ...
    # Tuning the bilateral filter is barely making any progress
    # I found this: https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf
    # Guided filter - fast edge-preserving filter
    # Basically assume that the output is a linear mapping of the guide image
    # within a local window, then solve for the linear coefficients
 
    agg_win_size = 21 # Window size for cost aggregation 
    half_agg = agg_win_size // 2
    
    # Bilateral filter parameters
    #sigma_spatial = 10
    #sigma_intensity = 20

    x = np.arange(-half_agg, half_agg + 1)
    y = np.arange(-half_agg, half_agg + 1)
    #X, Y = np.meshgrid(x, y)
    #spatial_weights = np.exp(-(X**2 + Y**2) / (2 * sigma_spatial**2))
    
    aggregated_costs = np.zeros_like(cost_volume) 
        
    def box_filter(img, r):
        """Uses integral images to do this very fast"""
        h, w = img.shape
        img_padded = np.pad(img, ((r, r), (r, r)), mode='edge')
            
        # Build integral image
        integral = np.zeros((h + 2*r + 1, w + 2*r + 1), dtype=np.float64)
        integral[1:, 1:] = np.cumsum(np.cumsum(img_padded, axis=0), axis=1)
            
        # Extract box sums using integral image
        win_size = 2*r + 1
        br = integral[win_size:h + win_size, win_size:w + win_size]
        tr = integral[0:h, win_size:w + win_size]
        bl = integral[win_size:h + win_size, 0:w]
        tl = integral[0:h, 0:w]
            
        return br - tr - bl + tl
        
    def guided_filter(I, p, r, eps):
        """
        Guided filter - edge-preserving smoothing
            
        I: guidance image (intensity image)
        p: input image to filter (cost slice)
        r: window radius
        eps: regularization parameter (controls edge preservation)
            
        Returns: filtered image

        Note: This function is vectorized over the entire image
        """
        #import pdb; pdb.set_trace()
        # Compute local statistics using box filters (very fast!)
        mean_I = box_filter(I, r) / ((2*r + 1) ** 2)
        mean_p = box_filter(p, r) / ((2*r + 1) ** 2)
        mean_Ip = box_filter(I * p, r) / ((2*r + 1) ** 2)
        cov_Ip = mean_Ip - mean_I * mean_p  # Covariance
            
        mean_II = box_filter(I * I, r) / ((2*r + 1) ** 2)
        var_I = mean_II - mean_I * mean_I  # Variance
            
        # Linear coefficients
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
 
        # Average coefficients over window
        mean_a = box_filter(a, r) / ((2*r + 1) ** 2)
        mean_b = box_filter(b, r) / ((2*r + 1) ** 2)
            
        # Apply linear transform
        return mean_a * I + mean_b
    

    # Guided filter parameters
    eps = 1e-5  # Regularization 
    # Tuning this doesn't actually seem to do very much

    # Apply guided filter to each disparity slice
    for d in range(0, maxd + 1):
        cost_slice = cost_volume[:, :, d]
            
        # Use intensity image as guide, filter the cost slice
        filtered = guided_filter(Il, cost_slice, half_agg, eps)
            
        aggregated_costs[:, :, d] = filtered
        
    Id = np.argmin(aggregated_costs, axis=2)

    #return Id
    # Here, we can use winner-take-all and 
    # just return the disparity with minimum cost
    # And it does pretty well actually
    
    # But we can do a little bit better, use a spline fit to get sub-pixel accuracy
 
    h, w, num_disp = aggregated_costs.shape
    
    for y in range(h):
        for x in range(w):
            d = int(Id[y, x])

            # Only refine if we're not at boundaries
            if d > 0 and d < maxd:
                # Get costs at d-1, d, d+1
                c_prev = aggregated_costs[y, x, d - 1]
                c_curr = aggregated_costs[y, x, d]
                c_next = aggregated_costs[y, x, d + 1]
                
                # Compute parabola minimum (sub-pixel correction)
                denom = 2 * (c_next - 2*c_curr + c_prev)
                if abs(denom) > 1e-10:  # Avoid division by zero
                    delta = (c_prev - c_next) / denom
                    # Clamp delta to reasonable range [-0.5, 0.5]
                    delta = np.clip(delta, -0.5, 0.5)
                    Id[y, x] += delta
    
    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id