import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    #print(centers[0][1])
    assignments = np.zeros(N, dtype=np.int32)
    oldassignments=np.empty(N,dtype=np.int32)

    for n in range(num_iters):
        ### YOUR CODE HERE

        distances=np.empty(len(centers),dtype=np.float64)
        
        #belongstocenter=np.empty(len(centers),dtype=np.int32)
        for i in range(0,N):
         for j in range(0,len(centers)):
             distanceX=(features[i][0]-centers[j][0])**2
             distanceY=(features[i][1]-centers[j][1])**2
             distanceEuclidean=math.sqrt(distanceX+distanceY)
             distances[j]=distanceEuclidean
             #belongstocenter[j]=centers[j][0]
         minIndex=0
         for k in range(0,len(distances)):
            if distances[k] < distances[minIndex]:
               minIndex = k
                 
         #minIndex=np.argmin(distances, axis=None)
         #print(minIndex)
         #chosenCentreX=belongstocenter[index]
         #print(chosenCentreX)
         assignments[i]=minIndex

        if(oldassignments.all()==assignments.all()):
          return assignments 
        oldassignments=assignments
        
        #newCenters=np.empty(shape=(4,2),dtype=np.int32)
        newCenters=centers
        #print(len(oldCentres))
        
        
        for j in range(len(centers)):
         sumX=0
         sumY=0
         counter=0
         for i in range(len(assignments)):
            
                if(assignments[i]==j):
                    sumX+=features[i][0]
                    sumY+=features[i][1]
                    counter+=1
         newCentreX= sumX/counter    
         newCentreY= sumY/counter   
         newCenters[j][0]=newCentreX
         newCenters[j][1]=newCentreY
        print(newCenters) 
        

        centers=newCenters
    
    
        
            
         ### END YOUR CODE



def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.int32)
    oldassignments=np.empty(N,dtype=np.int32)
    centerindex=np.empty(len(centers),dtype=np.int32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        #distances=np.empty(len(centers),dtype=np.float64)
        
        CDist = cdist(features,  centers, 'euclidean')
        #print(CDist)
        #print(CDist)
        #X, Y= CDist.shape
        #print(X)
        #print(Y)
        
        #a = np.tile(features,(k, 1))
        #b = np.repeat(centers,N,axis=0)
        #assignments = np.argmin(np.sum((a-b)**2,axis=1).reshape(k,N),axis=0) 
         #center2index=np.where(features == centers[1][0])
        #print(center2index)
         #center3index=np.where(features == centers[2][0])
         #center4index=np.where(features == centers[3][0])
        #for i in range(N):
            ##for j in range(len(centers)):
             #print(np.where(features == centers[j][0])[0])
             #centerindex[j]=((np.where(features == centers[j][0])[0])[0])
             
             ##distances[j]=CDist[i][j]
             # print(distances)
             #a = np.tile(features,(k, 1))
             #b = np.repeat(centers,N,axis=0)
        assignments=np.argmin(CDist,axis=1)
            #print(assignments)
         #distances1[i]=CDist[idxs[0]][i]
         #distances2[i]=CDist[idxs[1]][i]
         #distances3[i]=CDist[idxs[2]][i]
         #distances4[i]=CDist[idxs[3]][i]
         
        #closest1=np.argmin(distances1,axis=0)
        #print(centers[0][1])
         
         #print(center1index[0])
        
        #axis0, axis1 = np.where((features==centers[0][0]).view('i1').sum(-1) > 1)
        #print(axis0)
         
            
         #print(CDist[i][center1index[0]])
         #distances[1]=CDist[i][(centerindex[j])[0]]
         #distances[2]=CDist[i][(center3index[0])[0]]
         #distances[3]=CDist[i][(center4index[0])[0]]
            
        if(oldassignments.all()==assignments.all()):
          return assignments 
        oldassignments=assignments
        
        #newCenters=np.empty(shape=(4,2),dtype=np.int32)
        #newCenters=centers
        for j in range(k):
         #sumX=0
         #sumY=0
         #counter=0
         #for i in range(len(assignments)):
            
                #if(assignments[i]==j):
                   # sumX+=features[i][0]
                   # sumY+=features[i][1]
                   # counter+=1
         #newCentreX= sumX/counter    
         #newCentreY= sumY/counter   
         #newCenters[j][0]=newCentreX
         #newCenters[j][1]=newCentreY
         centers[j] =np.mean(features[assignments==j],axis=0) 
        #print(newCenters) 
        

        #centers=newCenters
        pass
        ### END YOUR CODE

    #return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N
    

    while n_clusters > k:
        ### YOUR CODE HERE
        pass
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))
    HtimesC=np.empty(H*C,dtype=np.float32)

    ### YOUR CODE HERE
    #for i in range(H*W):
             #features[i][0]=img[int(i/W)][int(i/H)][0]
             #features[i][1]=img[int(i/W)][int(i/H)][1]
             #features[i][2]=img[int(i/W)][int(i/H)][2]
             #pass
    ### END YOUR CODE
    features = img.reshape((H*W, C))
    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    Array1 = np.mgrid[0 : H, 0 : W]
    Array = np.dstack(Array1).reshape((H * W , 2))
    colorpos = color.reshape((H * W, C))
    ArrayConc = np.concatenate((colorpos, Array), axis = 1)
    features = (ArrayConc - ArrayConc.mean(axis = 0)) / ArrayConc.std(axis = 0)
    ### END YOUR CODE

    return features

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    M, N = mask_gt.shape
    acc = 0
    for i in range (M):
        for j in range (N):
            if (mask_gt[i][j] == mask[i][j]):
                acc += 1
    accuracy = acc/M
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
