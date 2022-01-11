## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/yarongyayun/Clustering/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### What is Clustering?
Clustering is finding similarities between data according to the characteristics found in the data and grouping similar data objects into clusters.

In machine learning, we often group examples as a first step to understand a subject (data set) in a machine learning system. Grouping unlabeled examples is called **clustering**. 

It is an optimization problem.
* Variability (c) [Sum(distance(mean(c),e)square)]
* Dissimilarity (C) [Sum(variability)]
* Apply Constrian (e.g. minium distance)
* Purity of clusters
* Scaling

### What are the Users of Clustering?
Used for:
* Understanding: what is in a dataset
* Summarization: reduce the size of large datasets
* Data smoothing and discretization
* Features for other learning tasks

Typical applications:
* As a stand-alone tool to get insight into data distribution
* As a preprocessing step for other algorithms


Clustering has a myriad of uses in a variety of industries. Some common applications for clustering include the following:

* market segmentation: help marketers discover distinct groups in their customer bases, and then use this knowledge to develop targeted marketing programs
* social network analysis
* search result grouping
* medical imaging
* image segmentation
* anomaly detection

After clustering, each cluster is assigned a number called a cluster ID. Now, you can condense the entire feature set for an example into its cluster ID. 

Distance Types:
* Jaccard distance: Document = set of words
* Euclidean distance: Document = point in space of words
* Cosine distance: vector in space of words (vector from ofigin to x1, x2...)

### Clustering Algorithms

References:
* [A comprehensive Survey of Clustering Algorithms]( https://link.springer.com/article/10.1007/s40745-015-0040-1)
* [Clustering Overview in ScikitLearn](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods)
* [MIT Open Course: clustering](https://www.youtube.com/watch?v=esmzYhuFnds)

Four main types of clustering:

1. Centroid-based Clustering
2. Density-based Clustering
3. Distribution-based Clustering
4. Hierarchical Clustering -- (dendogram, linkage metrics: single linkage, complete linkage, average linkage)

## 1. Centroid-based Clustering
(Clustering algorithm based on partition). Centroid-based algorithms are efficient but sensitive to initial conditions and outliers. k-means is the most widely-used centroid-based clustering algorithm.

### K-means Clustering
* [Advantages of k-means](https://developers.google.com/machine-learning/clustering/algorithm/advantages-disadvantages)
  * Relatively simple to implement.
  * Scales to large data sets.
  * Guarantees convergence.
  * Can warm-start the positions of centroids.
  * Easily adapts to new examples.
  * Generalizes to clusters of different shapes and sizes, such as elliptical clusters.
  <br/><br/>
* Disadvantages:
  * Choosing K mannually
  * Being dependent on initial values
  * Clustering data of varying sizes and density
  * Clustering outliers
  * Scaling with number of dimensions (solutions: PCA, or using '[spectral clustering](https://github.com/petermartigny/Advanced-Machine-Learning/blob/master/DataLab2/Luxburg07_tutorial_4488%5B0%5D.pdf)'-a pre-clustering step
  )
  <br/><br/>
* Limitations:
  * K-means may not work well with data having different scales in different dimensions<br/><br/>

* K-means++
  * K-means++ can dramatically improve the convergence speed by choosing initial centroid with a highe probability of being close to the final ones.
  * k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm. It was proposed in 2007 by David Arthur and Sergei Vassilvitskii, as an approximation algorithm for the NP-hard k-means problem—a way of avoiding the sometimes poor clusterings found by the standard k-means algorithm. 
  * The intuition behind this approach is that spreading out the k initial cluster centers is a good thing: the first cluster center is chosen uniformly at random from the data points that are being clustered, after which each subsequent cluster center is chosen from the remaining data points with probability proportional to its squared distance from the point's closest existing cluster center.


### [K-medoids Clustering](https://www.youtube.com/watch?v=GApaAnGx3Fw)
* Why from K-means to K-medoids?
  * The K-means algorithm is sensitive to outliers
  * Kimedoids: instead of taking the mean value of the object in a cluster as a reference point, medoids can be used, which is **the most centrally located object** in a cluster
  <br/><br/>
* Definitation
  * The k-medoids problem is a clustering problem similar to k-means. 
  * Both the k-means and k-medoids algorithms are partitional (breaking the dataset up into groups) and attempt to minimize the distance between points labeled to be in a cluster and a point designated as the center of that cluster. In contrast to the k-means algorithm, k-medoids chooses actual data points as centers (medoids or exemplars), and thereby allows for greater interpretability of the cluster centers than in k-means, where the center of a cluster is not necessarily one of the input data points (it is the average between the points in the cluster). 
  * Furthermore, k-medoids can be used with arbitrary dissimilarity measures, whereas k-means generally requires Euclidean distance for efficient solutions. Because k-medoids minimizes a sum of pairwise dissimilarities instead of a sum of squared Euclidean distances, it is more robust to noise and outliers than k-means.
  <br/><br/>
* [PAM (Partition Around Medoids) Clustering](https://www.cs.umb.edu/cs738/pam1.pdf):
  * The key issue with PAM is its high run time cost.
<br/><br/>

### CLARA
* Why:
  * K-means and PAM are greate but...it consumes a lot of memory and resources
  * K-means and PAM is not suitable for large datasets (nrow>10x10x10)
* CLARA does not take the whole dataset into consideration, instead it uses a **random sample** of the dataset, from which the best medoids are taken
* The effectiveness of CLARA depends on the sample size. CLARA cannot find a good clustering if any of the best sampled medoids is far from the best k-medoids.

### [CLARANS](https://medium.com/analytics-vidhya/partitional-clustering-using-clarans-method-with-python-example-545dd84e58b4)
* pyclustering.cluster: https://pypi.org/project/pyclustering/
* Why? - Limitations of CLARA
  * The best k medoids may not be selected during the sampling process, in this case, CLARA will never find the best clustering
  * If the sampling is biased or partial, we cannot find good quality of clusters
  * Trade-off efficiency
  <br/><br/>
* Definition 
  * CLARANS known as CLustering Large Applications based upon RANdomized Search.
  * CLARANDS like PAM starts with a randomly selected set of K medoids (or few paris) instead of examining all pairs, for swapping at the current state.
  * If check at most the "maxneighbor" number of pairs for swapping and, if a pair with negative cost is found, it update the medoids as a local optimum and restarts with a new randomly selected medoid, set to search for another local optimum.
  * CLARANS stops after the "numlocal" number of local optimal medoid sets are determined, and return the best among these.
  * CLARANS finds local minima among neighbours and a parameter determines how many times it performs this local search.
  <br/><br/>
* CLARA vs CLARANS
  * like CLARA, CLARANS does not check every neighbors of a node.
  * Unlike CLARA, CLARANS does not restrict its search to a particular graph. It search the original graph Gn,k
  * CLARANS draws a sample of nodes at the beginning of a search, CLARANS draws a sample of neighbors in each step of a search.
  * CLARANS gives higher quality clustering than CLARA.
  * CLARANS requires a very small number of searches than [CLARANS](https://www.youtube.com/watch?v=q3plVFIgjGQ).
  <br/><br/>
* Advantages of CLARANS algorithm
  * It is more efficient and scalable than both PAM and CLARA.
  * It does not restrict the search to any particular subset of objects.
  * It improves the time complexity based on randomized search.
  * It used for large database.
  * It gives higher quality clustering.
  * It requires very small number of searches.
  <br/><br/>
* **

## 2. Hierarchical Clustering
**Hierarchical Clustering** creates a tree of clusters. 
[**Key Operation:**](https://www.youtube.com/watch?v=rg2cjfMsCk4) repeatedly combine two nearest clusters. 

### [BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)](https://machinelearningmastery.com/clustering-algorithms-with-python/)
* BRICH (balanced iterative reducing and clustering using hierarchies) is a scalable clustering method designed for handling very large datasets. It uses statistical methods to compress the data such that an existing clustering algorithm could use the summaries rather than the original data to reduce the necessary number of computations (only one scan of the data is necessary). 
* BIRCH realizes the clustering result by constructing the feature tree of clustering, CF (Clustering Feature) tree, of which one notde stands for a subcluster. CF tree will dynamically grow when a new data point comes. CF is a set of summary statistics that can be used to build the CF-tree and for final clusterying (but can't do signle link). https://www.youtube.com/watch?v=PMBkL9lkoq4
* The CF Subclusters hold the necessary information for clustering which prevents the need to hold the entire input data in memory. This information includes: VF = (N,LS,SS) 
  * Number of samples in a subcluster. (N)
  * Linear Sum - An n-dimensional vector holding the sum of all samples. (LS) >> a vector
  * Squared Sum - Sum of the squared L2 norm of all samples. (SS)
  * Centroids - To avoid recalculation linear sum / n_samples.
  * Squared norm of the centroids.
  <br/><br/>

* Scales linearly: finds a good clustering with a single scan and improves the quality with a few additional scans
* In most cases, BIRCH only requires a single scan of the database.
* Its inventors claim BIRCH to be the "first clustering algorithm proposed in the database area to handle 'noise' (data points that are not part of the underlying pattern) effectively",beating DBSCAN by two months. The BIRCH algorithm received the SIGMOD 10 year test of time award in 2006.
<br/><br/>

* The BRICH clustering algorithm consists of two main phases of operation. 
  * The first phase (building the CF tree) loads the data into memory by building a cluster feature (CF) tree. Leaf nodes hold many small and "tight" clusters. (Daa reduction and rough clustering)
  * The second phase (global clustering) applies an existing algorithm on the leaves of the CF tree. Use other clustering algorithm to cluster the small & tight clusters, merge dense clusters and/or remove outliers (fine clustering)
   <br/><br/>
* An advantage of BIRCH is its ability to incrementally and dynamically cluster incoming, multi-dimensional metric data points in an attempt to produce the best quality clustering for a given set of resources (memory and time constraints). 
 <br/><br/>

* The BIRCH algorithm has Three parameters: 
  * T: cluster diameter threshold for leaf entries (how tight each node)
  * B: branching factor, length of an internal node. The branching factor limits the number of subclusters in a node and the threshold limits the distance between the entering sample and the existing subclusters.
  * L: length of an leaf node
  * B, L depends on th ememory page size

* Pros
  * Efficient for clustering tasks that can't fit into memory
  * Different clustering algorithms can be used on the 'tight' clusters in the leaf nodes
  * Can turn on outlier remover: e.g., subcluster with data points < 1/4 of the average size of all leaf subsclusters

* Cons
  * Only work with numerical data
  * Insertion order of observations affects the CF-tree
    * Duplicated observsations can be put into different clusters
    * Randomize observations to reduce this effect
    * Can scan db again and assign individual data points to the closest centroid
  * Due to the fixed size of leaf nodes, final clusters may not be natureal
  * Cluster tends to be spherical given the use of radius and diameter measures


### [CURE (Clustering Using REpresentatives)](https://www.youtube.com/watch?v=JrOJspZ1CUw):

CURE, suitable for large-scale clustering, takes random sampling technique to cluster sample separately and integrates the results finally.
<br/><br/>

Assumptions:
* Assumes a Euclidean distance
* Allows clusters to assume any shape
* Uses a collection of representative points to represents clusters

Starting CURE:
* Pick a random sample of points that fit in main memory
* Cluster sample points hierarchically to create the initial clusters
* Pick representative points:
  * for each cluter, pick k (e.g., 4) representative points, as dispersed as possible
  * move each representative point a fixed fraction (e.g., 20%) toward the centroid of the cluster

Finishing CURE:
* Now, rescan the whole dataset and visit each point p in the data set
* Place it in the "closest cluster"
  * normal definition of "closest": that cluster with the closest (to p) among all the representative points of all the clusters

Summary:
* Clustering: Given a set of points, with a notion of distance between points, group the points into some number of clusters.
* Algorithms:
  * agglometative **hierarchical clustering** - Centroid and Clusteroid
  
### [ROCK (RObust Clustering using linKs)](https://analyticsindiamag.com/hands-on-guide-to-rock-clustering-algorithm/)

* ROCK is an improvement of CURE for dealing with data of enumeration type, which takes the effect on the similarity from the data around the cluster into consideration.
* ROCK employs links when merging them into a cluster. 
* Not distance-based
* [Sampling-based clustering](https://medium.com/geekculture/the-rock-algorithm-in-machine-learning-5fa152f34de7)
* Good for categorical data. ROCK is a robust clustering algorithm for categorical attributes
* from pyclustering.cluster.rock import rock
* [Sample notebook](https://colab.research.google.com/drive/1A-TWGQUYP3oBQ9wIQ5xZgT_15LTXEXAE?usp=sharing)
* **evaluate the performance of the clustering approach by using Extrinsic measures such as Adjusted Rand index, Fowlkes-Mallows scores, Mutual information based scores, Homogeneity, Completeness and V-measure etc., or Intrinsic measures such as Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index etc., to decide on the best approach. 

### [Chameleon](https://www-users.cse.umn.edu/~hanxx023/dmclass/chameleon.pdf)
* CHAMELEON that measures the similarity of two clusters based on a dynamic model. In the clustering process, two clusters are merged only if the inter-connectivity and closeness (proximity) between two clusters are comparable to the internal inter-connectivity of the clusters and closeness of items within the clusters. 
* CHAMELEON is applicable to all types of data as long as a similarity matrix can be constructed.
* The key feature of CHAMELEON’s agglomerative hierarchical clustering algorithm is that it determines the pair of most similar sub-clusters by taking into account both the inter-connectivity as well as the closeness of the clusters; and thus it overcomes the limitations discussed in Section 3 that result from using only one of them (relative inter-connectivity & relative closeness)
* **

## 3. Fuzzy Theory Based Clustering
The advantages fuzzy clustering rather than hard/crisp is having lesser tendency to get trapped in local minimum. Fuzzy clustering algorithm widely used is based on objective function.
### [FCM (Fuzzy C-Means Clustering)](https://medium.com/geekculture/fuzzy-c-means-clustering-fcm-algorithm-in-machine-learning-c2e51e586fff)
* FCM is characterized by centre of cluster
* Soft clustering.
*  “fuzzy” here means “not sure”, which indicates that it’s a soft clustering method. “C-means” means c cluster centers, which only replaces the “K” in “K-means” with a “C” to make it look different.
* One data point can potentially belong to multiple clusters.
* The outcome is probability based.
It has better accuracy as in most scenarios the data points overlap
* This is better suitable for ambiguous clustering problems
<br/><br/>

* Steps:
  * Randomly select 'c cluster centers
  * Calculate the cluster membership probability (for ith data point to the jth cluster
  Compute the fuzzy centers again using Vj
  * Repeat the pevious 2 steps until minimal J value is achieved
  <br/><br/>
* Two Parameters:
  * μ_ij, membership value, is the probability that the jth data point belongs to the ith cluster, and it is constrained to that the sum of μ_ij over C cluster centers is 1 for every data point j. 
  * c_i is the center of the ith cluster (the same dimension with X). And m is the fuzzifier, which controls how fuzzy the cluster boundary should be.
  <br/><br/>
* [Minimize the objective function](https://towardsdatascience.com/fuzzy-c-means-clustering-with-python-f4908c714081)
* Package
  * from fcmeans import FCM
  * https://pypi.org/project/fuzzy-c-means/
   <br/><br/>
* [FCM implementation video](https://www.youtube.com/watch?v=W-3ZYGmLJ-4&list=PLH5lMW7dI2qfryTUxGvtpy9UNTdMgwCYy&index=71)
  * [Python notebook](https://github.com/rsharankumar/Learn_Data_Science_in_100Days/tree/master/Day70-71%20-%20Fuzzy%20C-Means)

### [FCS (Fuzzy c-shells clustering)](https://iopscience.iop.org/article/10.1088/1742-6596/1613/1/012006/pdf)
* FCS is the expansion of fuzzy c-means algorithm (FCM) with additional parameter, radius. FCS algorithm which observes radius and centre of clusters. The FCS algorithm is the development of FCM algorithm which only note the centre of cluster.
* FCS algorithm is the first algorithm which is utilizing the non-linear prototype from FCM algorithm and as reference to develop new algorithm with another varieties of prototypes. 
* FCS algorithm is often used in detects nonlinear space on image segmentations then can be used to clusters dataset with better accuracy because of the additional parameter.
<br/><br/>
* Comparison to K-means clustering:

  * K-means clustering also attempts to minimize the objective function shown above. This method differs from the k-means objective function by the addition of the membership values {\displaystyle w_{ij}}{\displaystyle w_{ij}} and the fuzzifier, {\displaystyle m\in R}{\displaystyle m\in R} , with {\displaystyle m\geq 1}{\displaystyle m\geq 1}. The fuzzifier {\displaystyle m}m determines the level of cluster fuzziness. A large {\displaystyle m}m results in smaller membership values, {\displaystyle w_{ij}}{\displaystyle w_{ij}}, and hence, fuzzier clusters. In the limit {\displaystyle m=1}{\displaystyle m=1}, the memberships, {\displaystyle w_{ij}}{\displaystyle w_{ij}} , converge to 0 or 1, which implies a crisp partitioning. In the absence of experimentation or domain knowledge, {\displaystyle m}m is commonly set to 2. The algorithm minimizes intra-cluster variance as well, but has the same problems as 'k'-means; the minimum is a local minimum, and the results depend on the initial choice of weights
 <br/><br/>
 
 * Advantages: more relasitc to give the probability of belonging relatively high accuracy of clustering
 * Disadvantages: relatively low scalability in general, easily drawn into local optimal, the clustering result sensitive tothe initial parameter values, and the  number of clusters needed to be preset.

### MM

* **

## 4. Distribution Based Clustering
This clustering approach assumes that the data consists of distributions (such as the Gaussian distribution). As the distance from the center of a distribution increases, the probability that a data point belongs to it decreases. The prerequisite for a distribution-based cluster analysis is always that several distributions are present in the data set. Typical algorithms here are DBCLASD and GMM.

###  [DBCLASD (Distribution-Based Clustering of LArge Spatial Databases)](https://ieeexplore.ieee.org/document/655795)
* DBCLASD, contrary to partitioning algorithms such as CLARANS (Clustering Large Applications based on RANdomized Search), discovers clusters of arbitrary shape. 
* Furthermore, DBCLASD does not require any input parameters, in contrast to the clustering algorithm DBSCAN (Density-Based Spatial Clustering of Applications with Noise) requiring two input parameters, which may be difficult to provide for large databases. 
* In terms of efficiency, DBCLASD is between CLARANS and DBSCAN, close to DBSCAN. Thus, the efficiency of DBCLASD on large spatial databases is very attractive when considering its nonparametric nature and its good quality for clusters of arbitrary shape.
### [GMM (Gaussian Mixture Models)](https://towardsdatascience.com/gaussian-mixture-models-d13a5e915c8e)
* Gaussian Mixture Models are probabilistic models and use the soft clustering approach for distributing the points in different clusters. 
* The core idea of GMM is that the data set consists of several Gaussian distributions generated from the original data. And that the data that belong to the same independent Gaussian distribution are considered to belong to the same cluster.
<br/><br/>

* for a dataset with d features, we would have a mixture of k Gaussian distributions (where k is equivalent to the number of clusters), each having a certain mean vector and variance matrix. But wait – how is the mean and variance value for each Gaussian assigned? These values are determined using a technique called **Expectation-Maximization (EM)**. We need to understand this technique before we dive deeper into the working of Gaussian Mixture Models. Broadly, the Expectation-Maximization algorithm has two steps:
  * E-step: In this step, the available data is used to estimate (guess) the values of the missing variables. For each point xi, calculate the probability that it belongs to cluster/distribution c1, c2, … ck.
  * M-step: Based on the estimated values generated in the E-step, the complete data is used to update the parameters
  * Expectation-Maximization is the base of many algorithms, including Gaussian Mixture Models. 
  * k-means only considers the mean to update the centroid while GMM takes into account the mean as well as the variance of the data!
<br/><br/>

* Combination of weighted Gaussians: pi = [0.47, 0.26, 0.27] = 1
  * video: https://www.youtube.com/watch?v=DODphRRL79c
  <br/><br/>
* package: [from sklearn.mixture import GaussianMixture](https://scikit-learn.org/stable/modules/mixture.html)
<br/><br/>
* Advantages:

  * More realistic values for the probability of adherence.
  * Relatively high scalability due to the change in the applied distribution and thus number of clusters.
  * Statistically well provable.
  <br/><br/>
* Disadvantages:
  * Basic assumption is not one hundred percent correct.
  * Numerous parameters that have a strong influence on the clustering result.
  * Relatively high time complexity.
  <br/><br/>
* **

## 5. Density Based Clustering

### [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) (Density-Based Spatial Clustering of Applications with Noise)
Density-Based Spatial Clustering of Applications with Noise (DBSCAN) assumes that clusters represent some dense concentration points. A similarity matrix is used in this case. 
* DBSCAN groups together points that are close to each other based on a distance measurement (usually Euclidean distance) and a minimum number of points.<br/><br/>

* The DBSCAN algorithm basically requires 2 parameters:
  * eps: specifies how close points should be to each other to be considered a part of a cluster. It means that if the distance between two points is lower or equal to this value (eps), these points are considered neighbors.
  * minPoints: the minimum number of points to form a dense region. For example, if we set the minPoints parameter as 5, then we need at least 5 points to form a dense region.<br/><br/>
* Advantages:
  * DBSCAN does not require one to specify the number of clusters in the data a priori, as opposed to k-means.
  * DBSCAN can find arbitrarily-shaped clusters. It can even find a cluster completely surrounded by (but not connected to) a different cluster. Due to the MinPts parameter, the so-called single-link effect (different clusters being connected by a thin line of points) is reduced.
  * DBSCAN has a notion of noise, and is robust to outliers.
  * DBSCAN requires just two parameters and is mostly insensitive to the ordering of the points in the database. (However, points sitting on the edge of two different clusters might swap cluster membership if the ordering of the points is changed, and the cluster assignment is unique only up to isomorphism.)
  * DBSCAN is designed for use with databases that can accelerate region queries, e.g. using an R* tree.
  * The parameters minPts and ε can be set by a domain expert, if the data is well understood.<br/><br/>

* Disadvantages
  * DBSCAN is not entirely deterministic: border points that are reachable from more than one cluster can be part of either cluster, depending on the order the data are processed.
  * The quality of DBSCAN depends on the distance measure used in the function regionQuery(P,ε). The most common distance metric used is Euclidean distance. Especially for high-dimensional data, this metric can be rendered almost useless due to the so-called "Curse of dimensionality", making it difficult to find an appropriate value for ε. This effect, however, is also present in any other algorithm based on Euclidean distance.
  * DBSCAN cannot cluster data sets well with large differences in densities, since the minPts-ε combination cannot then be chosen appropriately for all clusters
  * If the data and scale are not well understood, choosing a meaningful distance threshold ε can be difficult.<br/><br/>
* sample: from sklearn.cluster import DBSCAN: https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html<br/><br/>
* Clustering evaluation matrix: examples
  * Number of clusters:2
  * Homogeneity: 1.0
  * Completeness: 0.959
  * V-measure: 0.917
  * Adjusted Rnd Index: 0.959
  * Adjusted Mutual Information: 0.883
  * Silhouette Coefficient: -0.018
  

### [OPTICS](https://www.geeksforgeeks.org/ml-optics-clustering-explanation/) (Ordering Points To Identify Cluster Structur)
* OPTICS is an improvement of DBSCAN and it overcomes the shortcoming of DBSCAN that being sensitive to two parameters, the radius of the neighborhood and the minimum number of points in a neighborhood
* OPTICS can be seen as a generalization of DBSCAN that replaces the ε parameter(eps) with a maximum value that mostly affects performance. 
* Idea: Higher density points should be processed first -- find high-density cluster first: https://www.youtube.com/watch?v=GCnsmjwV3BE
* OPTICS Clustering stands for Ordering Points To Identify Cluster Structure. It draws inspiration from the DBSCAN clustering algorithm. It adds two more terms to the concepts of DBSCAN clustering. 
  1. Core Distance
  2. Reachability Distance
* OPTICS only works with numberical data
* OPTICS Clustering v/s DBSCAN Clustering:
  1. Memory Cost : The OPTICS clustering technique requires more memory as it maintains a priority queue (Min Heap) to determine the next data point which is closest to the point currently being processed in terms of Reachability Distance. It also requires more computational power because the nearest neighbour queries are more complicated than radius queries in DBSCAN.
  2. Fewer Parameters : The OPTICS clustering technique does not need to maintain the epsilon parameter and is only given in the above pseudo-code to reduce the time taken. This leads to the reduction of the analytical process of parameter tuning.
  3. This technique does not segregate the given data into clusters. It merely produces a Reachability distance plot and it is upon the interpretation of the programmer to cluster the points accordingly.
### [Mean-shift](https://www.youtube.com/watch?v=TMPEujQrY70)
MeanShift is a nonparametric technique of the feature space analysis to determine the maximal probability density location, the so-called mode search algorithm. The MeanShift strategy basically assigns data points to clusters iteratively by shifting points towards the highest density of the data points, which are cluster centroids (centers. The MeanShift inductive algorithm distinguishes objects with nonlinear patterns more precisely, while the inductive DBSCAN tool is less perceptive to nonlinear objects.
* The idea is to find the modes of a distribution, or a probability density
* The mean shift algorithm seeks modes or local maxima of density in the feature space
* Cluster: all data points in the attraction basin of a mode
  * attraction basin: the region for which all trajectories lead to the same mode
* Pros: 
  * automatically finds basins of attraction
  * one parameter choice (window size)
  * Does not assume (image) shape on clusters
  * Generic technique
  * Find multiple modes
* Cons: 
  * Slection of window size
  * does not scale well with dimension of feature space
* **

## 6. Graph Theory Based Clustering

### [CLICK](http://www.cs.tau.ac.il/~rshamir/abdbm/pres/17/Click.pdf) (CLuster Identification via Connectivity Kernels)
* Identify highly homogeneous sets of elements - connectivity kernels.
* Add elements to kernels via similarity to average kernel fingerprints.
* Uses tools from graph theory and probabilistic considerations for similarity evaluation and kernel identification.
* Efficient implementation <br/><br/>

*  The goal is to partition the elements into subsets, which are called clusters, so that two criteria are satisfied: 
   * Homogeneity - elements inside a cluster are highly similar to each other; and
   * separation - elements from different clusters have low similarity to each other.
### [MST(Minimum Spanning Trees)](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0248-7)
* The minimum spanning tree clustering algorithm is known to be capable of detecting clusters with irregular boundaries
* the MST-based methods are not optimised for clustering but aimed at manifold learning
* A minimum spanning tree (MST) or minimum weight spanning tree is a subset of the edges of a connected, edge-weighted undirected graph that connects all the vertices together, without any cycles and with the minimum possible total edge weight.That is, it is a spanning tree whose sum of edge weights is as small as possible. More generally, any edge-weighted undirected graph (not necessarily connected) has a minimum spanning forest, which is a union of the minimum spanning trees for its connected components.

### [Spectral Clustering](https://www.youtube.com/watch?v=VIu-ORmRspA&list=RDVIu-ORmRspA&start_radio=1&rv=VIu-ORmRspA&t=220)
Spectral clustering is a technique with roots in graph theory, where the approach is used to identify communities of nodes in a graph based on the edges connecting them. The method is flexible and allows us to cluster non graph data as well.
* Three basic sgtages:
  1. pre-processing - construct a matrix representation of the graph
  2. decomposition - compute eigenvalues and eigenvectors of the matrix; map each point to a lower-dimensional representation based on one or more eigenvectors
  3. group - assign points to two or more clusters, based on the new representation<br/><br/>

* How to select K?
  * Eigengap - the difference between two consecutive eigenvalues
  * Most stable clustering is generally given by the value k that maximizes eigengap<br/><br/>

* Spectral clustering is one of the most popular methods to cluster high-dimensional data, in which the similarity matrix plays an important role.

* [from sklearn.cluster import SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)

### MMC (Mahalanobis Metric for Clustering)
* **

## 7. Grid Based Clustering
Grid-Based Clustering method uses a multi-resolution grid data structure. https://www.datamining365.com/2020/04/grid-based-clustering.html
### [STING](https://www.youtube.com/watch?v=z0Hb_6s2e8c) (a STatistical INformation Grid approach)
* In this method, the spatial area is divided into rectangular cells at different levels of resolution, and these cells form a tree structure
* A cell at high level contains a number of smaller cells of the next lower level
* Statistical information of each cell is calculated and stored beforehand and is used to answer queries
* Parameters of higher level cells can be easily calculated from that of lower level cell, including:
  * count, mean, std, minm max
  * type of distribution - normal, uniform, etc.
* Query Processing in STING:
  * Start at the root and proceed to the next lower level, using the STING index 
  * Calculate the likelihood that a cell is relevant to the query at some confidence level using the statistical information of the cell
  * Only children of likely relevant cells are recursively explored
  * Repeat this process until the bottom layer is reached
* Advantages
  * Ouery-independent, easy to parallelize, incremental update
  * Efficiency: Complexity is O(k)
    * K: # of grid cells at the lowest level, and K << n (I.E., # of data points
* Disadvantages
  * Its probabilistic nature may imply a loss of accuracy in query processing

### [CLIQUE](https://www.coursera.org/lecture/cluster-analysis/5-6-clique-grid-based-subspace-clustering-AAHTA) (Clustering In QUEst)
It is based on automatically identifying the subspaces of high dimensional data space that allow better clustering than original space.
* CLIQUE is a density-based and grid-based subspace clustering algorithm
  * Grid-based: It discretizes the data space through a grid and estimates the density by counting the number of points in a grid cell
  * Density-based: A cluster is a maximal set of connected dense units in a subspace
    * A unit is dense if the fraction of total data points contained in the unit exceeds the input model parameter
  * Subspace clustering: A subspace cluster is a set of neighboring dense cells in a arbitrary subspace. It also discovers some minimal descriptions of the clusters <br/><br/>

* CLIQUE can be considered as both density-based and grid-based:
  * It partitions each dimension into the same number of equal-length intervals.
  * It partitions an m-dimensional data space into non-overlapping rectangular units.
  * A unit is dense if the fraction of the total data points contained in the unit exceeds the input model parameter.
  * A cluster is a maximal set of connected dense units within a subspace. <br/><br/>
* It automatically identifies subspaces of a high dimensional data space that allow better clustering than original space using the Apriori principle <br/><br/>
* Strengths
  * Automatically finds subspaces of the highest dimensionality as long as high density cluster exist in those subspaces
  * Insensitive to the order of records in input and does not presume some canonical data distribution
  * Scales linearly with the size of input and has good scaliability as the number of dimensions in the data increases <br/><br/>
* Weaknesses
  * As in all grid-based clustering approaches, the quality of the results crucially depends on the appropriate choice of the number and width of the partitions and grid cells.

### WaveCluster
It is a multi-resolution clustering approach which applies wavelet transform to the feature space

* **

## 8. Model Based Clustering
### COBWEB
### GMM
### [SOM](https://sites.pitt.edu/~is2470pb/Spring05/FinalProjects/Group1a/tutorial/som.html) (Self-Organizing Maps) - Neural network based
SOM are unsupervised neural networks that cluster high-dimensional data

Hypothesis: The model self-orgainzes based on learning rules and interactions. Processing units maintain proximity relationships as they grow. [48:00 viedo](https://www.youtube.com/watch?v=lFbxTlD5R98)
  * Goal:
    1. Find weight values such that adjacent units have similar values
    2. Inputs are assigned to units that are similar to them
    3. Each unit becomes the cebter if a cluster (SOM is a constraint K-means!)
  * SOM has 3 stages:
    1. **Competition** (at the beginning, every neuron will compete with every other neuros to represent the input)
       * Find the most similar unit
    2. **Collaboration** (the concept of neighbourhood)
       * Use the lateral distance d(ij) between the winner unit i and unit j
    3. **Weight update**
  * Convergence: Many iterations
  * Stopping: no noticeable chang; no big change in the feature map
  * Problems: convergence may take a long tim; variable results
  <br/><br/>
* Self Organizing Map(SOM) by Teuvo Kohonen provides a data visualization technique which helps to understand high dimensional data by reducing the dimensions of data to a map. 
* SOM reduces data dimensions and displays similarities among data.
* Reducing Data Dimensions
  * Unlike other learning technique in neural networks, training a SOM requires no target vector. 
  * A SOM learns to classify the training data without any external supervision.
* An SOM is a type of artificial neural network but is trained using competitive learning rather than the error-correction learning (e.g., backpropagation with gradient descent) used by other artificial neural networks. 
* [from sklearn_som.som import SOM](https://pypi.org/project/sklearn-som/)
* Sample notebook: https://www.youtube.com/watch?v=0qtvb_Nx2tA
  * from minisom import MinSom: https://github.com/JustGlowing/minisom
* Reference: https://www.analyticsvidhya.com/blog/2021/09/beginners-guide-to-anomaly-detection-using-self-organizing-maps/
### ART

* **

## 9. [Kernal Based Clustering](https://en.wikipedia.org/wiki/Kernel_method)
The basic idea of this kind of clustering algorithms is that data in the input space is transformed into the feature space of high dimension by the nonlinear mapping for the cluster analysis

Kernel methods owe their name to the use of kernel functions, which enable them to operate in a high-dimensional, implicit feature space without ever computing the coordinates of the data in that space, but rather by simply computing the inner products between the images of all pairs of data in the feature space. This operation is often computationally cheaper than the explicit computation of the coordinates. This approach is called the “kernel trick.”²

Popular Kernels:
* Fisher kernel
* Graph kernel
* Kernel smoother
* Polynomial kernel
* Radial basis fnction kernel (RBF)
* String kernel
* Neural tangent kernel
* Neural network Gaussian process (NNGP) kernel

### Kernel K-means
Kernel k-means clustering is a powerful tool for unsupervised learning of non-linearly separable data. Since the earliest attempts, researchers have noted that such algorithms often become trapped by local minima arising from non-convexity of the underlying objective function. In this paper, we generalize recent results leveraging a general family of means to combat sub-optimal local solutions to the kernel and multi-kernel settings. Called Kernel Power k-Means, our algorithm makes use of majorization-minimization (MM) to better solve this non-convex problem. We show the method implicitly performs annealing in kernel feature space while retaining efficient, closed-form updates, and we rigorously characterize its convergence properties both from computational and statistical points of view. In particular, we characterize the large sample behavior of the proposed method by establishing strong consistency guarantees. Its merits are thoroughly validated on a suite of simulated datasets and real data benchmarks that feature non-linear and multi-view separation.
### Kernel SOM
### kernel FCM
### SVC
### MMC
### MKC

* **

# Dimensionality Reduction Techniques (DRT)
**Curse of Dimensionality** - means that if the amount of data for which to train a model is fixed, then increasing dimensionality can lead to overfitting. **DRT** reduces the dimensionality of data by eliminating irrelevant features that make it feasible to train data for ML models.

* Reference: [Overview and comparative study of dimensionality reduction techniques for high dimensional data](https://www.sciencedirect.com/science/article/pii/S156625351930377X)
  1. Linear Dimension reduction techniques
      * PCA (principal component analysis)
      * SVD (singular value decomposition)
      * LSA (latent semantic analysis)
      * LPP (locality preserving projections)
      * ICA (independent component analysis)
      * LDA (linear discriminant analysis)
      * PP (projection pursuit)<br/><br/>
    
  2. Non-linear Dimensionality Reduction Techniques
     * Kernel PCA
     * MDS (multidimensional scaling)
     * Isomap
     * LLE (locally linear embedding)
     * SOM (self-organizing map)
     * LVQ (learning bector quantitization)
     * t-SNE (t-Stochastic Neighbor Embedding)
     * Autoencoders<br/><br/>

* [Ten quick tips for effective dimensionality reduction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006907)
  * Tip 1: Choose an appropriate method
  * Tip 2: Preprocess continuous and count input data
  * Tip 3: Handle categorical input data appropriately
  * Tip 4: Use embedding methods for reducing similarity and dissimilarity input data
  * Tip 5: Consciously decide on the number of dimensions to retain
  * Tip 6: Apply the correct aspect ratio for your visualizations
  * Tip 7: Understand the meaning of the new dimensions
  * Tip 8: Find the hidden signal
  * Tip 9: Take advantage of multidomain data
  * Tip 10: Check the robustness of your results and quantify uncertainties

* **
## Clustering Evaluation
https://en.wikipedia.org/wiki/Cluster_analysis#Evaluation_and_assessment

* Internal Evaluation
  * Davies-Bouldin Indicator
  * Dunn Indicator
  * Silhouette <br/><br/>
* External Evaluation
  * Rand Indicator
  * F Indicator
  * Jaccard Indicator
  * Fowlkes-Mallows Indicator
  * Confusion Matrix


  
https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf







