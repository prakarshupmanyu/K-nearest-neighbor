# K-nearest-neighbor
A Python3 implementation of K-nearest neighbor algorithm

1. Dataset (CIFAR-10)
   - Download the python version of CIFAR-10 dataset from the link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    
   - Look into the description on the web page (https://www.cs.toronto.edu/~kriz/cifar.html) to figure out how to extract the data from the downloaded pickle files.
   - Extract the first 1000 images from the data batch 1 for the following problems.
   - Split the images into training and testing sets. The first N images are used as testing images which are queries for K-NN classifier. The rest of (1000 − N ) images are used for training. (N is specified as an input argument.)
   
2. Principal Component Analysis(PCA)
   - Convert the RGB images to grayscale with the following formula (you should do it manually without using any package):
      L(Grayscale) = 0.299R + 0.587G + 0.114B
   - Compute the PCA transformation by using only the training set with the scikit-learn PCA package.
   - Perform dimensionality reduction on both training and testing sets to reduce the dimension of data from 1024 to D. (D is specified as an input argument.)
   - Specify a full SVD solver to build the PCA embeddings (i.e. svd solver=“full”). (The results may not be consistent if the randomized truncated SVD solver is used in the scikit-learn PCA package.)
   
3. K-Nearest Neighbors (K-NN)
    - Implement a K-Nearest Neighbors classifier to predict the class labels of testing images. Make sure you use the inverse of Euclidean distance as the metric for the voting. In other words, each neighbor ni , where i = 1, ..., K, represented as a vector, contributes to the voting with the weight of 1/(||x−n||2), where x is a queried vector. (K is specified as an input argument.)
    - **Not using the scikit-learn library to implement K-NN.**
  
  ***Running the Script***
  
  python3 knn.py K D N <path_to_data>
  
  **Output**
  
  Output will be stored in output.txt file in the same directory.
