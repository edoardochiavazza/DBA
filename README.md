# Project Summary: Advanced Data Mining (BDM) 2025

This project explores various techniques in image feature extraction, similarity measurement, dimensionality reduction, clustering, classification, and indexing. Below is a summary of the tasks involved:

## Task 0: Setup and Familiarization
- **Objective**: Set up the programming environment and familiarize yourself with the tools and datasets.
- **Tools**: Python, PyTorch, TorchVision, NumPy, and SciPy.
- **Dataset**: A modified version of the BRAIN MRI dataset (provided separately).
- **Pre-trained Model**: ResNet50 with default weights.
- **Storage**: Freedom to use relational databases (e.g., MySQL), NoSQL databases (e.g., MongoDB), or custom file structures.

## Task 1: Feature Extraction and Visualization
- **Objective**: Implement a program to extract and display feature descriptors for a given image.
- **Feature Models**:
  - **CM10x10**: Color moments (mean, standard deviation, skewness) in a 10x10 grid (900-dimensional descriptor).
  - **HOG**: Histograms of Oriented Gradients (900-dimensional descriptor).
  - **ResNet Variants**: Extract features from ResNet50 layers (AvgPool, Layer3, FC) with dimensionality adjustments.

## Task 2: Batch Feature Extraction
- **Objective**: Extract and store feature descriptors for all images in the dataset.

## Task 3: Similarity Search
- **Objective**: Given an image and a value `k`, retrieve and visualize the `k` most similar images using appropriate distance/similarity measures for each feature model. Display the similarity scores.

## Task 4: Label Prediction
- **Objective**: For a query image, a selected feature space, and `k`, predict the top `k` most likely labels along with their scores.

## Task 5: Dimensionality Reduction (LS1)
- **Objective**: Apply SVD, LDA, or k-means to reduce dimensions for a given feature model and extract top-k latent semantics. Store results in an output file with imageID-weight pairs.

## Task 6: Inherent Dimensionality Analysis
- **Task 7a**: Compute and print the inherent dimensionality of part 1 images.
- **Task 7b**: Compute and print the inherent dimensionality for each unique label in part 1 images.

## Task 7: Label-Specific Latent Semantics
- **Objective**: For each unique label, compute k latent semantics for part 1 images and predict labels for part 2 images using these semantics. Output precision, recall, F1-score, and overall accuracy.

## Task 8: Clustering with DBScan
- **Objective**: For each unique label, compute `c` significant clusters for part 1 images using DBScan. Visualize clusters as colored point clouds in 2D MDS space and as groups of image thumbnails.

## Task 9: Classification
- **Objective**: Train an m-NN classifier and a decision-tree classifier on part 1 images. Predict labels for part 2 images and output performance metrics (precision, recall, F1-score, accuracy).

## Task 10: Locality Sensitive Hashing (LSH)
- **10a**: Implement an LSH index for Euclidean distance with configurable layers (`L`) and hashes per layer (`h`).
- **10b**: Perform similar image search using the LSH index. For a query image and `t`, visualize the top `t` matches and report the number of unique/overall images considered during the search.

---

This project covers a wide range of techniques in data mining and computer vision, providing hands-on experience with feature extraction, similarity measures, dimensionality reduction, clustering, classification, and efficient indexing. The tasks are designed to build a comprehensive understanding of these concepts while working with real-world MRI data.
