# Machine-Learning-Workflow-in-R

This post deals with machine learning problem of prediction. The goal of this document is to create a replicable machine learning workflow in R using caret, tidymodels, tidyverse libraries. I will employ various modern techniques of feature engineering, feature selection, and machine learning models.

1. EDA to inform how we will do feature engineering 

2. Feature engineering process 

    a. Categoical Variable:
    
    - Dummy Encoding
    - One Hot Encoding
    - Label Encoding
    - Other Category
    - Novel Category Problem
    - Too Many Class Values Problem: Feature Hashing
    - Supervised Encoding Methods
      - Effect Encoding
      - Embedding

    b. Numeric Variable:
    
    - Centering, Scaling, Standardizing, Normalizing
    - Univariate Transformation: log/square root/inverse, Box-Cox, Yeo-Johnson
    - Basis Expansions: Splines, GAM, MARS
    - Dimensionality Reduction: PCA, Kernel PCA, ICA, NMF, PLS, Auto Encoder
    - Depth/Distance, Spatial Sign

    c. Missing Value:
    - Remmove Rows
    - Create an Unknown Category
    - Imputation:
      - Mean, Mode, Median
      - Rolling Statistics
      - KNN
      - Bagged Trees

    d. Interaction:
    
    - Interaction search methods

    e. Text Data:
    
    - Bag of Words
    - Word Embedding
      - Word2Vec
      - Doc2Vec

3. Modeling

    Using Caret library in R, I created a wrapper function that automates regression with a single line of code. It takes in two arguments: data and location where it will store results in csv and rds format. The function runs 30 different statistical and machine learning algorithms from linear regression to neural network. The function returns the following four items in the given location:

      a. summary table of evaluation metrics (R-squared, RMSE) of every regression model on test set

      b. predicted outputs of every regression model on test set

      c. variable importance plot of every regression model

      d. best hyperparameters on grid search

        - Linear Models
        - Discriminants
        - Support Vector Machines
        - Tree Models
        - Neural Networks

