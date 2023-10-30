rm(list=ls())
dev.off()

# Libraries

library(ggplot2)
library(glmnet)
library(caret)
library(rpart)

# setwd("C:/Users/Alessandro/Desktop/PIETRO/Università/3_Machine Learning and Data Mining/Exercises/02450Toolbox_R")

data(diamonds)
str(diamonds)

################################
# ---- Linear Regression  -----
################################

################################
# -------  Question 1 ---------
################################





################################
# -------  Question 2 ---------
################################





################################
# -------  Question 3 ---------
################################


################################
# -------  Question 4 ---------
################################



################################
# -------  Question 5 ---------
################################



################################
# -----  Classification  -------
################################

summary(diamonds$cut)
summary(diamonds$cut == 'Ideal')
(percentage_of_Ideal <- sum(diamonds$cut == 'Ideal')/length(diamonds$cut))

y <- as.numeric(diamonds$cut == 'Ideal')
X <- as.data.frame(diamonds[,c(1,5:10)])

# CHOOSE AN OPTION AMONG 1, 2 AND 3: ###########

# 1) Linear model
attributeNames <- colnames(X)

# Classification tree
classNames <- c("Non-Ideal","Ideal")
classassignments <- classNames[y + 1]
(fmla <- as.formula(paste("y ~ ", paste(attributeNames, collapse = "+"))))


# # 2) Quadratic model
# X <- transform(X,
#                carat2 = carat^2,
#                depth2 = depth^2,
#                table2 = table^2,
#                price2 = price^2,
#                x2 = x^2,
#                y2 = y^2,
#                z2 = z^2,
#                carat_depth = carat*depth,
#                carat_table = carat*table,
#                carat_price = carat*price,
#                carat_x = carat*x,
#                carat_y = carat*y,
#                carat_z = carat*z,
#                depth_table = depth*table,
#                depth_price = depth*price,
#                depth_x = depth*x,
#                depth_y = depth*y,
#                depth_z = depth*z,
#                table_price = table*price,
#                table_x = table*x,
#                table_y = table*y,
#                table_z = table*z,
#                price_x = price*x,
#                price_y = price*y,
#                price_z = price*z,
#                x_y = x*y,
#                x_z = x*z,
#                y_z = y*z
# )
# attributeNames <- colnames(X)

# # 3) Quadratic model applied to data projected onto the first four principal components
# stds <- apply(X, 2, sd)
# X <- t(apply(X, 1, "-", colMeans(X)))
# X <- t(apply(X, 1, "*", 1 / stds))
# X <- as.data.frame(X)
# S <- svd(X)
# X <- as.data.frame(S$u %*% diag(S$d))
# X <- X[,1:4]
# X <- transform(X,
#                V1_2 = V1^2,
#                V2_2 = V2^2,
#                V3_2 = V3^2,
#                V4_2 = V4^2,
#                V1_V2 = V1*V2,
#                V1_V3 = V1*V3,
#                V1_V4 = V1*V4,
#                V2_V3 = V2*V3,
#                V2_V4 = V2*V4,
#                V3_V4 = V3*V4
# )
# rm(S)

# END OF THE CHOICE #############################

head(X)

N <- as.numeric(dim(X)[1])
M <- as.numeric(dim(X)[2])

# K-folds cross-validation ######################

# Create cross-validation partition for evaluation of performance of optimal model
K <- 3
KK <- 5 # nr. of Inner loops # Use 10-fold cross-validation to estimate optimal value of lambda

# Values of lambda
lambda_tmp <- 10^(-5:-2)
cp_tmp <- c(0.05,0.01,0.005,0.001)
T <- length(lambda_tmp) # nr. of tested lambda

# Set seed for reproducibility
set.seed(4321)

CV <- list()
CV$which <- createFolds(y, k = K, list = F)

# Set up vectors that will store sizes of training and test sizes
CV$TrainSize <- c()
CV$TestSize <- c()

# Initialize variables
Error_train2 <- matrix(rep(NA, times = T * KK), nrow = T)
Error_test2 <- matrix(rep(NA, times = T * KK), nrow = T)
Error_train2_tree <- matrix(rep(NA, times = T * KK), nrow = T)
Error_test2_tree <- matrix(rep(NA, times = T * KK), nrow = T)
lambda_opt <- rep(NA, K)
cp_opt <- rep(NA, K)
mu <- matrix(rep(NA, times = M * K), nrow = K)
sigma <- matrix(rep(NA, times = M * K), nrow = K)
Error_train_rlr <- rep(NA, K) # Rate error of the regularized logistic regression
Error_test_rlr <- rep(NA, K)
Error_train <- rep(NA, K) # Rate error of the non-regularized logistic regression
Error_test <- rep(NA, K)
Error_train_nofeatures <- rep(NA, K) # Rate error of the baseline
Error_test_nofeatures <- rep(NA, K)
Error_train_tree <- rep(NA, K) # Rate error of the classification tree
Error_test_tree <- rep(NA, K)

for (k in 1:K) {
  paste("Crossvalidation fold ", k, "/", K, sep = "")
  
  # Extract the training and test set
  X_train <- X[CV$which != k, ]
  y_train <- y[CV$which != k]
  X_test <- X[CV$which == k, ]
  y_test <- y[CV$which == k]
  CV$TrainSize[k] <- length(y_train)
  CV$TestSize[k] <- length(y_test)
  
  CV2 <- list()
  CV2$which <- createFolds(y_train, k = KK, list = F)
  CV2$TrainSize <- c()
  CV2$TestSize <- c()
  
  
  for (kk in 1:KK) {
    X_train2 <- X_train[CV2$which != kk, ]
    y_train2 <- y_train[CV2$which != kk]
    X_test2 <- X_train[CV2$which == kk, ]
    y_test2 <- y_train[CV2$which == kk]
    fmla2 <- as.formula(paste("y_train2 ~ ", paste(attributeNames, collapse = "+")))
    
    mu2 <- colMeans(X_train2[,1:M])
    sigma2 <- apply(X_train2[,1:M], 2, sd)
    
    X_train2[,1:M] <- scale(X_train2[,1:M], mu2, sigma2)
    X_test2[,1:M] <- scale(X_test2[,1:M], mu2, sigma2)
    
    CV2$TrainSize[kk] <- length(y_train2)
    CV2$TestSize[kk] <- length(y_test2)
    
    mdl <- glmnet(X_train2, y_train2, family = "binomial", alpha = 0,
                  lambda = lambda_tmp,intercept=T)
    
    for (t in 1:T) {
      y_train2 <- as.character(y_train2)
      tree <- rpart(fmla2, data = X_train2,
                    control = rpart.control(minsplit = 100, minbucket = 1, cp = cp_tmp[t]),
                    parms = list(split = "gini"), method = "class")
      y_train2 <- as.numeric(y_train2)
      y_train_tree <- as.numeric(predict(tree, newdata=X_train2)[,2]>0.5)
      y_test_tree <- as.numeric(predict(tree, newdata=X_test2)[,2]>0.5)
      
      Error_train2_tree[t, kk] <- sum(y_train_tree != y_train2) / length(y_train2)
      Error_test2_tree[t, kk] <- sum(y_test_tree != y_test2) / length(y_test2)
      
      # Predict labels for both sets for current regularization strength
      y_train_est <- predict(mdl, newx=as.matrix(X_train2), type = "class",
                             s = lambda_tmp[t])
      y_test_est <- predict(mdl, newx=as.matrix(X_test2), type = "class",
                            s = lambda_tmp[t])
      
      # Determine training and test set error
      Error_train2[t, kk] <- sum(y_train_est != y_train2) / length(y_train2)
      Error_test2[t, kk] <- sum(y_test_est != y_test2) / length(y_test2)
    }
  }
  # Select optimal value of lambda
  ind_opt <- which.min(apply(Error_test2, 1, sum) / sum(CV2$TestSize))
  lambda_opt[k] <- lambda_tmp[ind_opt]
  
  ind_opt_tree <- which.min(apply(Error_test2_tree, 1, sum) / sum(CV2$TestSize))
  cp_opt[k] <- cp_tmp[ind_opt_tree]
  
  # Standardize outer fold based on training set, and save the mean and standard
  # deviations since they're part of the model (they would be needed for
  # making new predictions) - for brevity we won't always store these in the scripts
  mu[k, ] <- colMeans(X_train[,1:M])
  sigma[k, ] <- apply(X_train[,1:M], 2, sd)
  
  X_train[,1:M] <- scale(X_train[,1:M], mu[k, ], sigma[k, ])
  X_test[,1:M] <- scale(X_test[,1:M], mu[k, ], sigma[k, ])
  
  ### tree ###
  fmla1 <- as.formula(paste("y_train ~ ", paste(attributeNames, collapse = "+")))
  y_train <- as.character(y_train)
  tree <- rpart(fmla1, data = X_train,
                control = rpart.control(minsplit = 100, minbucket = 1, cp = cp_opt[k]),
                parms = list(split = "gini"), method = "class")
  y_train <- as.numeric(y_train)
  y_train_tree <- as.numeric(predict(tree, newdata=X_train)[,2]>0.5)
  y_test_tree <- as.numeric(predict(tree, newdata=X_test)[,2]>0.5)
  
  Error_train_tree[k] <- sum(y_train_tree != y_train) / length(y_train)
  Error_test_tree[k] <- sum(y_test_tree != y_test) / length(y_test)
  
  ### end of tree ###
  
  ### regularized logistic regression ###
  # Estimate w for the optimal value of lambda
  mdl <- glmnet(X_train, y_train, family = "binomial", alpha = 0,
                lambda = lambda_opt[k], intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class",
                         s = lambda_opt[k])
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class",
                        s = lambda_opt[k])
  
  # evaluate training and test error performance for optimal selected value of lambda
  Error_train_rlr[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test_rlr[k] <- sum(y_test_est != y_test) / length(y_test)
  ### end of regularized logistic regression ###
  
  # Compute squared error without regularization

  mdl <- glmnet(X_train, y_train, family = "binomial", alpha = 0,
                lambda = 1e-10, intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class", s = 1e-10)
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class", s = 1e-10)
  
  Error_train[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test[k] <- sum(y_test_est != y_test) / length(y_test)
  
  Error_train_nofeatures[k] <- sum(y_train_est != 0) / length(y_train)
  Error_test_nofeatures[k] <- sum(y_train_est != 0) / length(y_train)
}

# EXPLANATION OF (1): Classification errors of models with and without regularization are the same
# because the dataset has a large number of observations but the model has only 8 parameters
# to be estimated. Regularization improves the model particularly when it is overfitted, and
# it happens when observations are few wrt parameters to be estimated





################################################################
# APPEDIX
################################################################
Z <- transform(Z,
               V1_2 = V1^2,
               V2_2 = V2^2,
               V3_2 = V3^2,
               V4_2 = V4^2,
               V1_3 = V1^3,
               V2_3 = V2^3,
               V3_3 = V3^3,
               V4_3 = V4^3,
               V1_V2 = V1*V2,
               V1_V3 = V1*V3,
               V1_V4 = V1*V4,
               V1_2_V2 = V1^2*V2,
               V1_V2_2 = V1*V2^2,
               V1_2_V3 = V1^2*V3,
               V1_V3_2 = V1*V3^2,
               V1_2_V4 = V1^2*V4,
               V1_V4_2 = V1*V4^2,
               V2_V3 = V2*V3,
               V2_V4 = V2*V4,
               V2_2_V3 = V2^2*V3,
               V2_V3_2 = V2*V3^2,
               V2_2_V4 = V2^2*V4,
               V2_V4_2 = V2*V4^2,
               V3_V4 = V3*V4,
               V3_2_V4 = V3^2*V4,
               V3_V4_2 = V3*V4^2
)
