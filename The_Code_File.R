rm(list=ls())
dev.off()

# Libraries

library(ggplot2)
library(glmnet)

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

X <- as.data.frame(diamonds[,c(1,3:10)])
X <- X[,-(2:3)] # Remove Color and Clarity, prediction based on quantitative attributes
head(X)
y <- as.numeric(diamonds$cut == 'Ideal')
N <- dim(X)[1]
M <- dim(X)[2]
attributeNames <- colnames(X)

################################
# Holdout method
################################

# 8.1.2
# ----------------
# Cross-validation
# ----------------

# For reproducibility, if needed
set.seed(5142)

CV <- list()
CV$which <- createFolds(y, k = 20, list = F)
CV$TrainSize <- c()
CV$TestSize <- c()

# Extract the training and test set
X_train <- X[CV$which != 1, ]
# Train on 1/20 of the data, or 5 percent
y_train <- y[CV$which != 1]
X_test <- X[CV$which == 1, ]
# Test on the rest
y_test <- y[CV$which == 1]
CV$TrainSize[1] <- length(y_train)
CV$TestSize[1] <- length(y_test)

# Standardize based on training set
mu <- colMeans(X_train)
sigma <- apply(X_train, 2, sd)

X_train <- data.frame(scale(X_train, mu, sigma))
X_test <- data.frame(scale(X_test, mu, sigma))

#----------
# Fit model
#----------

# Fit logistic regression model to training data to predict the type of wine

N_lambdas <- 20
lambda_tmp <- 10^(seq(from = -8, to = 0, length = N_lambdas))

# alpha=0 gives ridge regression
# We will use glmnet to fit, which you can install using: install.packages('glmnet')
mdl <- glmnet(X_train, y_train, family = "binomial", alpha = 0,
              lambda = lambda_tmp)

train_error <- rep(NA, N_lambdas)
test_error <- rep(NA, N_lambdas)
coefficient_norm <- rep(NA, N_lambdas)
for (k in 1:N_lambdas) {
  # Predict labels for both sets for current regularization strength
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class",
                         s = lambda_tmp[k])
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class",
                        s = lambda_tmp[k])
  
  # Determine training and test set error
  train_error[k] <- sum(y_train_est != y_train) / length(y_train)
  test_error[k] <- sum(y_test_est != y_test) / length(y_test)
  
  # Determine betas and calculate norm of parameter vector
  w_est <- predict(mdl, type = "coef", s = lambda_tmp[k])[-1]
  coefficient_norm[k] <- sqrt(sum(w_est^2))
}

(min_error <- min(test_error))
lambda_opt <- lambda_tmp[which.min(test_error)]

#-------------
# Plot results
#-------------

par(mfrow = c(1, 1))
par(cex.main = 1.5) # Define size of title
par(cex.lab = 1) # Define size of axis labels
par(cex.axis = 1) # Define size of axis labels

# Plot classification error

{
  plot(range(log10(lambda_tmp)), range(100 * c(test_error, train_error)),
       type = "n",
       xlab = "Log10(lambda)", ylab = "Error (%)",
       main = "Classification error"
  )
  
  lines(log10(lambda_tmp), train_error * 100, col = "red")
  lines(log10(lambda_tmp), test_error * 100, col = "blue")
  points(log10(lambda_opt), min_error * 100, col = "green", cex = 5)
  legend("topleft", c(
    paste("Training, n=", round(length(y_train), 2)),
    paste("Test, n=", round(length(y_test), 2))
  ),
  col = c("red", "blue"), lty = 1, cex = 1
  )
  grid()
}

# Plot classification error (zoomed)
{
  plot(range(-6, -1), range(18, 22),
       type = "n",
       xlab = "Log10(lambda)", ylab = "Error (%)", main = "Classification error (zoomed)"
  )
  lines(log10(lambda_tmp), train_error * 100, col = "red")
  lines(log10(lambda_tmp), test_error * 100, col = "blue")
  points(log10(lambda_opt), min_error * 100, col = "green", cex = 5)
  text(-4, 0.5,
       labels = paste("Min error test: ", round(min_error * 100, 2), " % at 1e",
                      round(log10(lambda_opt), 1)),
       cex = 1
  )
  grid()
}

# Plot regularization vector
{
  plot(range(log10(lambda_tmp)), range(coefficient_norm),
       type = "n",
       xlab = "Log10(lambda)", ylab = "Norm",
       main = "Parameter vector L2-norm"
  )
  lines(log10(lambda_tmp), coefficient_norm)
  grid()
}

################################
# K-folds cross-validation
################################

# -----------------------------
# Regularized Linear regression
# -----------------------------

# Cross-validation

# Create cross-validation partition for evaluation of performance of optimal model
K <- 5

# Set seed for reproducibility
set.seed(4321)

CV <- list()
CV$which <- createFolds(y, k = K, list = F)

# Set up vectors that will store sizes of training and test sizes
CV$TrainSize <- c()
CV$TestSize <- c()

# Values of lambda
lambda_tmp <- 10^(-5:8)

# Initialize variables

KK <- 10 # nr. of Inner loops
T <- 14 # nr. of tested lambda
temp <- rep(NA, M * T * KK)
w <- array(temp, c(M, T, KK))
Error_train2 <- matrix(rep(NA, times = T * KK), nrow = T)
Error_test2 <- matrix(rep(NA, times = T * KK), nrow = T)
lambda_opt <- rep(NA, K)
w_rlr <- matrix(rep(NA, times = M * K), nrow = M)
Error_train_rlr <- rep(NA, K)
Error_test_rlr <- rep(NA, K)
w_noreg <- matrix(rep(NA, times = M * K), nrow = M)
mu <- matrix(rep(NA, times = M * K), nrow = K)
sigma <- matrix(rep(NA, times = M * K), nrow = K)
Error_train <- rep(NA, K)
Error_test <- rep(NA, K)
Error_train_nofeatures <- rep(NA, K)
Error_test_nofeatures <- rep(NA, K)


for (k in 1:K) {
  paste("Crossvalidation fold ", k, "/", K, sep = "")
  
  # Extract the training and test set
  X_train <- X[CV$which != k, ]
  y_train <- y[CV$which != k]
  X_test <- X[CV$which == k, ]
  y_test <- y[CV$which == k]
  CV$TrainSize[k] <- length(y_train)
  CV$TestSize[k] <- length(y_test)
  
  # Use 10-fold cross-validation to estimate optimal value of lambda
  KK <- 10
  
  CV2 <- list()
  CV2$which <- createFolds(y_train, k = KK, list = F)
  CV2$TrainSize <- c()
  CV2$TestSize <- c()
  
  
  for (kk in 1:KK) {
    X_train2 <- X_train[CV2$which != kk, ]
    y_train2 <- y_train[CV2$which != kk]
    X_test2 <- X_train[CV2$which == kk, ]
    y_test2 <- y_train[CV2$which == kk]
    
    mu2 <- colMeans(X_train2[,1:M])
    sigma2 <- apply(X_train2[,1:M], 2, sd)
    
    X_train2[,1:M] <- scale(X_train2[,1:M], mu2, sigma2)
    X_test2[,1:M] <- scale(X_test2[,1:M], mu2, sigma2)
    
    CV2$TrainSize[kk] <- length(y_train)
    CV2$TestSize[kk] <- length(y_test2)
    
    mdl <- glmnet(X_train2, y_train2, family = "binomial", alpha = 0,
                  lambda = lambda_tmp,intercept=T)
    
    train_error <- rep(NA, T)
    test_error <- rep(NA, T)
    coefficient_norm <- rep(NA, T)
    for (k in 1:T) {
      # Predict labels for both sets for current regularization strength
      y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class",
                             s = lambda_tmp[k])
      y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class",
                            s = lambda_tmp[k])
      
      # Determine training and test set error
      train_error[k] <- sum(y_train_est != y_train) / length(y_train)
      test_error[k] <- sum(y_test_est != y_test) / length(y_test)
      
      # Determine betas and calculate norm of parameter vector
      w_est <- predict(mdl, type = "coef", s = lambda_tmp[k])[-1]
      coefficient_norm[k] <- sqrt(sum(w_est^2))
    }
  }
  
  # Select optimal value of lambda
  min_error <- min(test_error)
  lambda_opt[k] <- lambda_tmp[which.min(test_error)]
  
  # Standardize outer fold based on training set, and save the mean and standard
  # deviations since they're part of the model (they would be needed for
  # making new predictions) - for brevity we won't always store these in the scripts
  mu[k, ] <- colMeans(X_train[,1:M])
  sigma[k, ] <- apply(X_train[,1:M], 2, sd)
  
  X_train[,1:M] <- scale(X_train[,1:M], mu[k, ], sigma[k, ])
  X_test[,1:M] <- scale(X_test[,1:M], mu[k, ], sigma[k, ])
  
  ###
  # Estimate w for the optimal value of lambda
  mdl <- glmnet(X_train, y_train, family = "binomial", alpha = 0,
                lambda = lambda_tmp,intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class",
                         s = lambda_opt[k])
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class",
                        s = lambda_opt[k])
  
  # evaluate training and test error performance for optimal selected value of lambda
  Error_train_rlr[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test_rlr[k] <- sum(y_test_est != y_test) / length(y_test)
  ###
  
  # # Compute squared error without regularization
  # # Adds a small value to diagonal to avoid a singular matrix
  # w_noreg[, k] <- solve(XtX + (diag(M) * 1e-10)) %*% Xty
  # Error_train[k] <- sum((y_train - X_train %*% w_noreg[, k])^2)
  # Error_test[k] <- sum((y_test - X_test %*% w_noreg[, k])^2)
  # 
  # # Compute squared error without using the input data at all
  # Error_train_nofeatures[k] <- sum((y_train - mean(y_train))^2)
  # Error_test_nofeatures[k] <- sum((y_test - mean(y_train))^2)
  
  # if (k == K) {
  #   dev.new()
  #   # Display result for cross-validation fold
  #   w_mean <- apply(w, c(1, 2), mean)
  #   
  #   # Plot weights as a function of the regularization strength (not offset)
  #   par(mfrow = c(1, 2))
  #   par(cex.main = 1.5) # Define size of title
  #   par(cex.lab = 1) # Define size of axis labels
  #   par(cex.axis = 1) # Define size of axis labels
  #   par(mar = c(5, 4, 3, 1) + .1) # Increase margin size to allow for larger axis labels
  #   
  #   plot(log(lambda_tmp), w_mean[2, ],
  #        xlab = "log(lambda)",
  #        ylab = "Coefficient Values", main = paste("Weights, fold ", k, "/", K),
  #        ylim = c(min(w_mean[-1, ]), max(w_mean[-1, ]))
  #   )
  #   lines(log(lambda_tmp), w_mean[2, ])
  #   
  #   colors_vector <- colors()[c(1, 50, 26, 59, 101, 126, 151, 551, 71, 257, 506, 634, 639, 383)]
  #   
  #   for (i in 3:M) {
  #     points(log(lambda_tmp), w_mean[i, ], col = rainbow(T)[i])
  #     lines(log(lambda_tmp), w_mean[i, ], col = rainbow(T)[i])
  #   }
  #   
  #   plot(log(lambda_tmp), log(apply(Error_train2, 1, sum) / sum(CV2$TrainSize)),
  #        xlab = "log(lambda)", ylab = "log(Error)",
  #        main = paste0("Optimal lambda: 1e", log10(lambda_opt[k]))
  #   )
  #   
  #   lines(log(lambda_tmp), log(apply(Error_train2, 1, sum) / sum(CV2$TrainSize)))
  #   points(log(lambda_tmp), log(apply(Error_test2, 1, sum) / sum(CV2$TestSize)), col = "red")
  #   lines(log(lambda_tmp), log(apply(Error_test2, 1, sum) / sum(CV2$TestSize)), col = "red")
  #   
  #   legend("bottomright", legend = c("Training", "Test"), col = c("black", "red"), lty = 1)
  #   
  # }
}

# Display Results
writeLines("Linear regression without feature selection:")
writeLines(paste("- Training error: ", sum(Error_train) / sum(CV$TrainSize)))
writeLines(paste("- Test error", sum(Error_test) / sum(CV$TestSize)))
writeLines(paste("- R^2 train:     %8.2f\n", (sum(Error_train_nofeatures) - sum(Error_train)) / sum(Error_train_nofeatures)))
writeLines(paste("- R^2 test:     %8.2f\n", (sum(Error_test_nofeatures) - sum(Error_test)) / sum(Error_test_nofeatures)))

writeLines("Regularized Linear regression:")
writeLines(paste("- Training error:", sum(Error_train_rlr) / sum(CV$TrainSize)))
writeLines(paste("- Test error:", sum(Error_test_rlr) / sum(CV$TestSize)))
writeLines(paste("- R^2 train: ", (sum(Error_train_nofeatures) - sum(Error_train_rlr)) / sum(Error_train_nofeatures)))
writeLines(paste("- R^2 test:", (sum(Error_test_nofeatures) - sum(Error_test_rlr)) / sum(Error_test_nofeatures)))


writeLines("Weights in last fold :")
for (m in 1:M) {
  writeLines(paste(attributeNames[m], w_rlr[m, k]))
}

