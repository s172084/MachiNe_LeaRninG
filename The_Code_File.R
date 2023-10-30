rm(list=ls())
dev.off()

#---------------------
# Access the Libraries
#---------------------
library(tidyverse)
library(magrittr)
library(forcats)
library(ggplot2)
library(glmnet)
library(caret)

# setwd("C:/Users/Alessandro/Desktop/PIETRO/Università/3_Machine Learning and Data Mining/Exercises/02450Toolbox_R")

#---------------------
# Load the Data 
#---------------------
data(diamonds)
#?diamonds
  
str(diamonds)


#---------------------
# See the Data 
#---------------------
diamonds


#---------------------
# Functions
#---------------------
determine_outliers <- function(group) {
  
  # ----- See 5 quantiles. 
  q <- quantile(group, type = 2)
  cat("QUARTILES:\nThe five quartiles at\n 0%   25 %   50 %   75 %   100 % are:\n",
      q[1]," ", q[2]," ", q[3]," ",q[4]," ", q[5]," ","\n\n")
  
  # ----- See Percentiles. 
  d <- quantile(group, c(0.25, 0.75) , type = 2)
  cat("PERCENTILES:\n25%,  75%\n", d[1]," ", d[2], " ","\n\n")
  
  # ----- Calculate Interquartile Range. 
  that_iqr <- IQR(group, type = 2)
  cat("The interquartile range is :", that_iqr, "\n\n")
  
  
  Q <- quantile(group, probs=c(.25, .75), na.rm = FALSE)
  iqr <- IQR(group)
  
  # ----- Calculate Upper and Lower Limits
  # Anything beyond the upper and lower limit are outliers. 
  upper_limit <- Q[2] + 1.5*iqr 
  lower_limit <- Q[1] - 1.5*iqr 
  cat("The upper limit is: ",upper_limit, "\n")
  cat("The lower limit is: ",lower_limit, "\n")
}


#---------------------
# Transform the Data
#---------------------
# - price Conversion from USD($) to DKK, Euro & Pound Sterling(£)

diamonds_price <- diamonds %>%  
  mutate(priceDKK = price * 7.066874,
         priceEuro = price * 0.949312,
         pricePS = price * 0.824538)

#- Remove the USD($) variable. 
diamonds_price <- diamonds_price %>% 
  select(-price)

diamonds_price

# - carat conversion to milligrams in weight
diam_conv_weight <- diamonds_price %>% 
  mutate(carat_mg = carat * 200) 

# - Remove the carat variable. 
diam_milligram <- diam_conv_weight %>% 
  select(-carat)

diam_milligram

# - X,Y,Z conversion from millimeters (mm) to micrometers(µm)
diamonds_conv <- diam_milligram %>% 
  mutate(the_length = x * 1000,
         the_width = y * 1000,
         the_depth = z * 1000) 

# - Remove the x,y and z variables. 
new_diamonds_data <- diamonds_conv %>% 
  select(- c(x,y,z))

new_diamonds_data

#---------------------
# Detect Outliers
#---------------------

# Quantile Finds the 25th and 75th Percentile of the data. 
# IQR Finds the interquartile range.
# Find the cut off ranges, beyond which all points are outliers. 
# Remove points beyond the ranges from the dataset. 

boxplot(new_diamonds_data)$out

boxplot(data.frame(x= new_diamonds_data$the_length,
                   y= new_diamonds_data$the_width,
                   z= new_diamonds_data$the_depth))
#---------------------
boxplot(c(x = new_diamonds_data$the_length), 
        col= "skyblue",
        family = "Avenir",
        main = "Length")


determine_outliers(new_diamonds_data$the_length)
# quantile(new_diamonds_data$the_length, probs=c(.25, .75), na.rm = FALSE)
# IQR(new_diamonds_data$the_length)
#---------------------

boxplot(c(x = new_diamonds_data$the_width), 
        col= "pink",
        main = "Width",  
        family = "Avenir")


determine_outliers(new_diamonds_data$the_width)
# quantile(new_diamonds_data$the_width, probs=c(.25, .75), na.rm = FALSE)
# IQR(new_diamonds_data$the_width)

#---------------------
boxplot(c(x = new_diamonds_data$the_depth), 
        col= "cyan",
        main = "Depth"
)


determine_outliers(new_diamonds_data$the_depth)
# quantile(new_diamonds_data$the_depth, probs=c(.25, .75), na.rm = FALSE)
# IQR(new_diamonds_data$the_depth)

#---------------------
# View Outliers
#---------------------
# Length : 
# The upper limit is:  9285 
# The lower limit is:  1965 

# Width:
# The upper limit is:  9270 
# The lower limit is:  1990 

# Depth : 
# The upper limit is:  5735 
# The lower limit is:  1215
new_diamonds_data

new_diamonds_data %>% filter (the_length > 9285)
new_diamonds_data %>% filter (the_length < 1965)

new_diamonds_data %>% filter (the_width > 9270)
new_diamonds_data %>% filter (the_width < 1990)

new_diamonds_data %>% filter (the_depth > 5735)
new_diamonds_data %>% filter (the_depth < 1215)

#---------------------
# Remove Outliers
#---------------------
# Extract the part of the dataset between the upper and lower ranges leaving out the outliers.

clean_diamonds_data <- new_diamonds_data %>% 
  filter(the_length < 9285) %>%  # 53,916
  filter(the_length > 1965) %>%  # 53,908
  filter(the_width < 9270) %>%   # 53,896
  filter(the_width > 1990) %>%  
  filter(the_depth < 5735) %>%   # 53,892
  filter(the_depth > 1215)       # 53,879




clean_diamonds_data

clean_diamonds_data <- clean_diamonds_data %>% 
  mutate(cut = forcats::as_factor(cut),
         color = forcats::as_factor(color),
         clarity = forcats::as_factor(clarity))

#---------------------
# The Data Set  -->>>>
#---------------------

clean_diamonds_data

#---------------------
# The Data Set <<<---
#---------------------



#---------------------
# Data Visualisation
#---------------------
clean_diamonds_data  %>% 
  ggplot(mapping = aes(x = the_length, y = cut, fill = cut))+
  coord_flip() + 
  geom_boxplot(colour = "black", alpha = 0.7)+
  scale_fill_brewer(palette="Accent")+
  #facet_wrap(~ cut)+
  theme(
    legend.position = "top",
    axis.line = element_line(colour = "darkblue"),
    panel.grid.major.y = element_line(linetype = "dashed"),
    axis.text.x = element_blank())+
  labs(
    title = "A Box Plot",
    subtitle = "of Diamonds",
    x = "Diamond Length in µm",
    y = "The Type of Cut",
    caption = "Diamonds data from Tidyverse"
  )

library(patchwork)

a <- clean_diamonds_data  %>% 
  ggplot(mapping = aes(x = the_length))+
  coord_flip() + 
  geom_boxplot(colour = "black", fill = "chocolate2", alpha = 0.7)+
  scale_fill_brewer(palette="Accent")+
  theme(
    legend.position = "top",
    axis.line = element_line(colour = "darkblue"),
    panel.grid.major.y = element_line(linetype = "dashed"),
    axis.text.x = element_blank())+
  labs(
    #title = "A Box Plot",
    subtitle = "Length of Diamonds",
    x = "Diamond Length in µm",
    y = "The Type of Cut",
    caption = "Diamonds data from Tidyverse"
  )


b <- clean_diamonds_data  %>% 
  ggplot(mapping = aes(x = the_width))+
  coord_flip() + 
  geom_boxplot(colour = "black", fill = "chartreuse2", alpha = 0.7)+
  scale_fill_brewer(palette="Accent")+
  theme(
    legend.position = "top",
    axis.line = element_line(colour = "darkblue"),
    panel.grid.major.y = element_line(linetype = "dashed"),
    axis.text.x = element_blank())+
  labs(
    # title = "A Box Plot",
    subtitle = "Width of Diamonds",
    x = "Diamond Width in µm",
    y = "The Type of Cut",
    caption = "Diamonds data from Tidyverse"
  )

c <- clean_diamonds_data  %>% 
  ggplot(mapping = aes(x = the_depth))+
  coord_flip() + 
  geom_boxplot(colour = "black", fill = "cyan3",alpha = 0.7)+
  scale_fill_brewer(palette="Accent")+
  theme(
    legend.position = "top",
    axis.line = element_line(colour = "darkblue"),
    panel.grid.major.y = element_line(linetype = "dashed"),
    axis.text.x = element_blank())+
  labs(
    #title = "A Box Plot",
    subtitle = "Depth of Diamonds",
    x = "Diamond Depth in µm",
    y = "The Type of Cut",
    caption = "Diamonds data from Tidyverse"
  )

# ------ Show Collective plots 
a | b | c

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

# Note: summary(clean_diamonds_data$cut) # This has all outliers removed.
# (30.10.2023)

summary(diamonds$cut)
summary(diamonds$cut == 'Ideal')
(percentage_of_Ideal <- sum(diamonds$cut == 'Ideal')/length(diamonds$cut))

X <- as.data.frame(diamonds[,c(1,3:10)])
X <- X[,-(2:3)] # Remove Color and Clarity, prediction based on quantitative attributes
head(X)
y <- as.numeric(diamonds$cut == 'Ideal')
N <- as.numeric(dim(X)[1])
M <- as.numeric(dim(X)[2])
attributeNames <- colnames(X)

################################
# K-folds cross-validation
################################

# -------------------------------------------------
# Regularized Linear regression of the linear model
# -------------------------------------------------

# Cross-validation

# Create cross-validation partition for evaluation of performance of optimal model
K <- 3
KK <- 5 # nr. of Inner loops # Use 10-fold cross-validation to estimate optimal value of lambda
T <- 14 # nr. of tested lambda

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
test_error_in <- rep(NA, KK)
train_error_in <- rep(NA, KK)

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
    
    mu2 <- colMeans(X_train2[,1:M])
    sigma2 <- apply(X_train2[,1:M], 2, sd)
    
    X_train2[,1:M] <- scale(X_train2[,1:M], mu2, sigma2)
    X_test2[,1:M] <- scale(X_test2[,1:M], mu2, sigma2)
    
    CV2$TrainSize[kk] <- length(y_train2)
    CV2$TestSize[kk] <- length(y_test2)
    
    mdl <- glmnet(X_train2, y_train2, family = "binomial", alpha = 0,
                  lambda = lambda_tmp,intercept=T)
    
    for (t in 1:T) {
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
                lambda = lambda_opt[k], intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class",
                         s = lambda_opt[k])
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class",
                        s = lambda_opt[k])
  
  # evaluate training and test error performance for optimal selected value of lambda
  Error_train_rlr[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test_rlr[k] <- sum(y_test_est != y_test) / length(y_test)
  ###
  
  # Compute squared error without regularization

  mdl <- glmnet(X_train, y_train, family = "binomial", alpha = 0,
                lambda = 1e-10, intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class", s = 1e-10)
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class", s = 1e-10)
  
  Error_train[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test[k] <- sum(y_test_est != y_test) / length(y_test)
}

# EXPLANATION: Classification errors of models with and without regularization are the same
# because the dataset has a large number of observations but the model has only 8 parameters
# to be estimated. Regularization improves the model particularly when it is overfitted, and
# it happens when observations are few wrt parameters to be estimated

# -----------------------------------------------------
# Regularized Linear regression of the non-linear model
# -----------------------------------------------------

# Manipulation of the data-set: insert some non-linear transformations of the attributes
X <- as.data.frame(diamonds[,c(1,5:10)])
X <- transform(X, 
               carat2 = carat^2,
               depth2 = depth^2,
               table2 = table^2,
               price2 = price^2,
               x2 = x^2,
               y2 = y^2,
               z2 = z^2,
               carat3 = carat^3,
               depth3 = depth^3,
               table3 = table^3,
               price3 = price^3,
               x3 = x^3,
               y3 = y^3,
               z3 = z^3,
               carat_depth = carat*depth,
               carat_table = carat*table,
               carat_price = carat*price,
               carat_x = carat*x,
               carat_y = carat*y,
               carat_z = carat*z,
               depth_table = depth*table,
               depth_price = depth*price,
               depth_x = depth*x,
               depth_y = depth*y,
               depth_z = depth*z,
               table_price = table*price,
               table_x = table*x,
               table_y = table*y,
               table_z = table*z,
               price_x = price*x,
               price_y = price*y,
               price_z = price*z,
               x_y = x*y,
               x_z = x*z,
               y_z = y*z
)

head(X)

N <- as.numeric(dim(X)[1])
M <- as.numeric(dim(X)[2])
attributeNames <- colnames(X)

# Cross-validation

# Create cross-validation partition for evaluation of performance of optimal model
K <- 2
KK <- 3 # nr. of Inner loops # Use 10-fold cross-validation to estimate optimal value of lambda

# Values of lambda
lambda_tmp <- 10^(-5:-3)
T <- length(lambda_tmp) # nr. of tested lambda

# Set seed for reproducibility
set.seed(4321)

CV <- list()
CV$which <- createFolds(y, k = K, list = F)

# Set up vectors that will store sizes of training and test sizes
CV$TrainSize <- c()
CV$TestSize <- c()

# Initialize variables
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
test_error_in <- rep(NA, KK)
train_error_in <- rep(NA, KK)

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
    
    mu2 <- colMeans(X_train2[,1:M])
    sigma2 <- apply(X_train2[,1:M], 2, sd)
    
    X_train2[,1:M] <- scale(X_train2[,1:M], mu2, sigma2)
    X_test2[,1:M] <- scale(X_test2[,1:M], mu2, sigma2)
    
    CV2$TrainSize[kk] <- length(y_train2)
    CV2$TestSize[kk] <- length(y_test2)
    
    mdl <- glmnet(X_train2, y_train2, family = "binomial", alpha = 0,
                  lambda = lambda_tmp,intercept=T)
    
    for (t in 1:T) {
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
                lambda = lambda_opt[k], intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class",
                         s = lambda_opt[k])
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class",
                        s = lambda_opt[k])
  
  # evaluate training and test error performance for optimal selected value of lambda
  Error_train_rlr[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test_rlr[k] <- sum(y_test_est != y_test) / length(y_test)
  ###
  
  # Compute squared error without regularization
  
  mdl <- glmnet(X_train, y_train, family = "binomial", alpha = 0,
                lambda = 1e-10, intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class", s = 1e-10)
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class", s = 1e-10)
  
  Error_train[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test[k] <- sum(y_test_est != y_test) / length(y_test)
}
