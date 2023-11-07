#rm(list=ls())
#dev.off()

# setwd("C:/Users/Alessandro/Desktop/PIETRO/Università/3_Machine Learning and Data Mining/Exercises/02450Toolbox_R")

#---------------------
# Access the Libraries
#---------------------
library(tidyverse)
library(magrittr)
library(forcats)
library(patchwork)
library(ggplot2)
library(glmnet)
library(caret) # Package for Cross-Validation
library(rpart) # Package for decision tree
library(knitr)
library(kableExtra)
#source("setup.R") # Contained in the R_toolbox

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
# Load the Data 
#---------------------
data(diamonds)
?diamonds
  
#---------------------
# See the Data 
#---------------------
diamonds

#---------------------
# Transform the Data
#---------------------
# - price Conversion from USD($) to DKK, Euro & Pound Sterling(£)

diamonds_price <- diamonds %>%  
  mutate(
    priceDKK = price * 7.066874,
    priceEuro = round((price * 0.949312),1),
    pricePS = round((price * 0.824538), 1))

#- Remove the USD($) variable. 
diamonds_price <- diamonds_price %>% 
  select(-price)


#diamonds_price

# - carat conversion to milligrams in weight
diam_conv_weight <- diamonds_price %>% 
  mutate(carat_mg = carat * 200) 

# - Remove the carat variable. 
diam_milligram <- diam_conv_weight %>% 
  select(-carat)

#diam_milligram

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
#new_diamonds_data

# new_diamonds_data %>% filter (the_length > 9285)
# new_diamonds_data %>% filter (the_length < 1965)
# 
# new_diamonds_data %>% filter (the_width > 9270)
# new_diamonds_data %>% filter (the_width < 1990)
# 
# new_diamonds_data %>% filter (the_depth > 5735)
# new_diamonds_data %>% filter (the_depth < 1215)

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

#---------------------
# Q1
#---------------------

# Aim: predict the price based on (some) other attributes. 

clear_diamonds_data <- clean_diamonds_data %>% 
  select(-c(cut,color,clarity,priceDKK, priceEuro, pricePS))


clear_diamonds_data

# Create a mean of 0 and sd of 1. 
scale_diamonds_data <- scale(clear_diamonds_data, 
                             center = TRUE, 
                             scale = TRUE) %>% 
  as_tibble()


# Assuming you have two data frames: scaled_diamonds_data and clean_diamonds_data

# Combine the columns from both data frames
scaled_diamonds_data <- cbind(Price = clean_diamonds_data$pricePS, scale_diamonds_data)
scaled_diamonds_data
#---------------------
# Training the Model
#---------------------

# Percentage of data to allocate for training (e.g., 80%)
train_percent <- 0.8

# make the index vector for splitting the data 
train_indices <- caret::createDataPartition(scaled_diamonds_data$Price,
                                            p = train_percent, 
                                            list = FALSE)

# Create training and holdout (test) sets
train_data <- scaled_diamonds_data[train_indices, ]
test_data <- scaled_diamonds_data[-train_indices, ]
train_data
test_data

# ----------------------------------------------------------------
#                       Split Training and Test Data
# ----------------------------------------------------------------


# The Train Data Split
X_train <- train_data %>% 
  select(-Price)

head(X_train)

y_train <- train_data %>% 
  select(Price)

head(y_train)

# The Test Data - split

X_test <- test_data %>% 
  select(-Price)
head(X_test)


y_test <- test_data %>% 
  select(Price)

head(y_test)

# Fit a model - how does it look on the ttrainign data 

lm(y)


 
# ----------------------------------------------------------------
#                      The Optimal Lambda
# ----------------------------------------------------------------

# Create a matrix from your training data
# (because it doesn't work with tibble or data frame)
X_train_matrix <- as.matrix(X_train)
head(X_train_matrix)

y_train_matrix <- as.matrix(y_train)
head(y_train_matrix)


# Make a sequence of lambdas. 
lambda_seq <- 10^(-5:10)
lambda_seq

# Fit ridge regression with cross-validation to find optimal lambda
ridge_model <- cv.glmnet(X_train_matrix, y_train_matrix, 
                         alpha = 0, 
                         lambda = lambda_seq, 
                         nfolds = 10, 
)  # alpha = 0 for ridge

# Print the cross-validation error
cat("The cross-validation error:")
print(ridge_model$cvm)

optimal_lambda <- ridge_model$lambda.min
cat("The optimal lambda is:", optimal_lambda)

# ----------------------------------------------------------------
#                        Visualisation
# ----------------------------------------------------------------

frame <- data.frame(broom::tidy(ridge_model))
frame


ggplot(data = frame, 
       mapping = aes(x = lambda, y = estimate, col = estimate))+
  geom_point(mapping = aes(x = log(lambda)))+
  geom_line(mapping = aes(x = log(lambda)))+
  labs(
    title = "K-Fold Cross Validation (K = 10)",
    subtitle = "Generalisation Error for different values of Lambda",
    x = "Log \U0003bb",
    y = "Mean Squared Error(estimate)",
    family = "Avenir",
    caption = "Mean Square Error | Cross-Validation Errors
    for different values of Lambda"
  )+
  theme_minimal()+
  theme(
    panel.border = element_rect(colour = "darkblue", fill=NA, size=1)
  )

# ----------------------------------------------------------------
#                        Predictions 
# ----------------------------------------------------------------

# Fit the model with the optimal lambda
L2_Regularisation_Ridge_Regression_model <- glmnet(X_train_matrix, y_train_matrix, alpha = 0, lambda = optimal_lambda)
L2_Regularisation_Ridge_Regression_model

broom::glance(L2_Regularisation_Ridge_Regression_model)
broom::tidy(L2_Regularisation_Ridge_Regression_model)

# Prepare the test data
X_test_matrix <- as.matrix(X_test)
y_test_matrix <- as.matrix(y_test)

# Use the fitted model to make predictions on the test data
predictions <- predict(L2_Regularisation_Ridge_Regression_model, 
                       s = optimal_lambda, 
                       newx = X_test_matrix)

# predictions are also known as *yhat
predictions


# Calculate RMS Error on the test data
rmse <- sqrt(mean((predictions - y_test_matrix)^2))
cat("RMSE on the test data:", rmse, "\n")

# mean of the actual target values
y_mean <- mean(y_test_matrix)
y_mean

# Calculate the total sum of squares (TSS)
tss <- sum((y_test_matrix - y_mean)^2)
tss

# Calculate the residual sum of squares (RSS)
rss <- sum((y_test_matrix - predictions)^2)
rss

# Calculate the R-squared (coefficient of determination)
rsquared <- 1 - (rss / tss)
cat("R-squared (Coefficient of Determination):", rsquared, "\n")

################################
# -------  Question 2 ---------
################################
# * In the report. 


################################
# -------  Question 3 ---------
################################


################################
# -------  Question 4 ---------
################################



################################
# -------  Question 5 ---------
################################



################################################################################
# -----  Classification  -------------------------------------------------------
################################################################################

rm(diam_conv_weight,diam_milligram,diamonds_conv,diamonds_price,new_diamonds_data)

summary(clean_diamonds_data$cut) # This has all outliers removed.
head(clean_diamonds_data)
# (30.10.2023)

summary(clean_diamonds_data$cut == 'Ideal')
(percentage_of_Ideal <- sum(clean_diamonds_data$cut == 'Ideal')/length(diamonds$cut))

y <- as.numeric(clean_diamonds_data$cut == 'Ideal')
X <- as.data.frame(clean_diamonds_data[,c(4:6,9:12)])

# CHOOSE AN OPTION AMONG 1, 2 AND 3: ###########

# 1) Linear model
attributeNames <- colnames(X)

# Classification tree
classNames <- c("Non-Ideal","Ideal")
classassignments <- classNames[y + 1]
(fmla <- as.formula(paste("y ~ ", paste(attributeNames, collapse = "+"))))

# END OF THE CHOICE #############################

head(X)

N <- as.numeric(dim(X)[1])
M <- as.numeric(dim(X)[2])

###################################
# -------  Question 2 & 3 ---------
###################################

# K-folds cross-validation ######################

# Create cross-validation partition for evaluation of performance of optimal model
K <- 4
KK <- 6 # nr. of Inner loops # Use 10-fold cross-validation to estimate optimal value of lambda

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
Error_train_rlr <- rep(1, K) # Rate error of the regularized logistic regression
Error_test_rlr <- rep(1, K)
Error_train <- rep(1, K) # Rate error of the non-regularized logistic regression
Error_test <- rep(1, K)
Error_train_nofeatures <- rep(1, K) # Rate error of the baseline
Error_test_nofeatures <- rep(1, K)
Error_train_tree <- rep(1, K) # Rate error of the classification tree
Error_test_tree <- rep(1, K)

for (k in 1:K) {
  print(paste("Crossvalidation fold ", k, "/", K, sep = ""))
  
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
  y_train_est <- as.numeric(predict(tree, newdata=X_train)[,2]>0.5)
  y_test_est <- as.numeric(predict(tree, newdata=X_test)[,2]>0.5)
  
  Error_train_tree[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test_tree[k] <- sum(y_test_est != y_test) / length(y_test)
  
  if (k == 1) mdl_tree <- tree
  if (k != 1){
    if (Error_test_tree[k] == min(Error_test_tree)){
      mdl_tree <- tree
      print("tree")
    }
  }
  
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
  
  if (k == 1) mdl_lr <- mdl
  if (k != 1){
    if (Error_test_rlr[k] == min(Error_test_rlr)){
      mdl_lr <- mdl
      y_lr <- y_test_est
      print("lr")
    }
  }
  ### end of regularized logistic regression ###
  
  # Compute squared error without regularization

  mdl <- glmnet(X_train, y_train, family = "binomial", alpha = 0,
                lambda = 1e-10, intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class", s = 1e-10)
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class", s = 1e-10)
  
  Error_train[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test[k] <- sum(y_test_est != y_test) / length(y_test)
  
  Error_train_nofeatures[k] <- sum(y_train != 0) / length(y_train)
  Error_test_nofeatures[k] <- sum(y_test != 0) / length(y_test)
}

(Results <- as.data.frame(matrix(c(1:K,
                                   round(Error_test_nofeatures*100,digits = 2),
                                   lambda_opt,
                                   round(Error_test_rlr*100,digits=2),
                                   cp_opt,
                                   round(Error_test_tree*100,digits=2)),
                                 nrow=K,byrow=F)))
ind_lr <- 2
ind_tree <- 1

# EXPLANATION OF (1): Classification errors of models with and without regularization are the same
# because the dataset has a large number of observations but the model has only 8 parameters
# to be estimated. Regularization improves the model particularly when it is overfitted, and
# it happens when observations are few wrt parameters to be estimated

plot(log(lambda_tmp), log(apply(Error_train2, 1, sum) / sum(CV2$TrainSize)),
     xlab = "log(lambda)", ylab = "log(Error)",
     main = paste0("Optimal lambda: 1e", log10(lambda_opt[k]))
)

lines(log(lambda_tmp), log(apply(Error_train2, 1, sum) / sum(CV2$TrainSize)))
plot(log(lambda_tmp), log(apply(Error_test2, 1, sum) / sum(CV2$TestSize)), col = "red")
lines(log(lambda_tmp), log(apply(Error_test2, 1, sum) / sum(CV2$TestSize)), col = "red")

legend("bottomright", legend = c("Training", "Test"), col = c("black", "red"), lty = 1)

#######################################################################
# QUADRATIC MODEL APPLIED TO THE PRINCIPAL COMPONENTS #################
#######################################################################

y <- as.numeric(clean_diamonds_data$cut == 'Ideal')
X <- as.data.frame(clean_diamonds_data[,c(4:6,9:12)])

# Quadratic model applied to data projected onto the first four principal components
stds <- apply(X, 2, sd)
X <- t(apply(X, 1, "-", colMeans(X)))
X <- t(apply(X, 1, "*", 1 / stds))
X <- as.data.frame(X)
S <- svd(X)
X <- as.data.frame(S$u %*% diag(S$d))
X <- X[,1:4]
X <- transform(X,
               V1_2 = V1^2,
               V2_2 = V2^2,
               V3_2 = V3^2,
               V4_2 = V4^2,
               V1_V2 = V1*V2,
               V1_V3 = V1*V3,
               V1_V4 = V1*V4,
               V2_V3 = V2*V3,
               V2_V4 = V2*V4,
               V3_V4 = V3*V4
)
rm(S)

head(X)

N <- as.numeric(dim(X)[1])
M <- as.numeric(dim(X)[2])

# K-folds cross-validation based on the number of folds and tempted lambdas of before

CV <- list()
CV$which <- createFolds(y, k = K, list = F)

# Set up vectors that will store sizes of training and test sizes
CV$TrainSize <- c()
CV$TestSize <- c()

# Initialize variables
Error_train2 <- matrix(rep(NA, times = T * KK), nrow = T)
Error_test2 <- matrix(rep(NA, times = T * KK), nrow = T)
lambda_opt_PCA <- rep(NA, K)
mu_PCA <- matrix(rep(NA, times = M * K), nrow = K)
sigma_PCA <- matrix(rep(NA, times = M * K), nrow = K)
Error_train_PCA <- rep(1, K) # Rate error of the regularized logistic regression
Error_test_PCA <- rep(1, K)

for (k in 1:K) {
  print(paste("Crossvalidation fold ", k, "/", K, sep = ""))
  
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
  lambda_opt_PCA[k] <- lambda_tmp[ind_opt]
  
  # Standardize outer fold based on training set, and save the mean and standard
  # deviations since they're part of the model (they would be needed for
  # making new predictions) - for brevity we won't always store these in the scripts
  mu_PCA[k, ] <- colMeans(X_train[,1:M])
  sigma_PCA[k, ] <- apply(X_train[,1:M], 2, sd)
  
  X_train[,1:M] <- scale(X_train[,1:M], mu_PCA[k, ], sigma_PCA[k, ])
  X_test[,1:M] <- scale(X_test[,1:M], mu_PCA[k, ], sigma_PCA[k, ])
  
  ### regularized logistic regression ###
  # Estimate w for the optimal value of lambda
  mdl <- glmnet(X_train, y_train, family = "binomial", alpha = 0,
                lambda = lambda_opt_PCA[k], intercept=T)
  
  y_train_est <- predict(mdl, newx=as.matrix(X_train), type = "class",
                         s = lambda_opt_PCA[k])
  y_test_est <- predict(mdl, newx=as.matrix(X_test), type = "class",
                        s = lambda_opt_PCA[k])
  
  print(length(y_test_est))
  
  # evaluate training and test error performance for optimal selected value of lambda
  Error_train_PCA[k] <- sum(y_train_est != y_train) / length(y_train)
  Error_test_PCA[k] <- sum(y_test_est != y_test) / length(y_test)
  
  if (k == 1) mdl_PCA <- mdl
  if (k != 1) {
    if (Error_test_PCA[k] == min(Error_test_PCA)){
      mdl_PCA <- mdl
      print("PCA")
    }
  }
  ### end of regularized logistic regression ###
}

(Results <- as.data.frame(matrix(c(1:K,
                                   round(Error_test_nofeatures*100,digits = 2),
                                   lambda_opt,
                                   round(Error_test_rlr*100,digits=2),
                                   lambda_opt_PCA,
                                   round(Error_test_PCA*100,digits=2),
                                   cp_opt,
                                   round(Error_test_tree*100,digits=2)),
                                 nrow=K,byrow=F)))
# Consider the unbalances of the dataset

# resultNames1 <- c("i","$E_i^{test}$ [%]","$olambda_i$","$E_i^{test}$ [%]",
#                   "$olambda_i$","$E_i^{test}$ [%]","$c_{P,i}^{*}$","$E_i^{test}$ [%]")
kable(Results, format = "latex", booktabs = T) %>%
  add_header_above(c("Outer fold" = 1, "Base-Line" = 1, "Logistic Regression (Linear)" = 2, "Logistic Regression (Quadratic)" = 2, "Decision Tree" = 2))
ind_PCA <- 1
# Page 216 of the book: why we cannot average the errors
# Use y_lr, y_tree and y_PCA, with y_base = 0, to compute statistics

################################
# -------  Question 4 ---------
################################

# Apply each model to the entire dataset:

y_base <- rep(0,N)

X <- as.data.frame(clean_diamonds_data[,c(4:6,9:12)])
X <- as.data.frame(scale(X, mu[ind_lr, ], sigma[ind_lr, ]))
y_lr <- as.numeric(predict(mdl_lr, newx=as.matrix(X), type = "class", s = lambda_opt[ind_lr]))

X <- as.data.frame(clean_diamonds_data[,c(4:6,9:12)])
X <- as.data.frame(scale(X, mu[ind_tree, ], sigma[ind_tree, ]))
y_tree <- as.numeric(predict(mdl_tree, newdata=X)[,2]>0.5)

par(mfrow=c(1,1), xpd = NA)
plot(mdl_tree)
text(mdl_tree)

X <- as.data.frame(clean_diamonds_data[,c(4:6,9:12)])
stds <- apply(X, 2, sd)
X <- t(apply(X, 1, "-", colMeans(X)))
X <- t(apply(X, 1, "*", 1 / stds))
X <- as.data.frame(X)
S <- svd(X)
X <- as.data.frame(S$u %*% diag(S$d))
X <- X[,1:4]
X <- transform(X,
               V1_2 = V1^2,
               V2_2 = V2^2,
               V3_2 = V3^2,
               V4_2 = V4^2,
               V1_V2 = V1*V2,
               V1_V3 = V1*V3,
               V1_V4 = V1*V4,
               V2_V3 = V2*V3,
               V2_V4 = V2*V4,
               V3_V4 = V3*V4
)
rm(S)
X <- as.data.frame(scale(X, mu_PCA[ind_PCA, ], sigma_PCA[ind_PCA, ]))
y_PCA <- as.numeric(predict(mdl_PCA, newx=as.matrix(X), type = "class", s = 1e-05))

# Setup I: McNemar test

alpha <- 0.05
rt_bl <- mcnemar(y, y_base, y_lr, alpha = alpha)
rt_bt <- mcnemar(y, y_base, y_tree, alpha = alpha)
rt_bp <- mcnemar(y, y_base, y_PCA, alpha = alpha)
rt_lt <- mcnemar(y, y_lr, y_tree, alpha = alpha)
rt_lp <- mcnemar(y, y_lr, y_PCA, alpha = alpha)
rt_tp <- mcnemar(y, y_tree, y_PCA, alpha = alpha)

# p-value is always 0 because the data-set is very large and p-value formula
# has a N at the denominator. It seems that the tree and the PCA are the
# closest models

################################
# -------  Question 5 ---------
################################

mdl_lr$a0
mdl_lr$beta

# table and length have a larger weight wrt the other attributes

################################################################
# APPENDIX
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
