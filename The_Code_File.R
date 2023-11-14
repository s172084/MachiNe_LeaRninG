#rm(list=ls())
#dev.off()

# setwd("C:/Users/Alessandro/Desktop/PIETRO/Università/3_Machine Learning and Data Mining/Exercises/02450Toolbox_R")


library(tidyverse)
library(magrittr)
library(forcats)
library(patchwork)
library(ggplot2)
library(glmnet)
library(knitr)
library(kableExtra)
source("setup.R")      # Contained in the R_toolbox
library(rpart)         # Package for Decision trees
library(caret)         # Package for Cross-Validation
library(tidyr)
library(neuralnet)      #Package for Neural Networks. 
library(broom)
#---------------------
# Functions
#---------------------
source("/Users/m.o.l.s/Desktop/Machine Learning /7. All Toolboxes/02450Toolbox_R/Tools/is.scalar.R")
source("/Users/m.o.l.s/Desktop/Machine Learning /7. All Toolboxes/02450Toolbox_R/Tools/forwardSelection.R")
source("/Users/m.o.l.s/Desktop/Machine Learning /7. All Toolboxes/02450Toolbox_R/Tools/bmplot.R")
source("/Users/m.o.l.s/Desktop/Machine Learning /7. All Toolboxes/02450Toolbox_R/Tools/train_neural_network.R")
source("/Users/m.o.l.s/Desktop/Machine Learning /7. All Toolboxes/02450Toolbox_R/Tools/plot.nnet.R") #doesn't work
source("/Users/m.o.l.s/Desktop/Machine Learning /7. All Toolboxes/02450Toolbox_R/Tools/statistics.R")

SLM_make_linear_model <- function(x, y) {
  plot(x,y, 
       main = "Linear Model",family = "Avenir", 
       lwd = 2,
       # change the shape of the points
       pch = 8,
       # change the size of the points
       cex = 1.5, 
       # change the colour of the points
       col = viridisLite::viridis(8, alpha = 0.9), 
  )
  # add a line of best fit.
  abline(lm(y ~ x), col="green", lwd = 1)
  
  D <- data.frame(x = x, y = y)
  m <- lm(y ~ x, data = D)
  
  n <- length(x)
  
  h <- data.frame(broom::glance(m))
  cat("Coefficient of Determination:(r.squared)", h$r.squared,"\n")
  cat("The proportion of the explained variation:", h$adj.r.squared,"\n")
  cat("Sigma:", h$sigma,"\n")
  cat("number of observations:",h$nobs,"\n")
  cat("The correlation is:",cor(x,y),"\n\n")
  cat("The variance of the slope:", vcov(m), "\n")
  
  if (h$p.value < 0.05){
    cat("Since the confidence interval does not include 0 and \n
    The p-value is less than 0.05 which gives 
    strong evidence against the null hypothesis so
        there is a relationship between y and x\n")
  }
  
  g <- summary(m)
  return(g)
}


SLM_calculate_a_prediction <- function(y, x, new_observation) {
  # predict a single new value from a change in one observation. 
  # fit simple linear regression model
  # define new observation
  # use the fitted model to predict the value for the new observation
  
  f <- data.frame(x= x,
                  y= y)
  
  model <- lm(y ~ x, data = f)
  new <- data.frame(x=c(new_observation))
  cat("The model predicts that this diamond will have a (£)price of:\n")
  predict(model, newdata = new, interval = 'confidence', level =  0.95)
  # can also do this for a prediction interval. 
}

# ------------------------------------
# Linear regression criterion function 
# For 2(a)
# ------------------------------------
#  This function takes as input a training and a test set.
#  1. It fits a linear model on the training set using lm.
#  2. It estimates the output of the test set using predict.
#  3. It computes the sum of squared errors.

funLinreg <- function(X_train, y_train, X_test, y_test) {
  
  X_train <- data.frame(X_train)
  X_test <- data.frame(X_test)
  
  xnam <- paste("X", 1:dim(X_train)[2], sep = "")
  colnames(X_train) <- xnam
  colnames(X_test) <- xnam
  (fmla <- as.formula(paste("y_train ~ ", paste(xnam, collapse = "+"))))
  
  mod <- lm(fmla, data = X_train)
  preds <- predict(mod, newdata = X_test)
  return(sum((y_test - preds)^2))
}

calculate_jeffrey_interval <- function(y_true, yhat) {
  # Calculate the jeffrey interval
  # Using the test data ys and predictions 
  rt <- jeffrey_interval(y_true, yhat[, 1])
  rt
  
  alpha <- 0.05
  
  # Theta Hat 
  thetahatA <- rt$thetahat
  
  # The confidence interval 
  CI <- rt$CI
  
  print(paste("Theta point estimate is :", round(thetahatA, 3)))
  print(paste("Confidence Interval is: "))
  print(CI)
}

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
?#diamonds
  
  #---------------------
# See the Data 
#---------------------
diamonds
colnames(diamonds)

#---------------------
# Transform the Data
#---------------------
# - price Conversion from USD($) to DKK, Euro & Pound Sterling(£)

diamonds_price <- diamonds %>%  
  mutate(
    priceDKK = price * 7.066874,
    priceEuro = round((price * 0.949312),1),
    pricePS = round((price * 0.824538), 1))

diamonds_price

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

boxplot(new_diamonds_data, 
        main = "All Attributes of Diamonds",
        family = "Avenir")$out

boxplot(data.frame(x= new_diamonds_data$the_length,
                   y= new_diamonds_data$the_width,
                   z= new_diamonds_data$the_depth),
        col = viridisLite::viridis(3), 
        main = "x , y and z Attributes of Diamonds", 
        family = "Avenir")
#---------------------

determine_outliers(new_diamonds_data$the_length)
# quantile(new_diamonds_data$the_length, probs=c(.25, .75), na.rm = FALSE)
# IQR(new_diamonds_data$the_length)
#---------------------

determine_outliers(new_diamonds_data$the_width)
# quantile(new_diamonds_data$the_width, probs=c(.25, .75), na.rm = FALSE)
# IQR(new_diamonds_data$the_width)

#---------------------
determine_outliers(new_diamonds_data$the_depth)
# quantile(new_diamonds_data$the_depth, probs=c(.25, .75), na.rm = FALSE)
# IQR(new_diamonds_data$the_depth)

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

#---------------------
# Data Visualisation
#---------------------

boxplot(clean_diamonds_data)$out # no outliers. 

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
    subtitle = " Length of Diamonds, based on cut type",
    x = "Diamond Length in µm",
    y = "The Type of Cut",
    caption = "Diamonds data from Tidyverse"
  )

#---------------------
# Q1
#---------------------

# Aim: Predict the price of Premium diamonds
# with colour of D
filt_diamonds_data <- clean_diamonds_data %>% 
  filter(cut == "Premium") %>% 
  filter(color == "D")

filt_diamonds_data

#---------------------
# The Data Set  -->>>>
#---------------------
filt_diamonds_data

#---------------------
# The Data Set <<<---
#---------------------


# Remove the categories (and some others.. )
clear_diamonds_data <- filt_diamonds_data %>% 
  select(-c(cut,color,clarity,priceDKK, priceEuro, pricePS))
clear_diamonds_data

# --------------------------------------------------
# Scale : A mean of 0 and standard deviation of 1
# --------------------------------------------------
# The categorical ones cant be scaled, just X, and not y
scale_diamonds_data <- scale(clear_diamonds_data, 
                             center = TRUE, 
                             scale = TRUE) %>% 
  as_tibble()


scale_diamonds_data

# Assuming you have two data frames: scaled_diamonds_data 
# Combine the columns from both data frames (£ Sterling Price of Diamonds)
scaled_diamonds_data <- cbind(Price = filt_diamonds_data$pricePS,
                              scale_diamonds_data)
head(scaled_diamonds_data)


# --------------------------------------------------
# Question 1(a)
# --------------------------------------------------
# linear regression


# non-scaled : carat(x) versus price (y)
SLM_make_linear_model(x = filt_diamonds_data$carat_mg,
                      y = filt_diamonds_data$pricePS)

# scaled
SLM_make_linear_model(x = scaled_diamonds_data$carat_mg,
                      y = scaled_diamonds_data$Price)

# predict the price, based on the carat of 142. 
SLM_calculate_a_prediction(y = filt_diamonds_data$pricePS,
                           x = filt_diamonds_data$carat_mg,
                           new_observation = 142)
head(scaled_diamonds_data)

# Multiple Linear Regression 
# --* Parameter Estimation *----
multiple_linear_model <- lm(Price ~ depth +
                              table + 
                              carat_mg + 
                              the_length + 
                              the_width + 
                              the_depth, 
                            data = scaled_diamonds_data)

broom::tidy(multiple_linear_model)



# Prediction of a new diamond price, based on multiple linear regression
new_diamond <- data.frame(depth = 40.5, 
                          table =  60, 
                          carat_mg = 200, 
                          the_length = 200,
                          the_width = 300,
                          the_depth = 200)

new_diamond

## Calculate confidence and prediction intervals based on the model. 
CI <- predict(multiple_linear_model, 
              newdata = new_diamond, 
              interval="confidence", 
              level=0.95)
CI

PI <- predict(multiple_linear_model,
              newdata = new_diamond,
              interval="prediction",
              level=0.95)
PI
# --------------------------------------------------
# Question 2(a)
# --------------------------------------------------
filt_diamonds_data
clear_diamonds_data
head(scaled_diamonds_data)

#------------------------------------------------------------------
# One Hot Encoded : Multicollinearity...
#------------------------------------------------------------------
# Dummy Variables = M-1 (subtract cut)

# filt_diamonds_data <-one_hot(filt_diamonds_data)
# filt_diamonds_data

# One-Hot Encoded Diamonds Data
# hot_diamonds_data <- filt_diamonds_data %>% 
#   
#   # One-Hot Encode Categorical Variables (not necessary)
#   mutate(cut = as.numeric(as.factor(cut)) - 1 ,
#     color = as.numeric(as.factor(color)) - 1 , 
#     clarity = as.numeric(as.factor(clarity)) - 1) %>% 
#   
#   # Remove price in euros and DKK for now. 
#   select(-c(priceDKK,priceEuro))
#     
# 
# hot_diamonds_data 
# 
# # Put the price first. 
# hot_diamonds_data <- hot_diamonds_data %>% 
#   select(pricePS, everything()) %>% 
#      # removing the categorical variables (is it necessary?)
#    select(-c(cut, color, clarity))
# 
# head(hot_diamonds_data)

hot_diamonds_data <- scaled_diamonds_data

hot_diamonds_data <- hot_diamonds_data %>% 
  rename(pricePS = Price)

head(hot_diamonds_data)
# ----------------------------------------------------
# Training and Test Data Split. 
# ----------------------------------------------------

set.seed(123)

# Amount of data to allocate for training (e.g, 70%)
train_percent <- 0.7

# make the index vector for splitting the data 
train_indices <- caret::createDataPartition(hot_diamonds_data$pricePS,
                                            p = train_percent, 
                                            list = FALSE)

# Create training and holdout (test) sets
diamonds_train_data <- hot_diamonds_data[train_indices, ]
diamonds_test_data <- hot_diamonds_data[-train_indices, ]


dim(diamonds_train_data)
dim(diamonds_test_data)

# ----------------------------------------------------
# Training X y  = Sparkle. 
# ----------------------------------------------------

sparkle_X <-diamonds_train_data %>%
  select(-pricePS)%>% as.matrix()

sparkle_X


sparkle_y <- diamonds_train_data %>% 
  select(pricePS) %>% as.matrix()

sparkle_y

# Training Data is Sparkle.

# ----------------------------------------------------
# The data is already scaled... 
# sparkle_X <- diamonds_train_data %>% 
#   select(-pricePS) %>% 
#   scale(center = TRUE, scale = FALSE) %>% as.matrix()
# ----------------------------------------------------

# ----------------------------------------------------
# Test X y  = Glitter
# ----------------------------------------------------
glitter_X <- diamonds_test_data %>% 
  select(-pricePS) %>% as.matrix()

glitter_X

glitter_y <- diamonds_test_data %>% select(pricePS)  %>% as.matrix()
glitter_y

# -----------------------------------------------------------------------
#   
#     Perform 10 Fold Cross Validation to Select the optimal Lambda 
#     Ridge Regularisation. 
#
# -----------------------------------------------------------------------

# Make a sequence of lambdas. 
lambda_seq <- 10^seq(-1, 13, length.out = 100)
lambda_seq


set.seed(123)
# Ridge regression model with cross-validation to find optimal lambda
ridge_model <- cv.glmnet(sparkle_X, sparkle_y, 
                         alpha = 0,  # Ridge == 0
                         lambda = lambda_seq, 
                         standardize = TRUE,
                         nfolds = 10, 
                         type.measure = "mse") 

# ----------------------
# Error
# ----------------------

cat("The cross-validation Error:")
plot(ridge_model$cvm) 
plot(ridge_model)
mean(ridge_model$cvm) # Red line is the mean

# ----------------------
# Optimal Lambda
# ----------------------
cat("The optimal lambda is:")
optimal_lambda <- ridge_model$lambda.min
optimal_lambda 

# -----------------------------------------
# K = 10 Fold Cross Validation Results 
# -----------------------------------------

cvresults <- data.frame(broom::tidy(ridge_model)) %>% 
  arrange(lambda) %>% 
  head(n = 10)

# ----------------------
# Results
# ----------------------
cvresults



fig1 <- ggplot(data = cvresults, 
               mapping = aes(x = lambda,y = estimate,
                             col = estimate))+
  geom_point(mapping = aes(x = log(lambda)), size = 2)+
  geom_line(mapping = aes(x = log(lambda)))+
  labs(
    title = "K-Fold Cross Validation (K = 10)",
    subtitle = "Generalisation Error for Different values of Lambda",
    x = "Log \U0003bb",
    y = "Mean Squared Error",
    family = "Avenir",
    caption = "Mean Square Error | Cross-Validation Errors
    for different values of Lambda"
  )+
  theme_minimal()

fig1 


ggsave("Images/figure1.png",
       fig1,
       width = 5,
       height = 5)


plot.new()

# --------------------------------------------------
# Question 3(a)
# --------------------------------------------------

# --------------------------------------------------
#   
#     Regularised Linear Model : Ridge
#
# --------------------------------------------------

# Fit final model using the training data. 
# Get its sum of squared residuals and multiple R-squared
final_ridge_model <- glmnet(sparkle_X, sparkle_y, 
                            alpha = 0,
                            lambda = optimal_lambda,
                            standardize = TRUE)

final_ridge_model

broom::tidy(final_ridge_model)
broom::glance(final_ridge_model)

# --------------------------------------------------
# Prepare the test data. 
# Use the fitted model to make predictions on the test data
diamonds_yhat <- predict(final_ridge_model, 
                         s = optimal_lambda, 
                         newx = glitter_X)

# predictions are also known as yhat
diamonds_yhat
plot(diamonds_yhat)



# Scatter plot of actual price against estimated price.
ggplot(
  mapping=aes(x = glitter_y ,
              y = diamonds_yhat)) +
  geom_point(alpha = 0.7,) +
  geom_smooth(method = "glm")+
  labs(
    x = "The Actual Price (£)",
    y = "The estimated Price (£)",
    title = "Price of Diamonds",
    subtitle = "Currrency : £ Sterling"
  ) +
  theme_minimal() +
  theme(text = element_text(family = "Avenir"))

# --------------------------------------------------
# Multiple R squared is: 
# Calculate the residuals using the test data and predicted data. 
residuals <- glitter_y - diamonds_yhat

# Histogram of actual price - estimated price.
hist(glitter_y - diamonds_yhat, breaks = 41,
     main = "The Residual Error", 
     col = "darkgreen", 
     family = "Avenir")

# Calculate the training error (MSE)
mse <- mean(residuals^2)
cat("The MSE is: ", mse)

# Calculate the RSS (sum of squared residuals)
rss <- sum(residuals^2)
cat("The sum of squared residuals is: ", rss)

# Multiple R-squared 
rsq <- cor(glitter_X, diamonds_yhat)^2 
rsq 
plot(rsq)

#RMSE (could have square rooted mse.. but.)
ModelMetrics::rmse(glitter_y, predicted = diamonds_yhat, final_ridge_model)

# Mean Absolute Error using the test data. 
ModelMetrics::mae(glitter_y, predicted = diamonds_yhat, final_ridge_model)

# Jeffreys Confidence interval: 
calculate_jeffrey_interval(glitter_y, diamonds_yhat)



# --------------------------------------------------
# Question 1(b)
# --------------------------------------------------
## Question 1 
# Compare 3 models:
#   - A baseline linear regression model with no features
#   - A regularised linear model 
#   - An Artificial Neural Network 
#   
#   - Use 2 level cross validation to compare the models
#   - with k1 = 10 folds
#    and  k2 = 10 folds 
#   
#   -Compare the mean of y on the training data
#   -Use this value to predict y on the test data
#   
#   -Is one model is better than the other ? 
#   
#   -Is the model better than the baseline ?
#
#   -Fit an Artificial Neural Network to the data and select a reasonable
#   -range of values for h
#   
#   -Describe the range of values you will use for h and lambda


# -----------------------------------------------------------------------
#   
#     Elastic Net Regression
#
# -----------------------------------------------------------------------

# X and Y datasets 
X <- sparkle_X
Y <- sparkle_y

# Model Building : Elastic Net Regression 
control <- trainControl(method = "repeatedcv", 
                        # Number of K Folds
                        number = 10, 
                        # Number of complete sets of folds to compute. 
                        repeats = 2, 
                        search = "random", 
                        verboseIter = FALSE) 

# Training ELastic Net Regression model 
elastic_model <- train(pricePS ~ ., 
                       data = cbind(X, Y), 
                       method = "glmnet", 
                       preProcess = c("center", "scale"), 
                       tuneLength = 25, 
                       trControl = control, 
                       verboseIter = FALSE) 

elasticated <- plot(elastic_model)
elasticated

# save image. 
jpeg("Images/elasticated_plot.jpeg", width = 800, 
     height = 600, units = "px", pointsize = 12)



# RESULTS>>
elastic_model$results

# Model Prediction 
x_hat_pre <- predict(elastic_model, glitter_X) 
x_hat_pre 

# Multiple R-squared on the test data and predictions. 
rsq <- cor(glitter_X, x_hat_pre)^2 
rsq 

#RMSE (could have square rooted mse.. but.)
ModelMetrics::rmse(glitter_y, predicted = x_hat_pre, elastic_model)

# Mean Absolute Error using the test data. 
ModelMetrics::mae(glitter_y, predicted = x_hat_pre, elastic_model)

ModelMetrics::mse(glitter_y, predicted = x_hat_pre, elastic_model)



# -----------------------------------------------------------------------
#   
#     Artificial Neural Network : NeuralNet
#
# -----------------------------------------------------------------------

# Preprocess the data
set.seed(123)

diamonds_train_data <- diamonds_train_data %>%  as.data.frame()  # Training Set
diamonds_test_data <- diamonds_train_data %>% as.data.frame()   # Test Set

library(neuralnet)
library(NeuralNetTools) # Visualisation of nn. 

# Define the formula for the neural network
formula <- pricePS ~ depth + table  + carat_mg + the_length + the_width +the_depth


?neuralnet
# Create and train the neural network
nn_diamonds_model <- neuralnet(formula,
                               data = diamonds_train_data, 
                               hidden = c(1), 
                               linear.output = TRUE)

print(nn_diamonds_model)
plot(nn_diamonds_model)
n <- plotnet(nn_diamonds_model)
n
diamonds_test_data

predictions <- predict(nn_diamonds_model, newdata = diamonds_test_data)
predictions

# Compare predicted and actual diamond price # [,1] means pricePS
comparison <- data.frame(Actual = diamonds_test_data$pricePS,
                         Predicted = predictions)
print(head(comparison))


# Calculate the test error
# Calculate Root Mean Squared Error (RMSE) for the test set
rmse <- sqrt(mean((predictions - diamonds_test_data$pricePS)^2))
cat("Root Mean Squared Error (RMSE) on the test set:", rmse, "\n")

mse <- mean((predictions - diamonds_test_data$pricePS)^2)
cat("Mean Squared Error (MSE) on the test set:", rmse, "\n")

# mean of the actual target values (ytest matrix)
y_mean <- mean(diamonds_test_data$pricePS)
y_mean

# Calculate the total sum of squares (TSS)
tss <- sum((diamonds_test_data$pricePS - y_mean)^2)
tss

# Calculate the residual sum of squares (RSS)
rss <- sum((diamonds_test_data$pricePS - predictions)^2)
rss

# Calculate the R-squared (coefficient of determination)
rsquared <- 1 - (rss / tss)
rsquared

#calculate_jeffrey_interval(diamonds_test_data$pricePS, predictions)

# -----------------------------------------------------------------------
#   
#     Baseline Model 
#
# -----------------------------------------------------------------------
# Baseline Model 
set.seed(123)

# Fit a baseline linear regression model with no features
baseline_model <- lm(sparkle_y ~ 1)  # '1' represents the intercept term


# Summarize the model
summary(baseline_model)
broom::tidy(baseline_model)

# Turn test data into a dataframe
glit <- glitter_X %>% 
  as_tibble()

# Model Prediction 
diamond_pre <- predict(baseline_model, glit) 
head(diamond_pre)

# Calculate the test error
# Calculate Root Mean Squared Error (RMSE) for the test set
rmse <- sqrt(mean((diamond_pre - glitter_y)^2))
cat("Root Mean Squared Error (RMSE) on the test set:", rmse, "\n")

rmse*rmse

# -----------------------------------------------------------------------
#   
#     Two Level Cross Validation with K = 10
#
# -----------------------------------------------------------------------

# The X data frame
# The y vector 

# Number of Rows (N) and Columns (M)
M <- dim(sparkle_X)[2]
N <- dim(sparkle_X)[1]

# Assign attribute names for the bmplot
attributeNames <- make.names(unlist(colnames(sparkle_X)))
attributeNames

sparkle_X
glitter_X

head(glitter_y)
head(sparkle_y)

# ****** DONT CHANGE******** 
# ------------------------------
#      Cross validation
# ------------------------------
# Number of folds for k-fold cross-validation
K<-10

# Create k-fold cross validation partition
set.seed(1234)
CV <- list()
CV$which <- caret::createFolds(sparkle_y,
                               k = K,
                               list = F)

# Set up vectors that will store sizes of training and test sizes
CV$TrainSize <- c()
CV$TestSize <- c()
# Initialize variables
Features <- matrix(rep(NA, times = K * M), nrow = K)
Error_train <- matrix(rep(NA, times = K), nrow = K)
Error_test <- matrix(rep(NA, times = K), nrow = K)
Error_train_fs <- matrix(rep(NA, times = K), nrow = K)
Error_test_fs <- matrix(rep(NA, times = K), nrow = K)

# For each cross-validation fold
for (k in 1:K) {
  
  print(paste("Cross Validation Fold:", k, "/", K, sep = ""))
  
  # Extract training and test set
  X_train <- sparkle_X[(CV$which != k), ]
  y_train <- sparkle_y[(CV$which != k)]
  
  
  X_test <- sparkle_X[(CV$which == k), ]
  y_test <- sparkle_y[(CV$which == k)]
  
  CV$TrainSize[k] <- length(y_train)
  CV$TestSize[k] <- length(y_test)
  
  # Use 10-fold cross validation for sequential feature selection
  fsres <- forwardSelection(funLinreg, 
                            sparkle_X, sparkle_y,
                            stoppingCrit = "minCostImprovement")
  
  
  # Extract selected features from the forward selection routing
  selected.features <- fsres$featsIncluded
  cat("selected features", selected.features, "\n")
  
  # Save the selected features
  Features[k, ] <- fsres$binaryFeatsIncluded
  
  cat("features", selected.features, "\n")
  
  # Compute squared error without feature subset selection
  Error_train[k] <- funLinreg(X_train, y_train, X_train, y_train)
  Error_test[k] <- funLinreg(X_train, y_train, X_test, y_test)
  
  # Compute squared error with feature subset selection
  Error_train_fs[k] <- funLinreg(X_train[, selected.features], y_train, X_train[, selected.features], y_train)
  Error_test_fs[k] <- funLinreg(X_train[, selected.features], y_train, X_test[, selected.features], y_test)
  
  # Show variable selection history
  cat("Feature selection", k)
  I <- length(fsres$costs) # Number of iterations
}


par(mfrow = c(1, 1))

# --------------------------------
# Error Criterion Plot (base)
# --------------------------------
plot(fsres$costs,
     col = "darkgreen", 
     pch = 1,
     xlab = "Number of Iterations", 
     ylab = "Squared error (crossvalidation)",
     main = "Value of error criterion", 
     family = "Avenir")


# --------------------------------
# Binary Matrix Plot
# --------------------------------
# Plot feature selection sequence
fig2 <-bmplot(attributeNames, 1:I,
              fsres$binaryFeatsIncludedMatrix)



# --------------------------------
# Error Criterion Plot (ggplot)
# --------------------------------
p <- data.frame(costs = fsres$costs,
                Iteration = seq(length(fsres$costs)))

p

p  %>% 
  ggplot(mapping = aes(seq(length(costs)), costs))+
  geom_point(colour = "cyan4",alpha = 0.7, size = 3)+
  geom_line(alpha = 0.7)+
  #scale_color_manual(values=c('Yellow','darkgreen'))+
  scale_fill_brewer(palette="Accent")+
  theme(
    legend.position = "bottom",
  )+
  labs(
    title = "Diamonds",
    subtitle = "Squared Error after Cross Validation",
    x = "Number of Iterations",
    y = "Costs: Squared Error",
    caption = "Diamonds data from ggplot2 Tidyverse"
  )+
  theme_minimal()




# Display results
print(paste("Linear regression without feature selection:"))
print(paste("Training error:", sum(Error_train) / sum(CV$TrainSize)))
print(paste("Test error:", sum(Error_test) / sum(CV$TestSize)))

print(paste("Linear regression with sequential feature selection:"))
print(paste("Training error:", sum(Error_train_fs) / sum(CV$TrainSize)))
print(paste("Test error:", sum(Error_test_fs) / sum(CV$TestSize)))

# Show the selected features
bmplot(attributeNames, 1:K, Features, 
       xlab = "Crossvalidation fold", 
       ylab = "",
       main = "Feature Selection")

# Visualisation of Training and Test Data. 
frame <- data.frame(
  Max.depth = 1:K, 
  Train = colSums(Error_train_fs) / sum(CV$TrainSize),
  Test = colSums(Error_test_fs) / sum(CV$TestSize)
)


frame <- pivot_longer(frame, 
                      cols = 2:3, 
                      names_to = "Data",
                      values_to = "Error")

frame
ggplot(frame, 
       mapping = aes(x = Max.depth, y = Error, color = Data)) +
  geom_point() +
  scale_colour_manual(values = c("Train" = "darkgreen", "Test" = "darkblue")) +
  geom_line() +
  labs(
    subtitle = "K Fold Cross Validation", 
    title = "Diamonds",
    x = "K", 
    y = "Classification Error")

# There is a gap between training and test data, which may signal overfitting.  
# ****** DONT CHANGE******** 

# -----------------------------------------------------------------------
#   
#     Model Comparison 
#
# -----------------------------------------------------------------------

library(performance)

# Look at the performance of the baseline model. 
model_performance(baseline_model)

# Look at the performance of the multiple linear model. 
model_performance(multiple_linear_model)

# Look at the performance of the ridge model
#model_performance(ridge_model)
glmnet::assess.glmnet(ridge_model, glitter_X, glitter_y)

# Look at the performance of the neural network.  
model_performance(nn_diamonds_model)

# Look at the performance of the elastic model
model_performance(elastic_model)


# -----------------------------------------------------------------------
#   
#     Model Comparison : SETUP I 
#
# -----------------------------------------------------------------------

X <- hot_diamonds_data
N <- nrow(X)
M <- ncol(X)


attributeNames <- colnames(X)

## Set the seed to make your partition reproducible
set.seed(1234)

# -----------------------------------------------
# Partition the Data into Training and Test Sets
# ------------------------------------------------
train_ind <- caret::createDataPartition(X$pricePS,
                                        p = 0.8,
                                        list = FALSE)

# Generate the training and test split
X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

#---------------------------------------
# Create a Formula for Linear Regression
#---------------------------------------
# Where attributeNames 11 is the target variable:price.

fmla <- pricePS ~ depth + table + carat_mg + the_length +
  the_width + the_depth

#fmla <- pricePS ~ cut + color + clarity + depth + table + carat_mg + the_length +
#  the_width + the_depth

#---------------------
# Make a Regression Tree
#---------------------
mytree <- rpart(fmla,
                data = X_train,
                method = "anova")

mytree
#---------------------------
#  Advanced Visualisation
#---------------------------
library(rpart.plot)

prp(mytree,
    box.col= viridis::viridis(n = 5, alpha = 0.5),
    border.col = viridis::viridis(n = 5),
    type = 0,
    extra = 1,
    tweak = 0.8,
    under = TRUE,
    compress = TRUE,
    main = "The Regression Tree",
    family = "Avenir")

# ---------------------------------------------------
#yhatA <- predict(mytree, newdata = X_train, type = "vector")
#yhatA <- as.matrix(yhatA)


#---------------------
# Make a Linear Model
#---------------------
linearMod <- lm(fmla,
                data = X_train)

#---------------------
# Make a Neural Network
#---------------------
model_performance(linearMod)

# A vector of nodes in hidden layers
# hidden_layers <- c(1, 2, 3)
# hidden_layers = c(1),

hidden_layers = c(3, 3)

the_neural_network <- neuralnet::neuralnet(formula,
                                           data = X_train,
                                           #hidden = c(1),
                                           hidden = hidden_layers,
                                           linear.output = TRUE)
plot(the_neural_network)

#---------------------
# Make a baseline Model
#---------------------
bline_model <- lm(pricePS  ~  1,
                  data = X_train)


#-------------------------
# Make a Regularised Model
#-------------------------
# lambda = optimal_lambda
lambda = lambda_seq

regularised_model <- glmnet(x = as.matrix(X_train[,2:7]),
                            y = as.matrix(X_train[,1]),
                            alpha = 0,
                            lambda = lambda_seq, # or optimal_lambda ??
                            standardize = TRUE)

# ----- Make predictions for both models using the test data
yhatA <- predict(linearMod, X_test)
yhatA

yhatB <- predict(mytree, X_test)
yhatB

yhatC <- predict(the_neural_network, X_test)

yhatD <- predict(bline_model, X_test)

yhatE <- predict(regularised_model, as.matrix(X_test[ ,2:7]))
yhatE

# ----------------------------------
y_test <- X_test[attributeNames[1]]
head(y_test)

# --------------------------------------------
# Perform statistical comparison of the models
# --------------------------------------------
zA <- abs(y_test - yhatA)^2 # linear model
zB <- abs(y_test - yhatB)^2 # Tree
#zC <- abs(y_test - yhatC)^2 # neural network
zD <- abs(y_test - yhatD)^2 # baseline
zE <- abs(y_test - yhatE)^2 # regularised


# Model Performance
ModelMetrics::rmse(as.matrix(y_test), predicted = as.matrix(yhatE), regularised_model)
ModelMetrics::rmse(as.matrix(y_test), predicted = as.matrix(yhatD), bline_model)
#ModelMetrics::rmse(as.matrix(y_test), predicted = as.matrix(yhatC), the_neural_network)

ModelMetrics::mae(as.matrix(y_test), predicted = as.matrix(yhatE), regularised_model)
ModelMetrics::mae(as.matrix(y_test), predicted = as.matrix(yhatD), bline_model)
#ModelMetrics::mae(as.matrix(y_test), predicted = as.matrix(yhatC), the_neural_network)


AllZeds <- data.frame(zA = zA, 
                      zB = zB, 
                      #zC = zC, 
                      zD = zD, 
                      zE = zE)

# Look at the means... 
#plot(colMeans(AllZeds))

colMeans(AllZeds)

calculate_squared_error <- function(the_test, the_pred) {
  z <- abs(the_test - the_pred)**2
  # cat("The squared error for the model is:" z
  # Confidence interval from t-test for model
  res <- t.test(z, alternative = "two.sided", alpha = 0.05)
  #res <- t.test(the_test, the_pred, paired = TRUE, mu = 0)
  
  CIA <- c(res$conf.int[1], res$conf.int[2])
  return(res)
}

# cat("--Linear Model--", "\n\n")
calculate_squared_error(y_test, yhatA) #zA

cat("-- Regression Tree --", "\n")
calculate_squared_error(y_test, yhatB) #zB

cat("-- Neural Network --", "\n")
#calculate_squared_error(y_test, yhatC) #zC

cat("-- Baseline --", "\n")
calculate_squared_error(y_test, yhatD) #zD

cat("-- Regularised --", "\n")
calculate_squared_error(y_test, yhatE) #zE


#---------------------------
# Model Comparison
#---------------------------

# Compute the Difference In the Squared Error between model A and B
model_comparison <- function(model1, model2) {
  cat("--Model Comparison--", "\n")
  z <- model1 - model2
  cat("The Difference in Squared Error between the Model1 and Model2 is z", "\n")
  
  rezult <- t.test(z,alternative = "two.sided", alpha = 0.05)
  
  cat("The Confidence Interval of z = model1 - model2 is ")
  cat(rezult$conf.int[1], "and",  rezult$conf.int[2], "\n")
  
  cat("The p-value is:", rezult$p.value)
}


cat("-- Neural Network -- and -- Baseline --", "\n")
model_comparison(zC, zD)


cat("-- Regularised -- and -- Baseline --", "\n")
model_comparison(zD, zE)


cat("-- Regularised -- and -- Neural Network--", "\n")
model_comparison(zE, zC)

# The confidence interval provides a range of values
# where the true mean squared error may lie,
# and the p-value indicates whether the difference in
# mean squared errors between the models
# is statistically significant.

# Set up the control parameters for cross-validation
# ctrl <- trainControl(method = "cv", number = 5)
# 
# tune.grid.neuralnet <- expand.grid(
#   layer1 = 1,
#   layer2 = 5,
#   layer3 = 5
# )
# 
# model.neuralnet.caret <- caret::train(
#   formula,
#   data = cbind(sparkle_X, sparkle_y) ,
#   method = "neuralnet",
#   linear.output = TRUE, 
#   tuneGrid = tune.grid.neuralnet,
#   metric = "RMSE",
#   trControl = trainControl(method = "none"))
# 
# sqrt(min(model.neuralnet.caret$results$RMSE))  
# model.neuralnet.caret$bestTune

# -----------------------------------------------------------------------
#   
#     Perform (Two Layer) Nested Cross Validation
#
# -----------------------------------------------------------------------

library(nestedcv)
?nestedcv::nestcv.glmnet
set.seed(123)
nested_model <-nestedcv::nestcv.glmnet(sparkle_y,sparkle_X,
                                       family="gaussian",
                                       # alphaSet = 0 or 1 ,    # Not sure about this one. 
                                       n_outer_folds = 10,      # 5 Fold For Outer Loop # Note: Did you ask for 10? 
                                       n_inner_folds = 10       # 10 Fold for Inner Loop. 
) 

summary(nested_model)        # Actual Wanted Output with Folds. 
#nested_model$output         # The entire output
nested_model$final_coef      # The coefficients
#nested_model$outer_folds    # The Outer Folds
nested_model$dimx            # The dimensions. 
nested_model$final_fit       # MSE and lambda
nested_model$summary         # The Model Summary. 

# ------------------------------
# Visualisation
# ------------------------------

pl <- plot(nested_model$outer_result[[1]]$cvafit)
pl

plot_var_stability(nested_model)

nested_model$outer_result[[1]]$cvafit


nested_model$outer_result[[1]]$cvafit$fits
nested_model$outer_result[[1]]$cvafit$which_alpha


plot_lambdas(nested_model,
             showLegend="bottomright")
nested_model

# overlay directionality using colour
p1 <- plot_var_stability(nested_model, final = FALSE, direction = 1)
p1 + scale_fill_manual(values=c("orange", "green3"))

# show directionality with the sign of the variable importance
p2 <- plot_var_stability(nested_model, final = FALSE, percent = F)
p2

# one hot encoding of the dataset (kind of done)
diamonds
one_hot(diamonds)


#library(glmnetr)
# nested.glmnetr.fit = nested.glmnetr(sparkle_X, y_ = sparkle_y,
#                                     family="gaussian", 
#                                     folds_n=10, 
#                                     doann = 1)
# plot(nested.glmnetr.fit) 
# plot(nested.glmnetr.fit, coefs=TRUE) 
# 
# summary(nested.glmnetr.fit) 
# summary(nested.glmnetr.fit, cvfit=TRUE) 

# -----------------------------------------------------------------------
#   
#    Cross Validation for a Neural Network from Radiant Model
#
# -----------------------------------------------------------------------
# https://radiant-rstats.github.io/radiant.model/reference/cv.nn.html
#install.packages("radiant", repos = "https://radiant-rstats.github.io/minicran/")
#radiant::radiant()
hot_diamonds_data

library(radiant.model)

# Building a Neural Network 
neural_net_mod <- nn(hot_diamonds_data, "pricePS", c("depth", "table", "carat_mg", "the_length", "the_width",  "the_depth"), 
                     type = "regression")

plotnet(neural_net_mod)
plot(neural_net_mod)
plot(neural_net_mod, plots = "garson", custom = TRUE) + labs(title = "Garson plot")

#import the function from Github
# library(devtools)
#source_url('https://gist.github.com/fawda123/7471137/raw/cd6e6a0b0bdb4e065c597e52165e5ac887f5fe95/nnet_plot_update.r')

#plot each model
#plot.nnet(neural_net_mod)


# Cross Validation of the Network. 
cv_radiant_result <- radiant.model::cv.nn(
  neural_net_mod,
  K = 10,                    # number of cross validation passes to use
  repeats = 2,               # repeated cross validation 
  decay = seq(0, 1, 0.2),    # parameter decay : L2 regularisation strength Lambda 
  size = 1:3,                # number of units (nodes) in the hidden layer. 
  seed = 1234,
  trace = TRUE,
  # fun = Rsq                # calculates the r-squared. 
  fun = RMSE                 # calculates the RMSE
) 

?cv.nn # What? 

# ------------------------------
# Results
# ------------------------------
cv_radiant_result

# ------------------------------
# Visualisation
# ------------------------------
q <- data.frame(cv_radiant_result)
q


q<- q %>% 
  mutate(MSE = RMSE..mean.^2) %>% 
  mutate(The_Index = 1:length(RMSE..mean.))

head(q, n = 10)

q  %>% 
  ggplot(mapping = aes(The_Index, MSE))+
  #ggplot(mapping = aes(decay, MSE))+
  geom_point(colour = "cyan4",alpha = 0.7, size = 3)+
  geom_line(alpha = 0.7)+
  #geom_smooth()+
  scale_fill_brewer(palette="Accent")+
  theme(
    legend.position = "bottom",
  )+
  labs(
    title = "Diamonds: Neural Network",
    subtitle = "Mean Squared Error after\nNeural Network Cross Validation",
    x = "Number of Iterations",
    y = "Costs: MSE",
    caption = "Diamonds data from ggplot2 Tidyverse"
  )+
  theme_minimal()


# Note: Check diamonds data does not contain the date. 

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

plot(log(cp_tmp), log(apply(Error_train2_tree, 1, sum) / sum(CV2$TrainSize)),
     xlab = "log(lambda)", ylab = "log(Error)"
)

lines(log(cp_tmp), log(apply(Error_train2_tree, 1, sum) / sum(CV2$TrainSize)))
plot(log(cp_tmp), log(apply(Error_test2_tree, 1, sum) / sum(CV2$TestSize)), col = "red")
lines(log(cp_tmp), log(apply(Error_test2_tree, 1, sum) / sum(CV2$TestSize)), col = "red")

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
rt_tp <- mcnemar(y, y_PCA, y_tree, alpha = alpha)

bar_min <- c(rt_bl$CI[1],rt_bp$CI[1],rt_bt$CI[1],rt_lt$CI[1],rt_lp$CI[1],rt_tp$CI[1])
bar_max <- c(rt_bl$CI[2],rt_bp$CI[2],rt_bt$CI[2],rt_lt$CI[2],rt_lp$CI[2],rt_tp$CI[2])
bar <- c(rt_bl$thetahat,rt_bp$thetahat,rt_bt$thetahat,rt_lt$thetahat,rt_lp$thetahat,rt_tp$thetahat)
models <- c("BL > LRL","BL > LRQ","BL > CT","LRL > CT","LRL > LRQ","LRQ > CT")
I <- as.data.frame(matrix(c(models,bar,bar_min,bar_max),ncol=4))
I$V2 <- as.numeric(I$V2)
I$V3 <- as.numeric(I$V3)
I$V4 <- as.numeric(I$V4)

ggplot() +
  xlab("Compared models") + ylab("Confidence interval of theta") +
  geom_hline(yintercept=0,color = "red") +
  ylim(-0.3,0.05) +
  geom_errorbar(aes(
    x=V1,
    ymin=V3,
    ymax=V4,
    width=0.5
  ), data=I)

# p-value is always 0 because the data-set is very large and p-value formula
# has a N at the denominator. It seems that the tree and the PCA are the
# closest models

################################
# -------  Question 5 ---------
################################

mdl_lr$a0
mdl_lr$beta

Name <- c("","Depth","Table","Price","Carat","x","y","z")
OBS <- clean_diamonds_data[1,c(4:6,9:12)]
colnames(OBS) <- Name[2:8]
OBS <- t(OBS)
colnames(OBS) <- "New"
knitr::kable(OBS,"pipe")

ST <- as.data.frame(matrix(c("Mean",mu[ind_lr, ]),nrow=1))
colnames(ST) <- Name
ST[nrow(ST) + 1,] <- c("St. Dev.",sigma[ind_lr, ])
ST <- t(ST)
ST <- ST[2:8,]
colnames(ST) <- c("Mean","St. Dev.")
knitr::kable(ST,"pipe")

Name <- c("","Depth","Table","Price","Carat","x","y","z")
OBS <- clean_diamonds_data[1,c(4:6,9:12)]
OBS <- as.data.frame(scale(OBS, mu[ind_lr, ], sigma[ind_lr, ]))
colnames(OBS) <- Name[2:8]
OBS <- t(OBS)
colnames(OBS) <- "New"
knitr::kable(OBS,"pipe")

mdl_PCA$a0
mdl_PCA$beta

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
