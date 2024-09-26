# Excerpt of code analysing data to develop automated classifier
# This code will not run as certain code has been removed
# A lot of the data loading / cleaning / preprocessing is removed

# Background:
#   Had ~3 weeks to read material, data, then make models / code
#   Whole 3 Month project (I am not joining) intends to create classifier, and explain it
#   They had engineered features from company,
#   Also had a deep neural network for classification from previous project.
#   I was interested in determining whether a deep neural network was needed,
#   or if it simply inhibits the follow-up task of explaining the classifier.
#   I did not pursue attempts to optimise a given model - this was more a scoping project
#
# There were 2 datasets:
#   1. Dataset 1 had only their engineered features and classifications.
#   2. Dataset 2 was split across multiple excel / csv files and was partially labelled.
# They had scaled their engineered features but not documented how too clearly, difficult to replicate.
# The results between Dataset 1 and Dataset 2 using same features were very different.
#
# Conclusion:
#   1. If they can replicate their scaling and get similar performance as they did for Dataset 1:
#      They should simply keep their features, as they are meaningful and performant.
#   2. If they cannot be replicated, proposed model: mlp_I3B4Y1, is a recommended candidate:
#      It uses their features (with my attempt at their scaling) combined with quantile-based data.
#      The features are therefore interpretable. The model is a simple MLP that should be further optimised.
#      It had an accuracy of ~93% versus the project's DNN's existing 85.8%.*
#      *Care with these metrics: I created my own data pipeline, so tests are not like-for-like.
#   3. If manually selected features are not desired, proposed model: mlp_I2AY1, is good starting point
#      It had an accuracy of ~92%. I think this can be easily increased with additional tuning.
#      I also think errors in the autoencoder should be fed in as additional inputs, out of time!

###
# Imports
###
library(tidyverse)
library(keras)
library(Rtsne)
library(xgboost)

###
# Helper Functions
###
# Original data was classed: 0-4, turned to 1-3.
Align_Grades <- function(df_in, col_names) {
  for(col_name in col_names) {
    df_in[(!(is.na(df_in[,col_name]))&(df_in[,col_name] == 0)),col_name] <- NA
    df_in[(!(is.na(df_in[,col_name]))&(df_in[,col_name] == 2)),col_name] <- 1
    df_in[(!(is.na(df_in[,col_name]))&(df_in[,col_name] == 3)),col_name] <- 2
    df_in[(!(is.na(df_in[,col_name]))&(df_in[,col_name] == 4)),col_name] <- 3
  }
  return(df_in)
}
# Get Accuracy, Precision, Recall of outputs of model
Get_Performance <- function(table_in) {
  accuracy <- sum(diag(table_in)) / sum(table_in)
  precision <- diag(table_in) / colSums(table_in)
  recall <- diag(table_in) / rowSums(table_in)
  cat("\nAcc: ", accuracy, "\n")
  cat("Precision:\n")
  print(precision)
  cat("Recall:\n")
  print(recall)
  return(list(accuracy = accuracy, precision = precision, recall = recall))
}
# Generic application of an xgboost
Do_XGBoost <- function(x_train_in, y_train_in, x_test_in, y_test_in) {
  xg_model <- xgboost(data = x_train_in, label = y_train_in, nround = 100, verbose = FALSE)
  xg_pred  <- round(predict(xg_model, x_test_in))
  xg_out   <- table(factor(y_test_in, levels = c("1","2","3")),
                    factor(xg_pred, levels = c("1","2","3")))
  xg_perf  <- Get_Performance(xg_out)
  n_train  <- nrow(x_train_in)
  n_test   <- nrow(x_test_in)
  return(list(model = xg_model, con_mat = xg_out, perf = xg_perf, counts = c(n_train,n_test)))
}
# Generic application of an MLP
Do_MLP <- function(x_train_in, y_train_in, x_test_in, y_test_in, layers_in, patience_in = 5) {
  if(length(layers_in) == 1) {
    mlp_classifier <- keras_model_sequential() %>%
      layer_dense(units = layers_in, activation = "relu", input_shape = c(ncol(x_train_in)), name = 'encoded_in') %>%
      layer_dense(units = 3, activation = "softmax", name = 'classifier')
  } else { # Assume 2
    mlp_classifier <- keras_model_sequential() %>%
      layer_dense(units = layers_in[1], activation = "relu", input_shape = c(ncol(x_train_in)), name = 'encoded_in') %>%
      layer_dense(units = layers_in[2], activation = "relu", name = 'encoded_in_2') %>%
      layer_dense(units = 3, activation = "softmax", name = 'classifier')
  }
  mlp_classifier %>% compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = c('accuracy'))
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = patience_in, restore_best_weights = TRUE)

  mlp_classifier %>% fit(x_train_in, Convert_Y(y_train_in), epochs = 600, batch_size = 32, callbacks = list(early_stop),
                         validation_data = list(x_test_in, Convert_Y(y_test_in)))

  mlp_pred <- mlp_classifier %>% predict(x_test_in)
  mlp_pred <- apply(mlp_pred, 1, which.max)
  mlp_out   <- table(factor(y_test_in, levels = c("1","2","3")),
                     factor(mlp_pred, levels = c("1","2","3")))
  mlp_perf  <- Get_Performance(mlp_out)
  n_train  <- nrow(x_train_in)
  n_test   <- nrow(x_test_in)
  return(list(model = mlp_classifier, con_mat = mlp_out, perf = mlp_perf, counts = c(n_train,n_test)))
}
# To some extent, features should be visually seperable according to the labels
Visualise <- function(x_in, y_in, name_in) {
  y_in[y_in == 1] <- "darkgray"
  y_in[y_in == 2] <- "blue"
  y_in[y_in == 3] <- "darkred"
  x_in <- x_in[!duplicated(x_in), ]
  x_in <- x_in[complete.cases(x_in),]
  x_plot <- Rtsne(x_in, dims=2)
  plot(x_plot$Y[,1], x_plot$Y[,2],col=y_in, asp=1,
       main = name_in)
}
# Converting to one hot encoding
Convert_Y <- function(y_in) {
  y_in_oh <- cbind(y_in, y_in, y_in)
  y_in_oh[y_in_oh[,1] != 1,1] <- 0
  y_in_oh[y_in_oh[,2] != 2,2] <- 0
  y_in_oh[y_in_oh[,2] == 2,2] <- 1
  y_in_oh[y_in_oh[,3] != 3,3] <- 0
  y_in_oh[y_in_oh[,3] == 3,3] <- 1
  return(y_in_oh)
}
# This extracted their features (Input 4) and 2 potential labels - REMOVED
Get_I4_Data <- function(data_set) {
  # REMOVED
  return(list(x = x_data_grp, y1 = y1_data_grp, y2 = y2_data_grp))
}
# This extracted my summary features (Input 3) and 2 potential labels - Partially REMOVED
Get_I3_Data <- function(data_set) {
  n_data <- nrow(data_set)
  data_grp_set <- data_set %>% group_by(_,_,_) # REMOVED
  data_grp_set <- data_grp_set[,c(_,_,_)]      # REMOVED
  # Need to combine the rows
  n_feats <- 11 # Number of features for single wave
  n_wav <- 56
  x_data_grp <- t(array(as.matrix(unlist(data_grp_set$sum_wav)), dim = c(n_feats*n_wav, n_data/n_wav)))
  mask <- seq(1, (n_data-n_wav)+1, by = n_wav)
  y1_data_grp <- data_grp_set$REMOVED[mask]    # REMOVED
  y2_data_grp <- data_grp_set$REMOVED2[mask]   # REMOVED
  n_data_grp <- nrow(x_data_grp)
  # Applying the scaling logic they used
  x_data_grp <- (x_data_grp * 2) - 1
  return(list(x = x_data_grp, y1 = y1_data_grp, y2 = y2_data_grp))
}
# This further condensed my summary features (Input 3B)
# Domain-specific relationships between the inputs discovered through personal EDA
# Summarising specific sets of inputs with quantile-data, to condense from 616 -> 213 features.
Get_I3B_Data <- function(x_data_grp) {
  # data_set is grouped already, need to undo scaling
  # Data cycles by "cm", we want orientation. "cm" has 7 values.
  x_data_grp <- (x_data_grp + 1) / 2
  n_grp <- nrow(x_data_grp)
  n_ft <- 11
  n_cft <- 3
  n_cm <- 7
  n_ft_cm <- n_ft*n_cm
  n_cft_ft <- n_cft*n_ft
  n_cft_ft_cm <- n_cft*n_ft_cm

  comp_ft <- numeric(n_grp*n_cft_ft_cm)
  temp_ft <- numeric(n_cft_ft_cm)
  q_vals <- c(0.00,0.25,0.50,0.75,1.00)
  for(i_grp in c(1:n_grp)){
    for(i_cm in c(1:n_cm)) {
      for(i_ft in c(1:n_ft)) {
        temp <- quantile(x_data_grp[i_grp, seq((n_ft*(i_cm-1))+i_ft,(n_wav*(n_ft-1))+i_ft, by=n_ft_cm)], q_vals)
        temp <- c(temp[5]-temp[1],temp[4]-temp[2],temp[3])
        temp_ft[((n_cft_ft*(i_cm-1))+(n_cft*(i_ft-1))+1):((n_cft_ft*(i_cm-1))+(n_cft*(i_ft-1))+n_cft)] <- temp
      }
    }
    comp_ft[(((i_grp-1)*n_cft_ft_cm)+1):(i_grp*n_cft_ft_cm)] <- temp_ft
  }
  comp_ft <- t(array(comp_ft, dim = c(n_cft_ft_cm,n_grp)))
  x_data_grp <- comp_ft
  n_data_grp <- nrow(x_data_grp)
  # Applying the scaling logic they used
  x_data_grp <- (x_data_grp * 2) - 1

  return(x_data_grp)
}
# This extracted my condensed version of the raw wave data (Input 2)
Get_I2_Data <- function(x_data) {
  n_data <- nrow(x_data)
  len_hex <- 44
  x_data <- t(array(unlist(x_data$comp_wav), dim = c(len_hex, n_data)))
  # I don't care about the "bounce" and will remove
  x_data[x_data < 0.1] <- 0
  # Applying the scaling logic they used
  x_data <- (x_data * 2) - 1
  return(x_data)
}
# Labels were associated with sets of waves, not individual, function groups waves appropriately
Group_Waves <- function(x_data) {
  n_data  <- nrow(x_data)
  n_feats <- ncol(x_data) # Number of features for single wave
  n_wav   <- 56
  x_data <- t(array(t(x_data), dim = c(n_feats*n_wav, n_data/n_wav)))
  return(x_data)
}

#####
### Section 0: Background - Alternative Data
#####
# This is a different data source, I am looking at it to check try and guess which "grade" is being used
df_train <- read_csv("REMOVED", col_names = FALSE)   # REMOVED
df_train <- Align_Grades(df_train, c("grade"))
df_test <- read_csv("REMOVED", col_names = FALSE)    # REMOVED
df_test <- Align_Grades(df_test, c("grade"))

train <- as.matrix(df_train)
test <- as.matrix(df_test)

Visualise(train[,c(2:6)], train[,1], "Alt Data")
boxplot(train[,c(2:6)])

xg_Alt <- Do_XGBoost(train[,c(2:6)], train[,1], test[,c(2:6)], test[,1])
# This is overall, very good performance.
# In theory, this should create a benchmark/proxy to compare other dataset against

#####
### Section 1: Import Data
#####
# Context:
# REMOVED


#####
### Section 2: PreProcess Inputs
#####
# Input 1: Raw Wave REMOVED
hex_wav <- REMOVED
# Input 2: compressed wave, REMOVED
comp_wav <- REMOVED
# Input 3: summary statistics
# Look for Q0, Q1, Q2, Q3, Q4 points, x and y values. And cumsum of all.
sum_wav <- hex_wav / 510
sum_wav[sum_wav < 0.1] <- 0
n_feats <- 11
feats <- numeric(n_hex*n_feats)
for(idy in c(1:ncol(sum_wav))) {
  if(sum(sum_wav[,idy]) == 0) {
    feats[(((idy-1)*n_feats)+1):(idy*n_feats)] <- rep(0, n_feats)
  } else {
    x_pt_st  <- which.max(sum_wav[,idy] >= 0.1)
    x_pt_end <- which.min(sum_wav[c(x_pt_st:128),idy] > 0.1) - 1
    x_temp <- cumsum(sum_wav[c(x_pt_st:(x_pt_st+x_pt_end)),idy])
    y_q <- quantile(x_temp, probs=c(0.25, 0.50, 0.75), type=4)
    x_pt_q1 <- which.max(x_temp > y_q[1])
    x_pt_q2 <- which.max(x_temp > y_q[2])
    x_pt_q3 <- which.max(x_temp > y_q[3])
    y_pt_st  <- sum_wav[x_pt_st,idy]
    y_pt_end <- sum_wav[(x_pt_st+x_pt_end)-1,idy]
    y_pt_q1  <- sum_wav[x_pt_st+x_pt_q1,idy]
    y_pt_q2  <- sum_wav[x_pt_st+x_pt_q2,idy]
    y_pt_q3  <- sum_wav[x_pt_st+x_pt_q3,idy]
    feats[(((idy-1)*n_feats)+1):(idy*n_feats)] <- c(x_pt_st/128, x_pt_q1/128, (x_pt_q2-x_pt_q1)/128, (x_pt_q3-x_pt_q2)/128, (x_pt_end-x_pt_q3)/128,
                                                    y_pt_st, y_pt_q1, y_pt_q2, y_pt_q3, y_pt_end, (x_temp[length(x_temp)])/64)
  }
}
sum_wav <- array(feats, dim = c(n_feats,n_hex))

# Input 4: Context, and their features, align data.frames REMOVED
df_id <- REMOVED

# Define Test / Train Set
mask <- REMOVED # Valid Rows
# I will store 10% for testing, these are kept isolated completely.
test_grp_id  <- mask[sample(c(1:nrow(mask)), round(nrow(mask)/10), replace = FALSE),]
train_grp_id <- anti_join(mask, test_grp_id)
# Save / Load these to keep consistency between sessions:
# Load Since I have these and don't want them changed each time
# test_grp_id <- read_csv("test_grp_id.csv", col_types = cols(...1 = col_skip()))
# train_grp_id <- read_csv("train_grp_id.csv", col_types = cols(...1 = col_skip()))

#####
### Section 3: Experiment Set 1: Looking at their features (I4) and Comparing Dataset
#####
test_set <- inner_join(df_id, test_grp_id, by = c("REMOVED"))
train_set <- inner_join(df_id, train_grp_id, by = c("REMOVED"))
# Get the x and y data for train and test
train_set_grp <- Get_I4_Data(train_set)
test_set_grp <- Get_I4_Data(test_set)
# Visualise the data
Visualise(train_set_grp$x, train_set_grp$y1, "I4Y1")
Visualise(train_set_grp$x, train_set_grp$y2, "I4Y2")
boxplot(train_set_grp$x)
# Fitting a basic xgboost model to the training data
# Trying Y1 as "grade_scores", Y2 as "grade_tester_scores"
xg_I4Y1 <- Do_XGBoost(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1)
xg_I4Y2 <- Do_XGBoost(train_set_grp$x, train_set_grp$y2, test_set_grp$x, test_set_grp$y2)
# Doing their MLP to check (THIS ONE IS COPYING THEIRS!!!)
mlp_I4Y1 <- Do_MLP(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1, 10)
mlp_I4Y2 <- Do_MLP(train_set_grp$x, train_set_grp$y2, test_set_grp$x, test_set_grp$y2, 10)
# MLP of different setting
mlp_I4Y1b <- Do_MLP(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1, c(5), 10)
mlp_I4Y2b <- Do_MLP(train_set_grp$x, train_set_grp$y2, test_set_grp$x, test_set_grp$y2, c(5), 10)
# TODO: Really big drop off in performance... This is important to be addressed!

###
# Experiment Set 2.A: Looking at Sum_Wav (I3)
###
# This is unchanged if running sequentially after Set 1
# test_set <- inner_join(df_id, test_grp_id, by = c("REMOVED"))
# train_set <- inner_join(df_id, train_grp_id, by = c("REMOVED"))
# Get the x and y data for train and test
train_set_grp <- Get_I3_Data(train_set)
test_set_grp <- Get_I3_Data(test_set)
# Visualise
Visualise(train_set_grp$x, train_set_grp$y1, "I3Y1")
Visualise(train_set_grp$x, train_set_grp$y2, "I3Y2")
# Fitting
xg_I3Y1 <- Do_XGBoost(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1)
xg_I3Y2 <- Do_XGBoost(train_set_grp$x, train_set_grp$y2, test_set_grp$x, test_set_grp$y2)
# MLP - Rest do not copy theirs
mlp_I3Y1 <- Do_MLP(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1, c(56,10), 40)
mlp_I3Y2 <- Do_MLP(train_set_grp$x, train_set_grp$y2, test_set_grp$x, test_set_grp$y2, c(56,10), 40)
# Something is different about this dataset than the initial

###
# Experiment Set 2.B: Looking at Sum_Wav (I3): Looking at compressing
###
# Want only the Q0,Q1,Q2,Q3,Q4 of each Feature for the waves for each orientation.
# Want then only the Q4-Q0, and Q3-Q1, and Q2, for each orientation.
# Only changing x data from Set 2.A not y
train_set_grp$x <- Get_I3B_Data(train_set_grp$x)
test_set_grp$x <- Get_I3B_Data(test_set_grp$x)
# Visualise
Visualise(train_set_grp$x, train_set_grp$y1, "I3BY1")
Visualise(train_set_grp$x, train_set_grp$y2, "I3BY2")
# Fitting
xg_I3BY1 <- Do_XGBoost(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1)
xg_I3BY2 <- Do_XGBoost(train_set_grp$x, train_set_grp$y2, test_set_grp$x, test_set_grp$y2)
mlp_I3BY1 <- Do_MLP(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1, c(56,10), 40)
mlp_I3BY2 <- Do_MLP(train_set_grp$x, train_set_grp$y2, test_set_grp$x, test_set_grp$y2, c(56,10), 40)
# Hard to say if this representation is "better"

###
# Experiment Set 3: Looking combinations: I3 + I4, I3B + I4 - Don't care about Y2 anymore
###
# Using existing I3B
train_set_grp$x <- cbind(train_set_grp$x, Get_I4_Data(train_set)$x)
test_set_grp$x <- cbind(test_set_grp$x, Get_I4_Data(test_set)$x)
# Visualise
Visualise(train_set_grp$x, train_set_grp$y1, "I3B4Y1")
Visualise(train_set_grp$x, train_set_grp$y2, "I3B4Y2")
# Fitting
xg_I3B4Y1 <- Do_XGBoost(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1)
mlp_I3B4Y1 <- Do_MLP(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1, 10, 40)
# This is the recommended model. Good performance, and interpretable features: a keeper!
# I34
train_set_grp <- Get_I4_Data(train_set)
test_set_grp <- Get_I4_Data(test_set)
train_set_grp$x <- cbind(train_set_grp$x, Get_I3_Data(train_set)$x)
test_set_grp$x <- cbind(test_set_grp$x, Get_I3_Data(test_set)$x)
# Visualise
Visualise(train_set_grp$x, train_set_grp$y1, "I34Y1")
Visualise(train_set_grp$x, train_set_grp$y2, "I34Y2")
# Fitting
xg_I34Y1 <- Do_XGBoost(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1)
mlp_I34Y1 <- Do_MLP(train_set_grp$x, train_set_grp$y1, test_set_grp$x, test_set_grp$y1, 10, 40)


###
# Experiment Set 4: Looking at Comp_Wav (I2): Don't care about grade_tester_scores (Y2) anymore
###
# The set up is now different. Doing it in 3 parts.
# Classification is done on Groups of Waves
# First looking at raw input on basic model set up
train_set <- inner_join(df_id, train_grp_id, by = c("REMOVED"))
x_train   <- Get_I2_Data(train_set)
test_set  <- inner_join(df_id, test_grp_id, by = c("REMOVED"))
x_test    <- Get_I2_Data(test_set) # Used for 3.3 Only
# Reuse Labels
train_set_grp$x <- Group_Waves(x_train)
test_set_grp$x <- Group_Waves(x_test)
# Visualise
Visualise(train_set_grp$x, train_set_grp$y1, "I2Y1")
# Simple Validation
xg_I2Y1  <- Do_XGBoost(train_set_grp$x, train_set_grp$y1,
                       test_set_grp$x, test_set_grp$y1)
mlp_I2Y1 <- Do_MLP(train_set_grp$x, train_set_grp$y1,
                   test_set_grp$x, test_set_grp$y1, c(308,77), 30)
# MLP seems to be doing okay here

###
# Experiment Set 5: Comp_Wav Encoder - No Y, No Test
###
# Can use all comp_wav, not just those with labels - will still isolate test though
# test_set <- inner_join(df_id, test_grp_id, by = c("REMOVED")) # Unchanged
# x_test   <- Get_I2_Data(test_set)
train_set_all <- anti_join(df_id, test_grp_id, by = c("REMOVED")) # ALL not test
x_train_all   <- Get_I2_Data(train_set_all)

# Approach 5.A) MLP
raw_enc_mlp_in  <- layer_input(shape = c(44), name = "raw_in")
raw_enc_mlp_out <- raw_enc_mlp_in %>%
  layer_dense(units = 22, activation = 'relu', name = "raw_encoding_1") %>%
  layer_dense(units = 11, activation = 'relu', name = "raw_encoding_2") %>%
  layer_dense(units = 5, activation = 'relu', name = "raw_encoded")
raw_enc_mlp <- keras_model(inputs = raw_enc_mlp_in, outputs = raw_enc_mlp_out)

raw_dec_mlp_in  <- layer_input(shape = c(5), name = "raw_encoded_in")
raw_dec_mlp_out <- raw_dec_mlp_in %>%
  layer_dense(units = 11, activation = 'relu', name = "raw_decoding_1") %>%
  layer_dense(units = 22, activation = 'relu', name = "raw_decoding_2") %>%
  layer_dense(units = 44, activation = 'linear', name = "raw_decoded")
raw_dec_mlp <- keras_model(inputs = raw_dec_mlp_in, outputs = raw_dec_mlp_out)

raw_ae_mlp <- keras_model(inputs = raw_enc_mlp$input, outputs = raw_dec_mlp(raw_enc_mlp$output))
raw_ae_mlp %>% compile(optimizer = 'adam', loss = 'mean_squared_error')
early_stop <- callback_early_stopping(monitor = "loss", patience = 5, restore_best_weights = TRUE)
raw_ae_mlp %>% fit(x_train_all, x_train_all, epochs = 600, batch_size = 256, callbacks = list(early_stop))
freeze_weights(raw_enc_mlp)
freeze_weights(raw_dec_mlp)
save_model_weights_tf(raw_enc_mlp, "./checkpoints/mlp_encoder_02")
save_model_weights_tf(raw_dec_mlp, "./checkpoints/mlp_decoder_02")
save_model_weights_tf(raw_ae_mlp, "./checkpoints/mlp_autoencoder_02")

# 5.A) Simple Validation - Encode, Group, Then Classify
x_train_enc_mlp <- raw_enc_mlp %>% predict(x_train)
x_test_enc_mlp <- raw_enc_mlp %>% predict(x_test)
train_set_grp$x <- Group_Waves(x_train_enc_mlp)
test_set_grp$x <- Group_Waves(x_test_enc_mlp)
# Visualise
Visualise(train_set_grp$x, train_set_grp$y1, "I2AY1")
# Test
xg_I2AY1  <- Do_XGBoost(train_set_grp$x, train_set_grp$y1,
                        test_set_grp$x, test_set_grp$y1)
mlp_I2AY1 <- Do_MLP(train_set_grp$x, train_set_grp$y1,
                    test_set_grp$x, test_set_grp$y1, c(308,77), 30)
# Being able to leverage extra data via AE seems to have helped a small amount

# Approach 5.B) LSTM - Takes over an hour to train, care!
raw_enc_lstm_in  <- layer_input(shape = c(44, 1), name = "raw_in")
raw_enc_lstm_out <- raw_enc_lstm_in %>%
  layer_lstm(units = 22, activation = 'relu', name = "raw_encoding_1", return_sequences = TRUE) %>%
  layer_lstm(units = 11, activation = 'relu', name = "raw_encoding_2", return_sequences = TRUE) %>%
  layer_lstm(units = 5, activation = 'relu', name = "raw_encoded", return_sequences = FALSE)
raw_enc_lstm <- keras_model(inputs = raw_enc_lstm_in, outputs = raw_enc_lstm_out)

raw_dec_lstm_in  <- layer_input(shape = c(5), name = "raw_encoded_in")
raw_dec_lstm_out <- raw_dec_lstm_in %>%
  layer_repeat_vector(44) %>%
  layer_lstm(units = 11, activation = 'relu', name = "raw_decoding_1", input_shape = c(5, 1), return_sequences = TRUE) %>%
  layer_lstm(units = 22, activation = 'relu', name = "raw_decoding_2", return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = 'linear', name = "raw_decoded")
raw_dec_lstm <- keras_model(inputs = raw_dec_lstm_in, outputs = raw_dec_lstm_out)

raw_ae_lstm <- keras_model(inputs = raw_enc_lstm$input, outputs = raw_dec_lstm(raw_enc_lstm$output))
raw_ae_lstm %>% compile(optimizer = 'adam', loss = 'mean_squared_error')
early_stop <- callback_early_stopping(monitor = "loss", patience = 5, restore_best_weights = TRUE)
raw_ae_lstm %>% fit(x_train_all, x_train_all, epochs = 600, batch_size = 256, callbacks = list(early_stop))
freeze_weights(raw_enc_lstm)
freeze_weights(raw_dec_lstm)
save_model_weights_tf(raw_enc_lstm, "./checkpoints/lstm_encoder_02")
save_model_weights_tf(raw_dec_lstm, "./checkpoints/lstm_decoder_02")
save_model_weights_tf(raw_ae_lstm, "./checkpoints/lstm_autoencoder_02")

# 5.B) Simple Validation - Encode, Group, Then Classify
x_train_enc_lstm <- raw_enc_lstm %>% predict(x_train)
x_test_enc_lstm <- raw_enc_lstm %>% predict(x_test)
train_set_grp$x <- Group_Waves(x_train_enc_lstm)
test_set_grp$x <- Group_Waves(x_test_enc_lstm)
# Visualise
Visualise(train_set_grp$x, train_set_grp$y1, "I2BY1")
# Test
xg_I2BY1  <- Do_XGBoost(train_set_grp$x, train_set_grp$y1,
                        test_set_grp$x, test_set_grp$y1)
mlp_I2BY1 <- Do_MLP(train_set_grp$x, train_set_grp$y1,
                    test_set_grp$x, test_set_grp$y1, c(308,77), 30)
# The LSTM itself is much better at encoding, but the MLP classifier finds it hard to interpret, interesting!

# Approach 5.C) CNN
train_set_all <- anti_join(df_id, test_grp_id, by = c("REMOVED")) # ALL not test
x_train_all   <- Get_I2_Data(train_set_all) # Get some time data

raw_enc_cnn_in  <- layer_input(shape = c(44, 1), name = "raw_in")
raw_enc_cnn_out <- raw_enc_cnn_in %>%
  layer_conv_1d(filters = 11, kernel_size = 3, activation = 'relu', name = "raw_encoding_1", padding = 'same') %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 5, kernel_size = 3, activation = 'relu', name = "raw_encoding_2", padding = 'same') %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_conv_1d(filters = 2, kernel_size = 3, activation = 'relu', name = "raw_encoding_3", padding = 'same') %>%
  layer_flatten() %>%
  layer_dense(units = 5, activation = 'relu', name = "raw_encoded")
raw_enc_cnn <- keras_model(inputs = raw_enc_cnn_in, outputs = raw_enc_cnn_out)

raw_dec_cnn_in  <- layer_input(shape = c(5), name = "raw_encoded_in")
raw_dec_cnn_out <- raw_dec_cnn_in %>%
  layer_dense(units = 11 * 2, activation = 'relu', name = "raw_decoding_1") %>%
  layer_reshape(target_shape = c(11, 2)) %>%
  layer_conv_1d(filters = 5, kernel_size = 3, activation = 'relu', name = "raw_decoding_2", padding = 'same') %>%
  layer_upsampling_1d(size = 2) %>%
  layer_conv_1d(filters = 11, kernel_size = 3, activation = 'relu', name = "raw_decoding_3", padding = 'same') %>%
  layer_upsampling_1d(size = 2) %>%
  layer_flatten() %>%
  layer_dense(units = 44, activation = 'linear', name = "raw_decoded")
raw_dec_cnn <- keras_model(inputs = raw_dec_cnn_in, outputs = raw_dec_cnn_out)

raw_ae_cnn <- keras_model(inputs = raw_enc_cnn$input, outputs = raw_dec_cnn(raw_enc_cnn$output))
raw_ae_cnn %>% compile(optimizer = 'adam', loss = 'mean_squared_error')
early_stop <- callback_early_stopping(monitor = "loss", patience = 5, restore_best_weights = TRUE)
raw_ae_cnn %>% fit(x_train_all, x_train_all, epochs = 600, batch_size = 256, callbacks = list(early_stop))
freeze_weights(raw_enc_cnn)
freeze_weights(raw_dec_cnn)
save_model_weights_tf(raw_enc_cnn, "./checkpoints/cnn_encoder_02")
save_model_weights_tf(raw_dec_cnn, "./checkpoints/cnn_decoder_02")
save_model_weights_tf(raw_ae_cnn, "./checkpoints/cnn_autoencoder_02")

# 5.C) Quick Validation
x_train_enc_cnn <- raw_enc_cnn %>% predict(x_train)
x_test_enc_cnn <- raw_enc_cnn %>% predict(x_test)
train_set_grp$x <- Group_Waves(x_train_enc_cnn)
test_set_grp$x <- Group_Waves(x_test_enc_cnn)
# Visualise
Visualise(train_set_grp$x, train_set_grp$y1, "I2CY1")
# Test
xg_I2CY1  <- Do_XGBoost(train_set_grp$x, train_set_grp$y1,
                        test_set_grp$x, test_set_grp$y1)
mlp_I2CY1 <- Do_MLP(train_set_grp$x, train_set_grp$y1,
                    test_set_grp$x, test_set_grp$y1, c(308,77), 30)
# I think this is more a "me" problem. The CNN is probably poorly architected.

# 5.A|B|C) Validation of Encoding
x_train_ae_mlp  <- raw_ae_mlp %>% predict(x_train)
x_train_ae_lstm <- raw_ae_lstm %>% predict(x_train)
x_train_ae_cnn  <- raw_ae_cnn %>% predict(x_train)
x_train_ae_mlp_err  <- x_train - x_train_ae_mlp
x_train_ae_lstm_err <- x_train - x_train_ae_lstm[,,1]
x_train_ae_cnn_err <- x_train - x_train_ae_cnn[,]
# Look at the particularly bad ones
temp <- rowSums(abs(x_train_ae_cnn_err))
quantile(temp)
mask <- which(temp > 5)
for (i_plot in mask) {
  plot(x_train[i_plot,], type = 'l', col = "black")
  lines(x_train_ae_mlp[i_plot,], col = "red")
  lines(x_train_ae_mlp_err[i_plot,], col = "orange")
  lines(x_train_ae_lstm[i_plot,,1], col = "blue")
  lines(x_train_ae_lstm_err[i_plot,], col = "cyan")
  lines(x_train_ae_cnn[i_plot,], col = "darkgreen")
  lines(x_train_ae_cnn_err[i_plot,], col = "green")
}
# The encoders are sensitive to time. CNN as trained did not perform well.

###
# Experiment Set 6.A: Grp_Wav Encoder - No Y, No Test
###
# Take Encoded (MLP - 5.A) as input
# Only want valid so going back to originally defined dataset
train_set <- inner_join(df_id, train_grp_id, by = c("REMOVED"))
train_set <- Get_I2_Data(train_set)
x_train_enc_mlp <- raw_enc_mlp %>% predict(train_set)
x_test_enc_mlp <- raw_enc_mlp %>% predict(test_set)
n_train <- nrow(x_train_enc_mlp)
n_test <- nrow(x_test_enc_mlp)
# Need to combine the rows for Groups
n_feats = 5 # Number of features for single wave
x_train_enc_mlp_grp <- Group_Waves(x_train_enc_mlp)
x_test_enc_mlp_grp  <- Group_Waves(x_test_enc_mlp)

grp_enc_mlp_in  <- layer_input(shape = c(280), name = "raw_encoded_in")
grp_enc_mlp_out <- grp_enc_mlp_in %>%
  layer_dense(units = 140, activation = "relu", name = "grp_encoding") %>%
  layer_dense(units = 70, activation = "relu", name = "grp_encoded")
grp_enc_mlp <- keras_model(inputs = grp_enc_mlp_in, outputs = grp_enc_mlp_out)

grp_dec_mlp_in  <- layer_input(shape = c(70), name = "grp_encoded_in")
grp_dec_mlp_out <- grp_dec_mlp_in %>%
  layer_dense(units = 140, activation = "relu", name = "grp_decoding") %>%
  layer_dense(units = 280, activation = "linear", name = "grp_decoded")
grp_dec_mlp <- keras_model(inputs = grp_dec_mlp_in, outputs = grp_dec_mlp_out)

grp_ae_mlp <- keras_model(inputs = grp_enc_mlp$input, outputs = grp_dec_mlp(grp_enc_mlp$output))
grp_ae_mlp %>% compile(optimizer = 'adam', loss = 'mean_squared_error')
early_stop <- callback_early_stopping(monitor = "loss", patience = 5, restore_best_weights = TRUE)
grp_ae_mlp %>% fit(x_train_enc_mlp_grp, x_train_enc_mlp_grp, epochs = 600, batch_size = 256, callbacks = list(early_stop))

# ...And so on... Looked at 6.B, 6.C. Generally, this second stage of encoding hurt performance
# It is enforcing expectation, masking abnormalities. The errors are key features being neglected here!
