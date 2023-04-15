#### info: Uses dataset GSE44762. Compare caret machine learning algorithms performance and ####
#### previoulsy run GA/KNN ####

## clean work space 
rm(list=ls())

## set working directory if needed (modify path as needed)
setwd("input_dir")

## install/load required libraries
if (!require("AppliedPredictiveModeling")) install.packages("AppliedPredictiveModeling"); require(AppliedPredictiveModeling)
if (!require("caret")) install.packages("caret"); require(caret)
if (!require("doParallel")) install.packages("doParallel"); require(doParallel)
if (!require("tictoc")) install.packages("tictoc"); require(tictoc)
if (!require("tidyverse")) install.packages("tidyverse"); require(tidyverse)
if (!require("e1071")) install.packages("e1071"); require(e1071)
if (!require("glmnet")) install.packages("glmnet"); require(glmnet)
if (!require("kknn")) install.packages("kknn"); require(kknn)
if (!require("mda")) install.packages("mda"); require(mda)
if (!require("rrcov")) install.packages("rrcov"); require(rrcov) # for C50
if (!require("rrcovHD")) install.packages("rrcovHD"); require(rrcovHD) # for C50
if (!require("CSimca")) install.packages("CSimca"); require(CSimca)
if (!require("pls")) install.packages("pls"); require(pls)
if (!require("klaR")) install.packages("nb"); require(klaR) # for NB
if (!require("randomForest")) install.packages("randomForest"); require(randomForest)
if (!require("import")) install.packages("import"); require(import) # for RF
if (!require("gbm")) install.packages("gbm"); require(gbm)

## read in data a fix format
my_data <- read_tsv("GSE44762_filtered_normalized.txt") %>%
 t() %>% 
 as.data.frame() %>% 
 select_if(~ !any(is.na(.))) %>% 
  `colnames<-`(.[1,]) %>%
  .[-1,] %>%
  `rownames<-`(NULL) %>% 
  mutate_at(vars(starts_with("Cla")),factor) %>%   
  mutate_at(vars(starts_with("ILMN")),as.numeric)



## Split dataset into train & test sets (set seed for replicability)
X = mydata[, -1] # features
y = mydata[, 1] # outcome

## set seed
set.seed(123)

## create statified 70/30 split based on Class for train and test datasets
part.index <- createDataPartition(mydata$Class, 
                                  p = 0.7,                        
                                  list = FALSE)
X_train <- X[part.index, ]
X_test <- X[-part.index, ]
y_train <- y[part.index]
y_test <- y[-part.index]


## set up parallel processing 
# getDoParWorkers() # check available cores
doParallel::registerDoParallel(cores = 8) 
doRNG::registerDoRNG(seed = 123)

#### RFE feature selection####
## define subsets sizes
subsets <- c(1:5, 10, 15, 20, 25)

## setup RFE control
rfe_ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
## run RFE
rfe_profile <- rfe(x=X_train, y=factor(y_train),
                 sizes = subsets,
                 rfeControl = rfe_ctrl,
                 allowParallel = TRUE)

## pick selected subset size 
new_profile <- update(rfe_profile, x=X_train, y=factor(y_train), size = 5) 

## save variables from custom model 
rfe_vars <- pickVars(new_profile$variables, 5)

## filter RFE selected variables
X_train <- X_train[rfe_vars]
X_test <- X_test[rfe_vars]

#### set up cross-validation/trainControl, for caret models ####
set.seed(123)

# 5-fold Cv repeated 20-times
my_control <- trainControl(method = "repeatedcv", # for "cross-validation"
                           number = 5, # 5-fold CV 
                           repeats = 20,# repeated 20 times
                           savePredictions = "final", 
                           returnResamp = "final", #returnResamp = "all",
                           summaryFunction = multiClassSummary,
                           allowParallel = TRUE) 

## create empty output data frame to store result from different models
out<-data.frame(matrix(vector(), 0, 1, dimnames=list(c(), c("Measure"))), stringsAsFactors=F)

## specify output metric
metric <- "Kappa"

##### run models, calculate variable importance and store to out data frame ####
#0 GLM - as performance reference
tic()
set.seed(123)
m_glm <- train(X_train, y_train, method="glm", metric=metric, 
               trControl=my_control, preProcess = c("center", "scale"))

m_glmrelimp<-varImp(m_glm)$importance %>% rownames_to_column("Measure") %>% arrange(-Overall) # relativ importance of features

yhats <- predict(m_glm, newdata = X_test, type = "raw")
m_glmCM <- confusionMatrix(yhats, y_test); m_glmCM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_glmCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_glm = "m_glmCM$overall"), 
             as.data.frame(m_glmCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_glm = "m_glmCM$byClass"))

out[1:length(m_glm$results[1:12]),1] <- colnames(m_glm$results[1:12])
out<-cbind(out,t(m_glm$results[1:12]))
colnames(out)[ncol(out)] <- "m_glm"
out[nrow(out)+1,1] <- "Results test data" # append row with test/prdicte data inof
out <- bind_rows(mutate_all(out, as.character), mutate_all(tmp, as.character))
exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_glm = "exectime$toc - exectime$tic")

out <- bind_rows(mutate_all(out, as.character), mutate_all(exectime, as.character))

#1 (kknn) k-Nearest Neighbors
tic()
m_kknn <- train(X_train, y_train, method="kknn", metric=metric, 
                trControl=my_control,
                tuneLength = 4, 
                preProcess = c("center", "scale")) 

yhats <- predict(m_kknn, newdata = X_test, type = "raw")

m_kknnCM <- confusionMatrix(yhats, y_test)

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_kknnCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_kknn = "m_kknnCM$overall"), 
             as.data.frame(m_kknnCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_kknn = "m_kknnCM$byClass"))

# save to out
tmp1 <- t(m_kknn$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% slice(-(1:2)) %>%
  mutate_all(as.character) %>% dplyr::rename(m_kknn = 1)

out$m_kknn <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers

out[13,"m_kknn"] <- "Results out data" # append with model name
out[14:17,"m_kknn"] <- tmp[,2] 

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_kknn = "exectime$toc - exectime$tic")

out[18,"m_kknn"] <- exectime[,2]

#2 (pda) Penalized Discriminant Analysis
tic()
set.seed(123)
m_pda <- train(X_train, y_train, method="pda", metric=metric, 
               trControl=my_control,
               preProcess = c("center", "scale"))

out <- cbind(out,t(m_pda$results))
colnames(out)[ncol(out)] <- "m_pda"

m_pdarelimp<-varImp(m_pda)$importance[1] %>% rownames_to_column("Variable") %>% rename(Overall= "X....1.0000") %>% arrange(-Overall) %>%
  head(100) # save top 100

yhats <- predict(m_pda, newdata = X_test, type = "raw")

m_pdaCM <-confusionMatrix(yhats, y_test); m_pdaCM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_pdaCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_pda = "m_pdaCM$overall"), 
             as.data.frame(m_pdaCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_pda = "m_pdaCM$byClass"))

# save to out
tmp1 <- t(m_pda$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% dplyr::rename(m_pda = "V1")
out$m_pda <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
# colnames(out)[ncol(out)] <- "m_pda"
out[13,"m_pda"] <- "Results out data" # append with model info
out[14:17,"m_pda"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_pda"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_pda = "exectime$toc - exectime$tic")

out[18,"m_pda"] <- exectime[,2]

#3 (glmnet) GLMnet
tic()
set.seed(123)
m_glmnet <- train(X_train, y_train, method="glmnet", metric=metric, 
                  trControl=my_control,
                  preProcess = c("center", "scale"))

yhats <- predict(m_glmnet, newdata = X_test, type = "raw")

m_glmnetCM <- confusionMatrix(yhats, y_test); m_glmnetCM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_glmnetCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_glmnet = "m_glmnetCM$overall"), 
             as.data.frame(m_glmnetCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_glmnet = "m_glmnetCM$byClass"))

# save to out
tmp1 <- t(m_glmnet$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% dplyr::rename(m_glmnet = 1)
out$m_glmnet <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
out[13,"m_glmnet"] <- "Results out data" # append with model info
out[14:17,"m_glmnet"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_glmnet"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_glmnet = "exectime$toc - exectime$tic")

out[18,"m_glmnet"] <- exectime[,2]

#4 C5.0Tree (Single C5.0 Tree)
tic()
set.seed(123)
m_C5 <- train(X_train, y_train, method="C5.0Tree", metric=metric, 
              trControl=my_control,
              preProcess = c("center", "scale"))

yhats <- predict(m_C5, newdata = X_test, type = "raw")

m_C5CM <- confusionMatrix(yhats, y_test);m_C5CM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_C5CM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_glmnet = "m_C5CM$overall"), 
             as.data.frame(m_C5CM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_glmnet = "m_C5CM$byClass"))

# save to out
tmp1 <- t(m_C5$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% dplyr::rename(m_C5 = 1)
out$m_C5 <- c(tmp1[,1]  %>% as.character(), rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
out[13,"m_C5"] <- "Results out data" # append with model info
out[14:17,"m_C5"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_C5"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_C5 = "exectime$toc - exectime$tic")

out[18,"m_C5"] <- exectime[,2]

#5 CSimca (SIMCA) 
tic()
set.seed(123)
m_CSimca <- train(X_train, y_train, method="CSimca", metric=metric,
                       trControl=my_control,
                       preProcess = c("center", "scale"))
#print(m_CSimca)
yhats <- predict(m_CSimca, newdata = X_test, type = "raw")
m_CSimcaCM <- confusionMatrix(yhats, y_test)

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_CSimcaCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_CSimca = "m_CSimcaCM$overall"), 
             as.data.frame(m_CSimcaCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_CSimca = "m_CSimcaCM$byClass"))

# save to out
tmp1 <- t(m_CSimcaLOOCV$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% dplyr::rename(m_CSimca = 1)
out$m_CSimca <- c(tmp1[,1]  %>% as.character(), rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
# colnames(out)[ncol(out)] <- "m_CSimca"
out[13,"m_CSimca"] <- "Results out data" # append with model info
out[14:17,"m_CSimca"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_CSimca"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_CSimca = "exectime$toc - exectime$tic")

out[18,"m_CSimca"] <- exectime[,2]


#6 pls (Partial Least Squares)  
tic()
set.seed(123)
m_pls <- train(X_train, y_train, method="pls", metric=metric,
               trControl=my_control,
               preProcess = c("center", "scale"))

yhats <- predict(m_pls, newdata = X_test, type = "raw")

m_plsCM <- confusionMatrix(yhats, y_test); m_plsCM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_plsCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_pls = "m_plsCM$overall"), 
             as.data.frame(m_plsCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_pls = "m_plsCM$byClass"))

# save to out
tmp1 <- t(m_pls$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% dplyr::rename(m_pls = 1)
out$m_pls <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
out[13,"m_pls"] <- "Results out data" # append with model info
out[14:17,"m_pls"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_pls"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_pls = "exectime$toc - exectime$tic")

out[18,"m_pls"] <- exectime[,2]

#7 NB (Naive bayes)
tic()
set.seed(123)
m_nb <- train(X_train, y_train, method="nb", metric=metric,
              trControl=my_control,
              preProcess = c("center", "scale"))

yhats <- predict(m_nb, newdata = X_test, type = "raw")

m_nbCM <- confusionMatrix(yhats, y_test); m_nbCM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_nbCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_nb = "m_nbCM$overall"), 
             as.data.frame(m_nbCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_nb = "m_nbCM$byClass"))

# save to out
tmp1 <- t(m_nb$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% 
  slice(-(1:2)) %>% dplyr::rename(m_nb = 1)
out$m_nb <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
out[13,"m_nb"] <- "Results out data" # append with model info
out[14:17,"m_nb"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_nb"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_nb = "exectime$toc - exectime$tic")

out[18,"m_nb"] <- exectime[,2]

#8  parRF (paralell RF)
tic()
m_parRF <- train(X_train, y_train, method="parRF", metric=metric,
                 trControl=my_control,
                 preProcess = c("center", "scale"))
yhats <- predict(m_parRF, newdata = X_test, type = "raw")
m_parRFCM <- confusionMatrix(yhats, y_test); m_parRFCM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_parRFCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_parRF = "m_parRFCM$overall"), 
             as.data.frame(m_parRFCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_parRF = "m_parRFCM$byClass"))
# save to out
tmp1 <- t(m_parRF$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% dplyr::rename(m_parRF = 1)
out$m_parRF <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
out[13,"m_parRF"] <- "Results out data" # append with model info
out[14:17,"m_parRF"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_parRF"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_parRF = "exectime$toc - exectime$tic")

out[18,"m_parRF"] <- exectime[,2]

#9  NN (neural nteworks) 
tic()
set.seed(123)
m_nnet <- train(X_train, y_train, method="nnet", metric=metric,
                trControl=my_control,
                preProcess = c("center", "scale", "pca"))
yhats <- predict(m_nnet, newdata = X_test, type = "raw")
m_nnetCM <- confusionMatrix(yhats, y_test); m_nnetCM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_nnetCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_nnet = "m_nnetCM$overall"), 
             as.data.frame(m_nnetCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_nnet = "m_nnetCM$byClass"))

# save to out
tmp1 <- t(m_nnet$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>%  slice(-1) %>% dplyr::rename(m_nnet = 1)
out$m_nnet <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
out[13,"m_nnet"] <- "Results out data" # append with model info
out[14:17,"m_nnet"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_nnet"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_nnet = "exectime$toc - exectime$tic")

out[18,"m_nnet"] <- exectime[,2]

#10  GBM (gradient boosting) 
tic()
m_gbm <- train(X_train, y_train, method="gbm", metric=metric,
               trControl=my_control,
               preProcess = c("center", "scale"))
yhats <- predict(m_gbm, newdata = X_test, type = "raw")

m_gbmCM <- confusionMatrix(yhats, y_test); m_gbmCM

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_gbmCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_gbm = "m_gbmCM$overall"), 
             as.data.frame(m_gbmCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_gbm = "m_gbmCM$byClass"))

# save to out
tmp1 <- t(m_gbm$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% slice(-(1:3)) %>% dplyr::rename(m_gbm = 1)
out$m_gbm <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
out[13,"m_gbm"] <- "Results out data" # append with model info
out[14:17,"m_gbm"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_gbm"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_gbm = "exectime$toc - exectime$tic")

out[18,"m_gbm"] <- exectime[,2]

#11  rpart (Recursive Partitioning and Regression Trees)
# train.control <- trainControl(
#   method = "repeatedcv",
#   number = 10, ## 10-fold CV
#   repeats = 3,## repeated three times
#   summaryFunction = multiClassSummary, 
#   #classProbs = TRUE)
# The tuneLength parameter is used to determine the total number of combinations that will be evaluated
tic()
m_rpart <- train(X_train, y_train, 
                 method = "rpart2", 
                 tuneLength = 6,
                 trControl = my_control,
                 metric = metric)
data = X_test, type = "raw")
m_rpartCM <- confusionMatrix(yhats, y_test)

# save results; rbind formatted output from confusion matrix 
tmp <- rbind(as.data.frame(m_rpartCM$overall) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_rpart = "m_rpartCM$overall"), 
             as.data.frame(m_rpartCM$byClass) %>% tibble::rownames_to_column(., "Measure") %>% slice(1:2) %>% 
               dplyr::rename(m_rpart = "m_rpartCM$byClass"))

# save to out
tmp1 <- t(m_rpart$results %>% slice(which.max(Balanced_Accuracy))) %>% as.data.frame() %>% dplyr::rename(m_rpart = 1)
out$m_rpart <- c(tmp1[,1], rep(NA, nrow(out)-nrow(tmp1))) # add column to dataframe with differetnt row numbers
out[13,"m_rpart"] <- "Results out data" # append with model info
out[14:17,"m_rpart"] <- tmp[,2] # out[14:(nrow(tmp1) + nrow(tmp) + 1),"m_rpart"] <- tmp[,2]

exectime <- toc()
exectime <- as.data.frame(exectime$toc - exectime$tic) %>% tibble::rownames_to_column(., "Measure") %>% 
  dplyr::rename(m_rpart = "exectime$toc - exectime$tic")

out[18,"m_rpart"] <- exectime[,2]

#### Save results####
write.table(out, "testmodels_kappa.tab", sep="\t", col.names=TRUE, row.names=F,dec=".", quote=FALSE, na="")

##### doseregression cortex with rfe selected vars####
m_lm <- train(X_train, y_train, method="lm", 
               trControl=my_control, preProcess = c("center", "scale"))
m_lmrelimp<-varImp(m_lm)$importance %>% rownames_to_column("Measure") %>% arrange(-Overall) # relativ importance of features
yhats <- predict(m_lm, newdata = X_test, type = "raw")

##### dos regression medulla - with rfe selected vars ####
m_lm_medulla <- train(X_test, y_test, method="lm",  
              trControl=my_control, preProcess = c("center", "scale"))
m_lmrelimp<-varImp(m_lm_medulla)$importance %>% rownames_to_column("Measure") %>% arrange(-Overall) # relativ importance of features
yhats_medulla <- predict(m_lm_medulla, newdata = X_test, type = "raw")
