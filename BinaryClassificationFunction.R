binaryClassification1 <- function(data, file_location) {
  # Models Used
  # Logistic Regression and with Regularizations
  # 1. Logistic Regression 'glm'
  # 2. Elastic Logistic, L1 and L2 cost 'glmnet'
  # 3. Boosted Logistic Regression 'LogitBoost'
  
  # Discriminant Analysis
  # 1. Linear Discriminant Analysis 'lda'
  # 2. Quadratic Discriminant Analysis 'qda'
  # 3. Sparse Discriminant Analysis 'sparseLDA'
  # 4. Naive Bayes Classifier 'nbDiscrete'
  # 5. Nearest Shrunked Centroid 'pam'
  # 6. Flexible Discrimant Analysis 'fda'
  # 7. Bagged Flexible Discriminant Analysis 'bagFDA'
  # 8. Mixed Discriminant Analysis 'mda'
  # 9. Penalized Discriminant Analysis 'pda'
  # 10. Regularized Discriminant Analysis 'rda'
  
  # Matrix Decomposition Method
  # 1. Partial Least Squares Discriminant 'pls'
  
  # Prototype Methods
  # 1. K Nearest Neighbor 'knn'

  
  library(caret)
  library(caretEnsemble)
  library(MLmetrics)
  library(mlbench)
  library(pROC)
  library(gbm)
  library(earth)
  library(mda)
  library(MASS)
  library(rpart)
  library(randomForest)
  library(adabag)
  library(dplyr)
  library(glmnet)
  library(TH.data)
  library(party)
  library(AppliedPredictiveModeling)
  library(caTools)
  library(EnvStats) # I will use geometric mean for this metrics
  set.seed(1)
  # create a train and test set

  trainIndex <- createDataPartition(y = data[,1], p = 0.7, list=F)
  train <- data[trainIndex,]
  test <- data[-trainIndex,]
  
  # this is the list of classification models used
  modNameList <- c('glm', 'glmnet','LogitBoost','lda','qda','sparseLDA',
                   'nbDiscrete','pam','fda','bagFDA','mda','pda','rda', 
                   'pls','knn')
  
  # define a resampling variable
  ctrl <- trainControl(method='cv',
                       # 10-fold CV
                       number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                       summaryFunction = twoClassSummary, # gives classification statistics
                       classProbs = T,
                       allowParallel=T)
  # this is for algorithm without probabilities
  ctrl2 <- trainControl(method='cv',
                        # 10-fold CV
                        number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                        summaryFunction = twoClassSummary, # gives classification statistics
                        classProbs = F,
                        allowParallel=T)
  
  n <- nrow(data) # number of rows in data
  p <- ncol(data)-1 # number of predictors
  
  modParamList <- list(
    # logistic regression 'glm'
    glm = caretModelSpec(method='glm', tuneGrid=c()), 
    # penalized logistic regression 'glmnet'
    glmnet = caretModelSpec(method='glmnet', tuneGrid=expand.grid(alpha=seq(0,1,by=0.05), # elasticnet mix percentage 0='ridge', 1='lasso'
                                                                  lambda=seq(0,5,by=0.1))), # regularization parameter
    # Basically, you only need alpha when training and can get predictions across different values of lambda using predict.glmnet. 
    # logistic boost 'LogitBoost'
    LogitBoost = caretModelSpec(method='LogitBoost',tuneGrid = expand.grid(nIter=3000)),
    # linear discriminant analysis 'lda'
    lda = caretModelSpec(method='lda', tuneGrid=c()),
    # quadratic discriminant analysis 'qda'
    qda = caretModelSpec(method='qda',tuneGrid=c()),  
    # sparse linear discriminant analysis 'sparseLDA'
    sparseLDA = caretModelSpec(method='sparseLDA', tuneGrid=expand.grid(NumVars = 1:p, 
                                                                        lambda=c(0,0.01,0.1,1))), # The weight on the L2-norm for elastic net regression
    # naive bayes classifier 'nbDiscrete'
    # if all variables are discrete
    nbDiscrete = caretModelSpec(method='nbDiscrete',tuneGrid=expand.grid(smooth=1:3)),
    # if all variables continuous
    
    # nearest shrunken centroid 'pam'
    pam = caretModelSpec(method='pam',tuneGrid=expand.grid(threshold=seq(0,3,by=0.5))),
    # flexible discriminant analysis 'fda'
    fda = caretModelSpec(method='fda', tuneGrid=expand.grid(degree=1:3, nprune=1:p)),
    # bagged flexible discriminants 'bagFDA'
    bagfda = caretModelSpec(method='bagFDA', tuneGrid=expand.grid(degree=1:3, nprune=1:p)),
    # mixture discriminant analysis 'mda'
    mda = caretModelSpec(method='mda', tuneGrid = expand.grid(subclasses=1:p)),
    # penalized discriminant analysis 'pda'
    pda = caretModelSpec(method='pda', tuneGrid=expand.grid(lambda=seq(0,1,by=0.1))),
    # regularized discriminant analysis 'rda'
    rda = caretModelSpec(method='rda', tuneGrid=expand.grid(gamma= seq(0,1,by=0.1),
                                                            lambda= seq(0,1,by=0.1))),
    # partial least squares 'pls'
    pls = caretModelSpec(method='pls',tuneGrid=expand.grid(ncomp=1:p)),
    # k-nearest neighbors 'knn'
    knn = caretModelSpec(method='knn', tuneGrid=expand.grid(k=seq(1,ceiling(sqrt(n)),by=2)))
    
  )
  
  # create a formlua object for feeding into caret train
  response <- noquote(colnames(data)[1])
  formula <- formula(paste0(response, "~", "."))
  
  # initialize with empty list for 1) model 2) confusion matrix 3) auc 4) variable importance plot
  modList <- list() 
  modList <- caretList(formula,
                       train,
                       trControl = ctrl,
                       tuneList = modParamList,
                       continue_on_fail = T,
                       preProcess = c('center','scale'))
  # file_location <- 'C:/Users/jalee/Desktop/Email_Model/'
  saveRDS(modList, paste0(file_location, 'classification_models1'))
  # readRDS('C:/Users/jalee/Desktop/Email_Model/classification_models')
  
  i <- 1
  modPred <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  modPredProb <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  colnames(modPred) <- names(modList)
  colnames(modPredProb) <- names(modList)
  cmList <- list()
  aucList <- c()
  pltList <- list()
  bestParamList <- list()
  for (mod  in modList) {
    modPred[,i] <- predict(mod, newdata=test) # output 1, 2
    modPredProb[,i] <- predict(mod, newdata=test, type='prob')[,1]
    # confusionMatrix requires factor for both predicted and reference
    cmList[[i]] <- confusionMatrix(data=factor(ifelse(modPred[,i]==1,levels(test[,1])[1],levels(test[,1])[2])),
                                   reference=factor(test[,1]))
    # aucList requires numeric values
    aucList[i] <- roc(ifelse(test[,1]==levels(test[,1])[1],1,2),
                      modPred[,i])$auc
    # list of variable importance
    pltList[[i]] <- plot(varImp(mod,scale=F))
    # list of best hyperparameters
    bestParamList[[i]] <- mod$bestTune
    i <- i + 1
  }
  
  # model_preds <- lapply(model_list, predict, newdata=testing, type="prob")
  # model_preds <- lapply(model_preds, function(x) x[,"M"])
  # model_preds <- data.frame(model_preds)
  
  write.csv(modPred, paste0(file_location, 'modPred1.csv'))
  write.csv(modPredProb, paste0(file_location, 'modPredProb1.csv'))
  saveRDS(cmList, paste0(file_location, 'cmList1'))
  write.csv(aucList, paste0(file_location, 'aucList1.csv'))
  saveRDS(pltList, paste0(file_location, 'pltList1'))
  saveRDS(bestParamList, paste0(file_location, 'bestParamList1'))
  
  result.comparison <- c()
  i <- 1
  for (cm in cmList) {
    result.comparison <- cbind(result.comparison, c(cm$overall,aucList[i],cm$byClass))
    i <- i+1
  }
  rownames(result.comparison) <- c(names(cm$overall),'AUC',names(cm$byClass))
  colnames(result.comparison) <- names(modList)
  result.comparison <- as.data.frame(result.comparison)
  write.csv(result.comparison, paste0(file_location, 'result_comparison.csv'))
  
  # model selection 
  # 1. choose the best three overall-maximizing (Kappa, Accuracy, balanced accuracy,F1, AUC)
  # 2. choose the best three False Positive Rate-minimizing (Sensitivity, Negative Prediction Rate)
  # 3. choose the best three False Negative Rate-minimizing (Specificity, Positive Prediction Rate)
  rownames1 <- c('Accuracy', 'Kappa', 'F1', 'Balanced Accuracy', 'AUC')
  rownames2 <- c('Sensitivity','Neg Pred Value')
  rownames3 <- c('Specificity', 'Pos Pred Value')
  
  (avgOverall <- apply(result.comparison[rownames1,],MARGIN=2, FUN=geoMean))
  (avgFPR <- apply(result.comparison[rownames2,],MARGIN=2, FUN=geoMean))
  (avgFNR <-  apply(result.comparison[rownames3,],MARGIN=2, FUN=geoMean))
  
  topOverall <- names(head(sort(avgOverall,decreasing=T),3))
  topminFPR <- names(head(sort(avgFPR,decreasing=T),3))
  topminFNR <- names(head(sort(avgFNR,decreasing=T),3))
  
  print(paste('top three best overall model are ', toString(topOverall)))
  print(paste0('top three FPR minimizing model are ', toString(topminFPR)))
  print(paste0('top three FNR minimizing model are ', toString(topminFNR)))
  
  # per each comparison, print each model's ROC curve, lift chart, calibration
  
  # Calibration Curve
  # how do I know if I can trust them as probabilities?
  # It says on Sunday, there's an 80% chance of rain. How trustworthy is this 80% call? If I dig into weather.com's past forecasts and found that 8 out 10 days are rainy when they called an 80%, then I can convince myself to load up my audiobooks and prepare for crazy traffic on the highway in the afternoon.
  # A probabilistic model is calibrated if I binned the test samples based on their predicted probabilities, each bin's true outcomes has a proportion close to the probabilities in the bin
  # For each bin, the y-value is the proportion of true outcomes, and x-value is the mean predicted probability. Therefore, a well-calibrated model has a calibration curve that hugs the straight line y=x. 
  # - divide into bins
  # - for each bin, compute the observed event rate
  # - good probabilities: 45 degree line; below the line undercalibrated, above the line overcalibrated
  
  testProbOverall <- data.frame(cbind(test[,1], modPredProb[,topOverall]))
  testProbFPR <- data.frame(cbind(test[,1], modPredProb[,topminFPR]))
  testProbFNR <- data.frame(cbind(test[,1], modPredProb[,topminFNR]))
  
  calCurveOverall <- calibration(factor(testProbOverall[,1]) ~ testProbOverall[,2] + testProbOverall[,3] + testProbOverall[,4])
  xyplot(calCurveOverall,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbOverall[,2:4]))
  )
  calCurveFPR <- calibration(factor(testProbFPR[,1]) ~ testProbFPR[,2] + testProbFPR[,3] + testProbFPR[,4])
  xyplot(calCurveFPR,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbFPR[,2:4]))
  )
  calCurveFNR <- calibration(factor(testProbFNR[,1]) ~ testProbFNR[,2] + testProbFNR[,3] + testProbFNR[,4])
  xyplot(calCurveFNR,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbFNR[,2:4]))
  )
  
  # lift curve
  # comes from gains chart: gains, the specific name is cumulative gains chart 
  # comes from marketing and best intuitively explained in terms of marketing
  # x axis = percentage of customer base to target 
  # y axis = percentage of all positive response customers have been found in the targeted sample.
  # y-axis = sensitivity
  # x-axis = proportion of positive predictions
  liftCurveOverall <- lift(factor(testProbOverall[,1]) ~ testProbOverall[,2] + testProbOverall[,3] + testProbOverall[,4])
  xyplot(liftCurveOverall,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbOverall[,2:4]))
  )
  
  liftCurveFPR <- lift(factor(testProbFPR[,1]) ~ testProbFPR[,2] + testProbFPR[,3] + testProbFPR[,4])
  xyplot(liftCurveFPR,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbFPR[,2:4]))
  )
  
  liftCurveFNR <- lift(factor(testProbFNR[,1]) ~ testProbFNR[,2] + testProbFNR[,3] + testProbFNR[,4])
  xyplot(liftCurveFNR,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbFNR[,2:4]))
  )
  
  # ROC
  # like gains chart, it has sensitivity as y-axis
  # Model B is more discriminative than A, because it is easier to make decisions (hiking/no hiking) based on model B's outputs.
  # x-axis: specificity 
  
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,2],
                              levels=levels(test[,1])), 
                          col='blue', 
                          main='Overall Top') 
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,3],
                              levels=levels(test[,1])), 
                          col='red',
                          add=T)
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,4],
                              levels=levels(test[,1])), 
                          col='green', 
                          add=T)
  legend('bottomright',legend=colnames(testProbOverall[,2:4]), fill=c('blue','red','green'))
  
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,2],
                          levels=levels(test[,1])), 
                      col='blue', 
                      main='MIN FPR Top') 
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,3],
                          levels=levels(test[,1])), 
                      col='red',
                      add=T)
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,4],
                          levels=levels(test[,1])), 
                      col='green', 
                      add=T)
  legend('bottomright',legend=colnames(testProbFPR[,2:4]), fill=c('blue','red','green'))
  
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbOverall[,2],
                          levels=levels(test[,1])), 
                      col='blue', 
                      main='MIN FNR Top') 
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbFNR[,3],
                          levels=levels(test[,1])), 
                      col='red',
                      add=T)
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbFNR[,4],
                          levels=levels(test[,1])), 
                      col='green', 
                      add=T)
  legend('bottomright',legend=colnames(testProbFNR[,2:4]), fill=c('blue','red','green'))
  
}

binaryClassification2 <- function(data, file_location) {
  # Models Used

  # Kernel Methods
  # 1. Support Vector Machine with Linear Kernel 'svmLinear'
  # 2. Support Vector Machine with Polynomial Kernel 'svmPoly'
  # 3. Support Vector Machine with Radial Basis Function Kernel 'svmRadial'
  
  library(caret)
  library(caretEnsemble)
  library(MLmetrics)
  library(mlbench)
  library(pROC)
  library(gbm)
  library(earth)
  library(mda)
  library(MASS)
  library(rpart)
  library(randomForest)
  library(adabag)
  library(dplyr)
  library(glmnet)
  library(TH.data)
  library(party)
  library(AppliedPredictiveModeling)
  library(caTools)
  library(EnvStats) # I will use geometric mean for this metrics
  set.seed(1)
  # create a train and test set
  # data <- email_data
  trainIndex <- createDataPartition(y = data[,1], p = 0.7, list=F)
  train <- data[trainIndex,]
  test <- data[-trainIndex,]
  
  # this is the list of classification models used
  modNameList <- c('svmLinear','svmPoly','svmRadial')
  
  # define a resampling variable
  ctrl <- trainControl(method='cv',
                       # 10-fold CV
                       number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                       summaryFunction = twoClassSummary, # gives classification statistics
                       classProbs = T,
                       allowParallel=T)
  # this is for algorithm without probabilities
  ctrl2 <- trainControl(method='cv',
                        # 10-fold CV
                        number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                        summaryFunction = twoClassSummary, # gives classification statistics
                        classProbs = F,
                        allowParallel=T)
  
  n <- nrow(data) # number of rows in data
  p <- ncol(data)-1 # number of predictors
  
  modParamList <- list(
    # support vector machine with linear kernel 'svmLinear'
    svmLinear = caretModelSpec(method='svmLinear', tuneGrid=expand.grid(C=2^seq(from=-4,to=4,by=1))),
    # support vector machine polynomial basis function 'svmPoly'
    svmPoly = caretModelSpec(method='svmPoly', tuneGrid=expand.grid(degree=1:3, 
                                                                    scale=0.1, 
                                                                    C=2^seq(from=-4,to=4,by=1))),
    # support vector machine radial basis function kernel 'svmRadial'
    svmRadial = caretModelSpec(method='svmRadial',tuneGrid=expand.grid(sigma=c(0.01,0.05,0.1,0.5,1,2,5,10), 
                                                                       C=2^seq(from=-4,to=4,by=1))) # sigma=Sigma, C = cost
    
  )
  
  # https://stackoverflow.com/questions/53635127/dynamic-variable-names-in-r-regressions
  response <- noquote(colnames(data)[1])
  formula <- formula(paste0(response, "~", "."))
  
  # initialize with empty list for 1) model 2) confusion matrix 3) auc 4) variable importance plot
  modList <- list() 
  modList <- caretList(formula,
                       train,
                       trControl = ctrl,
                       tuneList = modParamList,
                       continue_on_fail = T,
                       preProcess = c('center','scale'))
  # file_location <- 'C:/Users/jalee/Desktop/Email_Model/'
  saveRDS(modList, paste0(file_location, 'classification_models2'))
  # readRDS('C:/Users/jalee/Desktop/Email_Model/classification_models')
  
  i <- 1
  modPred <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  modPredProb <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  colnames(modPred) <- names(modList)
  colnames(modPredProb) <- names(modList)
  cmList <- list()
  aucList <- c()
  pltList <- list()
  bestParamList <- list()
  for (mod  in modList) {
    # positive = 1 = first level, 2= second level; whatever is alphabetically the first is the first level, and positive
    # modPred[,i] <- ifelse(predict(mod, newdata=test)==1,levels(test[,1])[1],levels(test[,1])[2])
    modPred[,i] <- predict(mod, newdata=test) # output 1, 2
    modPredProb[,i] <- predict(mod, newdata=test, type='prob')[,1]
    # confusionMatrix requires factor for both
    cmList[[i]] <- confusionMatrix(data=factor(ifelse(modPred[,i]==1,levels(test[,1])[1],levels(test[,1])[2])),
                                   reference=factor(test[,1]))
    # aucList requires numeric values
    aucList[i] <- roc(ifelse(test[,1]==levels(test[,1])[1],1,2),
                      modPred[,i])$auc
    pltList[[i]] <- plot(varImp(mod,scale=F))
    bestParamList[[i]] <- mod$bestTune
    i <- i + 1
  }
  
  # model_preds <- lapply(model_list, predict, newdata=testing, type="prob")
  # model_preds <- lapply(model_preds, function(x) x[,"M"])
  # model_preds <- data.frame(model_preds)
  
  write.csv(modPred, paste0(file_location, 'modPred2.csv'))
  write.csv(modPredProb, paste0(file_location, 'modPredProb2.csv'))
  saveRDS(cmList, paste0(file_location, 'cmList2'))
  write.csv(aucList, paste0(file_location, 'aucList2.csv'))
  saveRDS(pltList, paste0(file_location, 'pltList2'))
  saveRDS(bestParamList, paste0(file_location, 'bestParamList2'))
  
  result.comparison <- c()
  i <- 1
  for (cm in cmList) {
    result.comparison <- cbind(result.comparison, c(cm$overall,aucList[i],cm$byClass))
    i <- i+1
  }
  rownames(result.comparison) <- c(names(cm$overall),'AUC',names(cm$byClass))
  colnames(result.comparison) <- names(modList)
  result.comparison <- as.data.frame(result.comparison)
  write.csv(result.comparison, paste0(file_location, 'result_comparison.csv'))
  
  # model selection 
  # 1. choose the best three overall-maximizing (Kappa, Accuracy, balanced accuracy,F1, AUC)
  # 2. choose the best three False Positive Rate-minimizing (Sensitivity, Negative Prediction Rate)
  # 3. choose the best three False Negative Rate-minimizing (Specificity, Positive Prediction Rate)
  rownames1 <- c('Accuracy', 'Kappa', 'F1', 'Balanced Accuracy', 'AUC')
  rownames2 <- c('Sensitivity','Neg Pred Value')
  rownames3 <- c('Specificity', 'Pos Pred Value')
  
  (avgOverall <- apply(result.comparison[rownames1,],MARGIN=2, FUN=geoMean))
  (avgFPR <- apply(result.comparison[rownames2,],MARGIN=2, FUN=geoMean))
  (avgFNR <-  apply(result.comparison[rownames3,],MARGIN=2, FUN=geoMean))
  
  topOverall <- names(head(sort(avgOverall,decreasing=T),3))
  topminFPR <- names(head(sort(avgFPR,decreasing=T),3))
  topminFNR <- names(head(sort(avgFNR,decreasing=T),3))
  
  print(paste('top three best overall model are ', toString(topOverall)))
  print(paste0('top three FPR minimizing model are ', toString(topminFPR)))
  print(paste0('top three FNR minimizing model are ', toString(topminFNR)))
  
  # per each comparison, print each model's ROC curve, lift chart, calibration
  
  # Calibration Curve
  # how do I know if I can trust them as probabilities?
  # It says on Sunday, there's an 80% chance of rain. How trustworthy is this 80% call? If I dig into weather.com's past forecasts and found that 8 out 10 days are rainy when they called an 80%, then I can convince myself to load up my audiobooks and prepare for crazy traffic on the highway in the afternoon.
  # A probabilistic model is calibrated if I binned the test samples based on their predicted probabilities, each bin's true outcomes has a proportion close to the probabilities in the bin
  # For each bin, the y-value is the proportion of true outcomes, and x-value is the mean predicted probability. Therefore, a well-calibrated model has a calibration curve that hugs the straight line y=x. 
  # - divide into bins
  # - for each bin, compute the observed event rate
  # - good probabilities: 45 degree line; below the line undercalibrated, above the line overcalibrated
  
  testProbOverall <- data.frame(cbind(test[,1], modPredProb[,topOverall]))
  testProbFPR <- data.frame(cbind(test[,1], modPredProb[,topminFPR]))
  testProbFNR <- data.frame(cbind(test[,1], modPredProb[,topminFNR]))
  
  calCurveOverall <- calibration(factor(testProbOverall[,1]) ~ testProbOverall[,2] + testProbOverall[,3] + testProbOverall[,4])
  xyplot(calCurveOverall,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbOverall[,2:4]))
  )
  calCurveFPR <- calibration(factor(testProbFPR[,1]) ~ testProbFPR[,2] + testProbFPR[,3] + testProbFPR[,4])
  xyplot(calCurveFPR,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbFPR[,2:4]))
  )
  calCurveFNR <- calibration(factor(testProbFNR[,1]) ~ testProbFNR[,2] + testProbFNR[,3] + testProbFNR[,4])
  xyplot(calCurveFNR,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbFNR[,2:4]))
  )
  
  # lift curve
  # comes from gains chart: gains, the specific name is cumulative gains chart 
  # comes from marketing and best intuitively explained in terms of marketing
  # x axis = percentage of customer base to target 
  # y axis = percentage of all positive response customers have been found in the targeted sample.
  # y-axis = sensitivity
  # x-axis = proportion of positive predictions
  liftCurveOverall <- lift(factor(testProbOverall[,1]) ~ testProbOverall[,2] + testProbOverall[,3] + testProbOverall[,4])
  xyplot(liftCurveOverall,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbOverall[,2:4]))
  )
  
  liftCurveFPR <- lift(factor(testProbFPR[,1]) ~ testProbFPR[,2] + testProbFPR[,3] + testProbFPR[,4])
  xyplot(liftCurveFPR,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbFPR[,2:4]))
  )
  
  liftCurveFNR <- lift(factor(testProbFNR[,1]) ~ testProbFNR[,2] + testProbFNR[,3] + testProbFNR[,4])
  xyplot(liftCurveFNR,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbFNR[,2:4]))
  )
  
  # ROC
  # like gains chart, it has sensitivity as y-axis
  # Model B is more discriminative than A, because it is easier to make decisions (hiking/no hiking) based on model B's outputs.
  # x-axis: specificity 
  
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,2],
                              levels=levels(test[,1])), 
                          col='blue', 
                          main='Overall Top') 
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,3],
                              levels=levels(test[,1])), 
                          col='red',
                          add=T)
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,4],
                              levels=levels(test[,1])), 
                          col='green', 
                          add=T)
  legend('bottomright',legend=colnames(testProbOverall[,2:4]), fill=c('blue','red','green'))
  
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,2],
                          levels=levels(test[,1])), 
                      col='blue', 
                      main='MIN FPR Top') 
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,3],
                          levels=levels(test[,1])), 
                      col='red',
                      add=T)
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,4],
                          levels=levels(test[,1])), 
                      col='green', 
                      add=T)
  legend('bottomright',legend=colnames(testProbFPR[,2:4]), fill=c('blue','red','green'))
  
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbOverall[,2],
                          levels=levels(test[,1])), 
                      col='blue', 
                      main='MIN FNR Top') 
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbFNR[,3],
                          levels=levels(test[,1])), 
                      col='red',
                      add=T)
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbFNR[,4],
                          levels=levels(test[,1])), 
                      col='green', 
                      add=T)
  legend('bottomright',legend=colnames(testProbFNR[,2:4]), fill=c('blue','red','green'))
  
}

binaryClassification3 <- function(data, file_location) {
  # Models Used

  # Tree Method
  # 1. Bagged Trees 'treebag'
  # 2. Random Forest 'rf'
  # 3. Oblique Random Forest 'ORFlog'
  # 4. Conditional Random Forest 'cforest'
  # 5. Parallel Random Forest 'parRF'
  # 6. Regularized Random Forest 'RRF'
  
  library(caret)
  library(caretEnsemble)
  library(MLmetrics)
  library(mlbench)
  library(pROC)
  library(gbm)
  library(earth)
  library(mda)
  library(MASS)
  library(rpart)
  library(randomForest)
  library(adabag)
  library(dplyr)
  library(glmnet)
  library(TH.data)
  library(party)
  library(AppliedPredictiveModeling)
  library(caTools)
  library(EnvStats) # I will use geometric mean for this metrics
  set.seed(1)
  # create a train and test set
  # data <- email_data
  trainIndex <- createDataPartition(y = data[,1], p = 0.7, list=F)
  train <- data[trainIndex,]
  test <- data[-trainIndex,]
  
  # this is the list of classification models used
  modNameList <- c('treebag',
                   'rf','ORFlog', 'cforest','parRF','RRF')
  
  # define a resampling variable
  ctrl <- trainControl(method='cv',
                       # 10-fold CV
                       number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                       summaryFunction = twoClassSummary, # gives classification statistics
                       classProbs = T,
                       allowParallel=T)
  # this is for algorithm without probabilities
  ctrl2 <- trainControl(method='cv',
                        # 10-fold CV
                        number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                        summaryFunction = twoClassSummary, # gives classification statistics
                        classProbs = F,
                        allowParallel=T)
  
  n <- nrow(data) # number of rows in data
  p <- ncol(data)-1 # number of predictors
  
  modParamList <- list(

    # bagged trees 'treebag'
    treebag = caretModelSpec(method='treebag',tuneGrid=c()),
    # random forest 'rf'
    rf = caretModelSpec(method='rf',tuneGrid=expand.grid(mtry=1:p)),
    # oblique random forest 'ORFlog'
    ORFlog = caretModelSpec(method='ORFlog', tuneGrid = expand.grid(mtry=1:p)),
    # conditional random forest 'cforest'
    cforest = caretModelSpec(method='cforest',tuneGrid=expand.grid(mtry=1:p)),
    # parallel random forest 'parRF'
    prfMod = caretModelSpec(method='parRF', tuneGrid=expand.grid(mtry=1:p)),
    # regularized random forest 'RRF'
    RRF = caretModelSpec(method='RRF', tuneGrid=expand.grid(mtry=1:p,
                                                            coefReg=c(0.1,0.5,1),
                                                            coefImp=T))
    
  )
  
  # https://stackoverflow.com/questions/53635127/dynamic-variable-names-in-r-regressions
  response <- noquote(colnames(data)[1])
  formula <- formula(paste0(response, "~", "."))
  
  # initialize with empty list for 1) model 2) confusion matrix 3) auc 4) variable importance plot
  modList <- list() 
  modList <- caretList(formula,
                       train,
                       trControl = ctrl,
                       tuneList = modParamList,
                       continue_on_fail = T,
                       preProcess = c('center','scale'))
  # file_location <- 'C:/Users/jalee/Desktop/Email_Model/'
  saveRDS(modList, paste0(file_location, 'classification_models3'))
  # readRDS('C:/Users/jalee/Desktop/Email_Model/classification_models')
  
  i <- 1
  modPred <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  modPredProb <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  colnames(modPred) <- names(modList)
  colnames(modPredProb) <- names(modList)
  cmList <- list()
  aucList <- c()
  pltList <- list()
  bestParamList <- list()
  for (mod  in modList) {
    # positive = 1 = first level, 2= second level; whatever is alphabetically the first is the first level, and positive
    # modPred[,i] <- ifelse(predict(mod, newdata=test)==1,levels(test[,1])[1],levels(test[,1])[2])
    modPred[,i] <- predict(mod, newdata=test) # output 1, 2
    modPredProb[,i] <- predict(mod, newdata=test, type='prob')[,1]
    # confusionMatrix requires factor for both
    cmList[[i]] <- confusionMatrix(data=factor(ifelse(modPred[,i]==1,levels(test[,1])[1],levels(test[,1])[2])),
                                   reference=factor(test[,1]))
    # aucList requires numeric values
    aucList[i] <- roc(ifelse(test[,1]==levels(test[,1])[1],1,2),
                      modPred[,i])$auc
    pltList[[i]] <- plot(varImp(mod,scale=F))
    bestParamList[[i]] <- mod$bestTune
    i <- i + 1
  }
  
  # model_preds <- lapply(model_list, predict, newdata=testing, type="prob")
  # model_preds <- lapply(model_preds, function(x) x[,"M"])
  # model_preds <- data.frame(model_preds)
  
  write.csv(modPred, paste0(file_location, 'modPred3.csv'))
  write.csv(modPredProb, paste0(file_location, 'modPredProb3.csv'))
  saveRDS(cmList, paste0(file_location, 'cmList3'))
  write.csv(aucList, paste0(file_location, 'aucList3.csv'))
  saveRDS(pltList, paste0(file_location, 'pltList3'))
  saveRDS(bestParamList, paste0(file_location, 'bestParamList3'))
  
  result.comparison <- c()
  i <- 1
  for (cm in cmList) {
    result.comparison <- cbind(result.comparison, c(cm$overall,aucList[i],cm$byClass))
    i <- i+1
  }
  rownames(result.comparison) <- c(names(cm$overall),'AUC',names(cm$byClass))
  colnames(result.comparison) <- names(modList)
  result.comparison <- as.data.frame(result.comparison)
  write.csv(result.comparison, paste0(file_location, 'result_comparison.csv'))
  
  # model selection 
  # 1. choose the best three overall-maximizing (Kappa, Accuracy, balanced accuracy,F1, AUC)
  # 2. choose the best three False Positive Rate-minimizing (Sensitivity, Negative Prediction Rate)
  # 3. choose the best three False Negative Rate-minimizing (Specificity, Positive Prediction Rate)
  rownames1 <- c('Accuracy', 'Kappa', 'F1', 'Balanced Accuracy', 'AUC')
  rownames2 <- c('Sensitivity','Neg Pred Value')
  rownames3 <- c('Specificity', 'Pos Pred Value')
  
  (avgOverall <- apply(result.comparison[rownames1,],MARGIN=2, FUN=geoMean))
  (avgFPR <- apply(result.comparison[rownames2,],MARGIN=2, FUN=geoMean))
  (avgFNR <-  apply(result.comparison[rownames3,],MARGIN=2, FUN=geoMean))
  
  topOverall <- names(head(sort(avgOverall,decreasing=T),3))
  topminFPR <- names(head(sort(avgFPR,decreasing=T),3))
  topminFNR <- names(head(sort(avgFNR,decreasing=T),3))
  
  print(paste('top three best overall model are ', toString(topOverall)))
  print(paste0('top three FPR minimizing model are ', toString(topminFPR)))
  print(paste0('top three FNR minimizing model are ', toString(topminFNR)))
  
  # per each comparison, print each model's ROC curve, lift chart, calibration
  
  # Calibration Curve
  # how do I know if I can trust them as probabilities?
  # It says on Sunday, there's an 80% chance of rain. How trustworthy is this 80% call? If I dig into weather.com's past forecasts and found that 8 out 10 days are rainy when they called an 80%, then I can convince myself to load up my audiobooks and prepare for crazy traffic on the highway in the afternoon.
  # A probabilistic model is calibrated if I binned the test samples based on their predicted probabilities, each bin's true outcomes has a proportion close to the probabilities in the bin
  # For each bin, the y-value is the proportion of true outcomes, and x-value is the mean predicted probability. Therefore, a well-calibrated model has a calibration curve that hugs the straight line y=x. 
  # - divide into bins
  # - for each bin, compute the observed event rate
  # - good probabilities: 45 degree line; below the line undercalibrated, above the line overcalibrated
  
  testProbOverall <- data.frame(cbind(test[,1], modPredProb[,topOverall]))
  testProbFPR <- data.frame(cbind(test[,1], modPredProb[,topminFPR]))
  testProbFNR <- data.frame(cbind(test[,1], modPredProb[,topminFNR]))
  
  calCurveOverall <- calibration(factor(testProbOverall[,1]) ~ testProbOverall[,2] + testProbOverall[,3] + testProbOverall[,4])
  xyplot(calCurveOverall,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbOverall[,2:4]))
  )
  calCurveFPR <- calibration(factor(testProbFPR[,1]) ~ testProbFPR[,2] + testProbFPR[,3] + testProbFPR[,4])
  xyplot(calCurveFPR,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbFPR[,2:4]))
  )
  calCurveFNR <- calibration(factor(testProbFNR[,1]) ~ testProbFNR[,2] + testProbFNR[,3] + testProbFNR[,4])
  xyplot(calCurveFNR,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbFNR[,2:4]))
  )
  
  # lift curve
  # comes from gains chart: gains, the specific name is cumulative gains chart 
  # comes from marketing and best intuitively explained in terms of marketing
  # x axis = percentage of customer base to target 
  # y axis = percentage of all positive response customers have been found in the targeted sample.
  # y-axis = sensitivity
  # x-axis = proportion of positive predictions
  liftCurveOverall <- lift(factor(testProbOverall[,1]) ~ testProbOverall[,2] + testProbOverall[,3] + testProbOverall[,4])
  xyplot(liftCurveOverall,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbOverall[,2:4]))
  )
  
  liftCurveFPR <- lift(factor(testProbFPR[,1]) ~ testProbFPR[,2] + testProbFPR[,3] + testProbFPR[,4])
  xyplot(liftCurveFPR,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbFPR[,2:4]))
  )
  
  liftCurveFNR <- lift(factor(testProbFNR[,1]) ~ testProbFNR[,2] + testProbFNR[,3] + testProbFNR[,4])
  xyplot(liftCurveFNR,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbFNR[,2:4]))
  )
  
  # ROC
  # like gains chart, it has sensitivity as y-axis
  # Model B is more discriminative than A, because it is easier to make decisions (hiking/no hiking) based on model B's outputs.
  # x-axis: specificity 
  
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,2],
                              levels=levels(test[,1])), 
                          col='blue', 
                          main='Overall Top') 
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,3],
                              levels=levels(test[,1])), 
                          col='red',
                          add=T)
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,4],
                              levels=levels(test[,1])), 
                          col='green', 
                          add=T)
  legend('bottomright',legend=colnames(testProbOverall[,2:4]), fill=c('blue','red','green'))
  
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,2],
                          levels=levels(test[,1])), 
                      col='blue', 
                      main='MIN FPR Top') 
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,3],
                          levels=levels(test[,1])), 
                      col='red',
                      add=T)
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,4],
                          levels=levels(test[,1])), 
                      col='green', 
                      add=T)
  legend('bottomright',legend=colnames(testProbFPR[,2:4]), fill=c('blue','red','green'))
  
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbOverall[,2],
                          levels=levels(test[,1])), 
                      col='blue', 
                      main='MIN FNR Top') 
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbFNR[,3],
                          levels=levels(test[,1])), 
                      col='red',
                      add=T)
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbFNR[,4],
                          levels=levels(test[,1])), 
                      col='green', 
                      add=T)
  legend('bottomright',legend=colnames(testProbFNR[,2:4]), fill=c('blue','red','green'))
  
}

binaryClassification4 <- function(data, file_location) {
  # Models Used

  # 7. Stochastic Gradient Boosting 'gbm'
  # 8. eXtreme Gradient Boosting 'xgbTree'
  # 9. C5.0 'C5.0'
  # 10. Bagged Adaboost
  
  library(caret)
  library(caretEnsemble)
  library(MLmetrics)
  library(mlbench)
  library(pROC)
  library(gbm)
  library(earth)
  library(mda)
  library(MASS)
  library(rpart)
  library(randomForest)
  library(adabag)
  library(dplyr)
  library(glmnet)
  library(TH.data)
  library(party)
  library(AppliedPredictiveModeling)
  library(caTools)
  library(EnvStats) # I will use geometric mean for this metrics
  set.seed(1)
  # create a train and test set
  # data <- email_data
  trainIndex <- createDataPartition(y = data[,1], p = 0.7, list=F)
  train <- data[trainIndex,]
  test <- data[-trainIndex,]
  
  # this is the list of classification models used
  modNameList <- c('gbm','xgbTree','C5.0','adabag')
  
  # define a resampling variable
  ctrl <- trainControl(method='cv',
                       # 10-fold CV
                       number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                       summaryFunction = twoClassSummary, # gives classification statistics
                       classProbs = T,
                       allowParallel=T)
  # this is for algorithm without probabilities
  ctrl2 <- trainControl(method='cv',
                        # 10-fold CV
                        number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                        summaryFunction = twoClassSummary, # gives classification statistics
                        classProbs = F,
                        allowParallel=T)
  
  n <- nrow(data) # number of rows in data
  p <- ncol(data)-1 # number of predictors
  
  modParamList <- list(
    # gradient boosting machine 'gbm'
    gbm = caretModelSpec(method='gbm',
                         tuneGrid= expand.grid(n.trees=500, 
                                               interaction.depth=seq(1,p,by=1), 
                                               shrinkage=seq(0.01,0.1,by=0.01),
                                               n.minobsinnode=1:p)),
    # extreme Gradient boosting 'xgbTree'
    xgbTree = caretModelSpec(method='xgbTree',
                             tuneGrid=expand.grid(nrounds=c(2,10),
                                                  max_depth=c(5,10,15),
                                                  eta=c(0.01,0.1),
                                                  gamma=0:3,
                                                  colsample_bytree=c(0.4,0.7,1),
                                                  min_child_weight=c(0.5,1,1.5),
                                                  subsample=1)),
    # C5.0 'C5.0'
    C5.0 = caretModelSpec(method='C5.0',
                          tuneGrid=expand.grid(trials=c(10,20), # boosting iterations
                                               model=c('tree','rules'), # model type
                                               winnow=c(T,F))), # winnow 
    # bagged adaboost 'AdaBag'
    AdaBag = caretModelSpec(method='AdaBag',
                            tuneGrid = expand.grid(mfinal=500,
                                                   maxdepth=seq(2,p,by=2)))
  )
  
  # https://stackoverflow.com/questions/53635127/dynamic-variable-names-in-r-regressions
  response <- noquote(colnames(data)[1])
  formula <- formula(paste0(response, "~", "."))
  
  # initialize with empty list for 1) model 2) confusion matrix 3) auc 4) variable importance plot
  modList <- list() 
  modList <- caretList(formula,
                       train,
                       trControl = ctrl,
                       tuneList = modParamList,
                       continue_on_fail = T,
                       preProcess = c('center','scale'))
  # file_location <- 'C:/Users/jalee/Desktop/Email_Model/'
  saveRDS(modList, paste0(file_location, 'classification_models4'))
  # readRDS('C:/Users/jalee/Desktop/Email_Model/classification_models')
  
  i <- 1
  modPred <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  modPredProb <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  colnames(modPred) <- names(modList)
  colnames(modPredProb) <- names(modList)
  cmList <- list()
  aucList <- c()
  pltList <- list()
  bestParamList <- list()
  for (mod  in modList) {
    # positive = 1 = first level, 2= second level; whatever is alphabetically the first is the first level, and positive
    # modPred[,i] <- ifelse(predict(mod, newdata=test)==1,levels(test[,1])[1],levels(test[,1])[2])
    modPred[,i] <- predict(mod, newdata=test) # output 1, 2
    modPredProb[,i] <- predict(mod, newdata=test, type='prob')[,1]
    # confusionMatrix requires factor for both
    cmList[[i]] <- confusionMatrix(data=factor(ifelse(modPred[,i]==1,levels(test[,1])[1],levels(test[,1])[2])),
                                   reference=factor(test[,1]))
    # aucList requires numeric values
    aucList[i] <- roc(ifelse(test[,1]==levels(test[,1])[1],1,2),
                      modPred[,i])$auc
    pltList[[i]] <- plot(varImp(mod,scale=F))
    bestParamList[[i]] <- mod$bestTune
    i <- i + 1
  }
  
  # model_preds <- lapply(model_list, predict, newdata=testing, type="prob")
  # model_preds <- lapply(model_preds, function(x) x[,"M"])
  # model_preds <- data.frame(model_preds)
  
  write.csv(modPred, paste0(file_location, 'modPred4.csv'))
  write.csv(modPredProb, paste0(file_location, 'modPredProb4.csv'))
  saveRDS(cmList, paste0(file_location, 'cmList4'))
  write.csv(aucList, paste0(file_location, 'aucList4.csv'))
  saveRDS(pltList, paste0(file_location, 'pltList4'))
  saveRDS(bestParamList, paste0(file_location, 'bestParamList4'))
  
  result.comparison <- c()
  i <- 1
  for (cm in cmList) {
    result.comparison <- cbind(result.comparison, c(cm$overall,aucList[i],cm$byClass))
    i <- i+1
  }
  rownames(result.comparison) <- c(names(cm$overall),'AUC',names(cm$byClass))
  colnames(result.comparison) <- names(modList)
  result.comparison <- as.data.frame(result.comparison)
  write.csv(result.comparison, paste0(file_location, 'result_comparison.csv'))
  
  # model selection 
  # 1. choose the best three overall-maximizing (Kappa, Accuracy, balanced accuracy,F1, AUC)
  # 2. choose the best three False Positive Rate-minimizing (Sensitivity, Negative Prediction Rate)
  # 3. choose the best three False Negative Rate-minimizing (Specificity, Positive Prediction Rate)
  rownames1 <- c('Accuracy', 'Kappa', 'F1', 'Balanced Accuracy', 'AUC')
  rownames2 <- c('Sensitivity','Neg Pred Value')
  rownames3 <- c('Specificity', 'Pos Pred Value')
  
  (avgOverall <- apply(result.comparison[rownames1,],MARGIN=2, FUN=geoMean))
  (avgFPR <- apply(result.comparison[rownames2,],MARGIN=2, FUN=geoMean))
  (avgFNR <-  apply(result.comparison[rownames3,],MARGIN=2, FUN=geoMean))
  
  topOverall <- names(head(sort(avgOverall,decreasing=T),3))
  topminFPR <- names(head(sort(avgFPR,decreasing=T),3))
  topminFNR <- names(head(sort(avgFNR,decreasing=T),3))
  
  print(paste('top three best overall model are ', toString(topOverall)))
  print(paste0('top three FPR minimizing model are ', toString(topminFPR)))
  print(paste0('top three FNR minimizing model are ', toString(topminFNR)))
  
  # per each comparison, print each model's ROC curve, lift chart, calibration
  
  # Calibration Curve
  # how do I know if I can trust them as probabilities?
  # It says on Sunday, there's an 80% chance of rain. How trustworthy is this 80% call? If I dig into weather.com's past forecasts and found that 8 out 10 days are rainy when they called an 80%, then I can convince myself to load up my audiobooks and prepare for crazy traffic on the highway in the afternoon.
  # A probabilistic model is calibrated if I binned the test samples based on their predicted probabilities, each bin's true outcomes has a proportion close to the probabilities in the bin
  # For each bin, the y-value is the proportion of true outcomes, and x-value is the mean predicted probability. Therefore, a well-calibrated model has a calibration curve that hugs the straight line y=x. 
  # - divide into bins
  # - for each bin, compute the observed event rate
  # - good probabilities: 45 degree line; below the line undercalibrated, above the line overcalibrated
  
  testProbOverall <- data.frame(cbind(test[,1], modPredProb[,topOverall]))
  testProbFPR <- data.frame(cbind(test[,1], modPredProb[,topminFPR]))
  testProbFNR <- data.frame(cbind(test[,1], modPredProb[,topminFNR]))
  
  calCurveOverall <- calibration(factor(testProbOverall[,1]) ~ testProbOverall[,2] + testProbOverall[,3] + testProbOverall[,4])
  xyplot(calCurveOverall,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbOverall[,2:4]))
  )
  calCurveFPR <- calibration(factor(testProbFPR[,1]) ~ testProbFPR[,2] + testProbFPR[,3] + testProbFPR[,4])
  xyplot(calCurveFPR,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbFPR[,2:4]))
  )
  calCurveFNR <- calibration(factor(testProbFNR[,1]) ~ testProbFNR[,2] + testProbFNR[,3] + testProbFNR[,4])
  xyplot(calCurveFNR,
         auto.key=list(x=0.01,y=0.99,
                       text=colnames(testProbFNR[,2:4]))
  )
  
  # lift curve
  # comes from gains chart: gains, the specific name is cumulative gains chart 
  # comes from marketing and best intuitively explained in terms of marketing
  # x axis = percentage of customer base to target 
  # y axis = percentage of all positive response customers have been found in the targeted sample.
  # y-axis = sensitivity
  # x-axis = proportion of positive predictions
  liftCurveOverall <- lift(factor(testProbOverall[,1]) ~ testProbOverall[,2] + testProbOverall[,3] + testProbOverall[,4])
  xyplot(liftCurveOverall,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbOverall[,2:4]))
  )
  
  liftCurveFPR <- lift(factor(testProbFPR[,1]) ~ testProbFPR[,2] + testProbFPR[,3] + testProbFPR[,4])
  xyplot(liftCurveFPR,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbFPR[,2:4]))
  )
  
  liftCurveFNR <- lift(factor(testProbFNR[,1]) ~ testProbFNR[,2] + testProbFNR[,3] + testProbFNR[,4])
  xyplot(liftCurveFNR,
         auto.key=list(x=0.69,y=0.15,
                       text=colnames(testProbFNR[,2:4]))
  )
  
  # ROC
  # like gains chart, it has sensitivity as y-axis
  # Model B is more discriminative than A, because it is easier to make decisions (hiking/no hiking) based on model B's outputs.
  # x-axis: specificity 
  
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,2],
                              levels=levels(test[,1])), 
                          col='blue', 
                          main='Overall Top') 
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,3],
                              levels=levels(test[,1])), 
                          col='red',
                          add=T)
  rocCurveOverall <- plot(roc(response=test[,1],
                              predictor=testProbOverall[,4],
                              levels=levels(test[,1])), 
                          col='green', 
                          add=T)
  legend('bottomright',legend=colnames(testProbOverall[,2:4]), fill=c('blue','red','green'))
  
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,2],
                          levels=levels(test[,1])), 
                      col='blue', 
                      main='MIN FPR Top') 
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,3],
                          levels=levels(test[,1])), 
                      col='red',
                      add=T)
  rocCurveFPR <- plot(roc(response=test[,1],
                          predictor=testProbFPR[,4],
                          levels=levels(test[,1])), 
                      col='green', 
                      add=T)
  legend('bottomright',legend=colnames(testProbFPR[,2:4]), fill=c('blue','red','green'))
  
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbOverall[,2],
                          levels=levels(test[,1])), 
                      col='blue', 
                      main='MIN FNR Top') 
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbFNR[,3],
                          levels=levels(test[,1])), 
                      col='red',
                      add=T)
  rocCurveFNR <- plot(roc(response=test[,1],
                          predictor=testProbFNR[,4],
                          levels=levels(test[,1])), 
                      col='green', 
                      add=T)
  legend('bottomright',legend=colnames(testProbFNR[,2:4]), fill=c('blue','red','green'))
  
}

binaryClassification5 <- function(data, file_location) {
  # Models Used

  # Neural Network
  # 1. Neural Network 'nnet'
  # 2. Model Averaged Neural Network 'avNNet'
  
  library(caret)
  library(caretEnsemble)
  library(MLmetrics)
  library(mlbench)
  library(pROC)
  library(gbm)
  library(earth)
  library(mda)
  library(MASS)
  library(rpart)
  library(randomForest)
  library(adabag)
  library(dplyr)
  library(glmnet)
  library(TH.data)
  library(party)
  library(AppliedPredictiveModeling)
  library(caTools)
  library(EnvStats) # I will use geometric mean for this metrics
  set.seed(1)
  # create a train and test set
  # data <- email_data
  trainIndex <- createDataPartition(y = data[,1], p = 0.7, list=F)
  train <- data[trainIndex,]
  test <- data[-trainIndex,]
  
  # this is the list of classification models used
  modNameList <- c('glm', 'glmnet','LogitBoost','lda','qda','sparseLDA',
                   'nbDiscrete','pam','fda','bagFDA','mda','pda','rda', 
                   'pls','knn','svmLinear','svmPoly','svmRadial','treebag',
                   'rf','ORFlog', 'cforest','parRF','RRF','gbm',
                   'xgbTree','C5.0','adabag','nnet','avNNet')
  
  # define a resampling variable
  ctrl <- trainControl(method='cv',
                       # 10-fold CV
                       number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                       summaryFunction = twoClassSummary, # gives classification statistics
                       classProbs = T,
                       allowParallel=T)
  # this is for algorithm without probabilities
  ctrl2 <- trainControl(method='cv',
                        # 10-fold CV
                        number=10, # repeat 10 times: compute average of 10 10-fold CV errors
                        summaryFunction = twoClassSummary, # gives classification statistics
                        classProbs = F,
                        allowParallel=T)
  
  n <- nrow(data) # number of rows in data
  p <- ncol(data)-1 # number of predictors
  
  modParamList <- list(

    # neural net 'nnet'
    nnet = caretModelSpec(method='nnet',
                          tuneGrid=expand.grid(size=seq(1,p,by=1), 
                                               decay=seq(0.01,2,by=0.2))),
    # model averaged neural network 'avNNet'
    avnnMod = caretModelSpec(method='avNNet',
                             tuneGrid=expand.grid(size=1:p, # number of hidden units
                                                  decay=seq(0.01,2,by=0.2), # weight decay
                                                  bag=c(T,F))) # bagging
  )
  
  # https://stackoverflow.com/questions/53635127/dynamic-variable-names-in-r-regressions
  response <- noquote(colnames(data)[1])
  formula <- formula(paste0(response, "~", "."))
  
  # initialize with empty list for 1) model 2) confusion matrix 3) auc 4) variable importance plot
  modList <- list() 
  modList <- caretList(formula,
                       train,
                       trControl = ctrl,
                       tuneList = modParamList,
                       continue_on_fail = T,
                       preProcess = c('center','scale'))
  # file_location <- 'C:/Users/jalee/Desktop/Email_Model/'
  saveRDS(modList, paste0(file_location, 'classification_models5'))
  # readRDS('C:/Users/jalee/Desktop/Email_Model/classification_models')
  
  i <- 1
  modPred <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  modPredProb <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  colnames(modPred) <- names(modList)
  colnames(modPredProb) <- names(modList)
  cmList <- list()
  aucList <- c()
  pltList <- list()
  bestParamList <- list()
  for (mod  in modList) {
    # positive = 1 = first level, 2= second level; whatever is alphabetically the first is the first level, and positive
    # modPred[,i] <- ifelse(predict(mod, newdata=test)==1,levels(test[,1])[1],levels(test[,1])[2])
    modPred[,i] <- predict(mod, newdata=test) # output 1, 2
    modPredProb[,i] <- predict(mod, newdata=test, type='prob')[,1]
    # confusionMatrix requires factor for both
    cmList[[i]] <- confusionMatrix(data=factor(ifelse(modPred[,i]==1,levels(test[,1])[1],levels(test[,1])[2])),
                                   reference=factor(test[,1]))
    # aucList requires numeric values
    aucList[i] <- roc(ifelse(test[,1]==levels(test[,1])[1],1,2),
                      modPred[,i])$auc
    pltList[[i]] <- plot(varImp(mod,scale=F))
    bestParamList[[i]] <- mod$bestTune
    i <- i + 1
  }
  
  # model_preds <- lapply(model_list, predict, newdata=testing, type="prob")
  # model_preds <- lapply(model_preds, function(x) x[,"M"])
  # model_preds <- data.frame(model_preds)
  
  write.csv(modPred, paste0(file_location, 'modPred5.csv'))
  write.csv(modPredProb, paste0(file_location, 'modPredProb5.csv'))
  saveRDS(cmList, paste0(file_location, 'cmList5'))
  write.csv(aucList, paste0(file_location, 'aucList5.csv'))
  saveRDS(pltList, paste0(file_location, 'pltList5'))
  saveRDS(bestParamList, paste0(file_location, 'bestParamList5'))
  
  result.comparison <- c()
  i <- 1
  for (cm in cmList) {
    result.comparison <- cbind(result.comparison, c(cm$overall,aucList[i],cm$byClass))
    i <- i+1
  }
  rownames(result.comparison) <- c(names(cm$overall),'AUC',names(cm$byClass))
  colnames(result.comparison) <- names(modList)
  result.comparison <- as.data.frame(result.comparison)
  write.csv(result.comparison, paste0(file_location, 'result_comparison.csv'))

  
}
