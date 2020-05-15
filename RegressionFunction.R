regressionFunction <- function(data, file_location='C:/Users/jihun/Desktop/') {
  
  # Models Used
  # Linear Regression and with Regularizations
  # 1. Linear Regression 'lm'
  # 2. Robust Linear Regression 'rlm'
  # 3. Lasso 'lasso'
  # 4. Ridge 'ridge'
  # 5. Elastic 'enet'
  # 6. Bayesian Ridge Regression 'bridge' 
  # 7. Model Averaged Bayesian Ridge Regression 'blassoAveraged'
  # 8. Spikes and Slab Regression 'spikeslab' 
  # 9. Bayesian Lasso 'blasso'
  # 10. Boosted Linear Model 'Bstlm'
  
  # Gaussian Process
  # 1. Gaussian Process 'gaussprLinear'
  # 2. Gaussian Process Polynomial 'gaussprPoly'
  # 3. Gaussian Process RBF 'gaussprRadial'
  
  # Matrix Decomposition Method
  # 1. Partial Least Squares 'pls'
  # 2. Sparse Partial Least Squaress 'spls'
  
  # Splines
  # 1. Multivariate Adaptive Regression Splines 'earth'
  # 2. Bagged MARS 'bagMARS'
  # 3. Boosted Smoothing Spline 'bstSm'
  
  # Kernel Methods
  # 1. Support Vector Machine with Linear Kernel 'svmLinear'
  # 2. Support Vector Machine with Polynomial Kernel 'svmPoly'
  # 3. Support Vector Machine with Radial Basis Function Kernel 'svmRadial'
  
  # Tree Method
  # 1. Bagged CART 'treebag'
  # 2. Random Forest 'rf'
  # 3. Conditional Random Forest 'cforest'
  # 4. Parallel Random Forest 'parRF'
  # 5. Regularized Random Forest 'RRF'
  # 6. Quantile Random Forest 'qrf'
  # 7.  Bayesian Additive Regression Tree 'bartMachine'
  # 8. Stochastic Gradient Boosting 'gbm'
  # 9. eXtreme Gradient Boosting 'xgbTree'
  # 11. Cubist 'cubist'
  
  # Neural Network
  # 1. Neural Network 'nnet'
  # 2. Model Averaged Neural Network 'avNNet'
  # 3. Neural Network with Feature Extraction 'pcaNNet'
  # 4. Quantile Regression Neural Network 'qrnn'
  # 5. Bayesian Regularized Neural Network 'brnn'
  
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
  trainIndex <- createDataPartition(y = data[,1], p = 0.75, list=F)
  train <- data[trainIndex,]
  test <- data[-trainIndex,]
  
  # define a resampling variable for cross-validation
  ctrl <- trainControl(method='cv',
                       # 10-fold CV
                       number=10 # repeat 10 times: compute average of 10 10-fold CV errors
                       )
  
  n <- nrow(data) # number of rows in data
  p <- ncol(data)-1 # number of predictors
  
  modParamList <- list(
    # Regression ------------------------------------------------
    # 1. linear regression
    lm = caretModelSpec(method='lm', tuneGrid=c()), 
    # 2. robust linear regression
    rlm = caretModelSpec(method='rlm', tuneGrid=expand.grid(alpha=seq(0,1,by=0.1), # mix percentage
                                                                      lambda=seq(0,5,by=0.5))), # regularization parameter
    # 3. lasso regression 
    lasso = caretModelSpec(method='lasso',tuneGrid = expand.grid(nIter=3000)),
    # 4. ridge regression
    ridge = caretModelSpec(method='ridge', tuneGrid=c()),
    # 5. elastic net
    enet = caretModelSpec(method='enet',tuneGrid=c()),  
    # 6. bayesian ridge regression
    bridge = caretModelSpec(method='bridge', tuneGrid=expand.grid(NumVars = 1:p, 
                                                                      lambda=c(0,0.01,0.1,1))),
    # 7. bayesian lasso
    blasso = caretModelSpec(method='blasso',tuneGrid=expand.grid(smooth=1:3)),
    # 8. Model Averaged Bayesian Ridge Regression
    blassoAveraged = caretModelSpec(method='blassoAveraged',tuneGrid=expand.grid(threshold=seq(0,3,by=0.5))),
    # 9. Boosted Linear Model 
    Bstlm = caretModelSpec(method='Bstlm', tuneGrid=expand.grid(degree=1:3, nprune=1:p)),
    # 10. Spikes and Slab Regression
    spikeslab = caretModelSpec(method='spikeslab', tuneGrid=expand.grid(degree=1:3, nprune=1:p)),
   
    # Gaussian Process ---------------------------------------------------------
    # 1. Linear Gaussian Process
    gaussprLinear = caretModelSpec(method='gaussprLinear', tuneGrid = expand.grid(subclasses=1:p)),
    # 2. Polynomial Gaussian Process
    gaussprPoly = caretModelSpec(method='gaussprPoly', tuneGrid=expand.grid(lambda=seq(0,1,by=0.1))),
    # 3. RBF Gaussian Process
    gaussprRBF = caretModelSpec(method='gaussprRBF', tuneGrid=expand.grid(gamma= seq(0,1,by=0.1),
                                                               lambda= seq(0,1,by=0.1))),
    
    # Matrix Decomposition Methods ----------------------------------------------------------
    # Partial Least Squares
    pls = caretModelSpec(method='pls',tuneGrid=expand.grid(ncomp=1:p)),
    # Sparse Partial Least Squares
    spls = caretModelSpec(method='spls', tuneGrid=expand.grid(k=seq(1,ceiling(sqrt(n)),by=2))),
    
    # Splines -------------------------------------------------------------------------------
    # Multivariate Adaptive Regression Splines
    earth = caretModelSpec(method='earth', tuneGrid=expand.grid(C=2^seq(from=-4,to=4,by=1))),
    # Bagged MARS
    bagMARS = caretModelSpec(method='bagMARS', tuneGrid=expand.grid(degree=1:3, 
                                                                    scale=0.1, 
                                                                    C=2^seq(from=-4,to=4,by=1))),
    # Boosted Smoothing Splines
    bstSm = caretModelSpec(method='bstSm',tuneGrid=expand.grid(sigma=c(0.01,0.05,0.1,0.5,1,2,5,10), 
                                                                     C=2^seq(from=-4,to=4,by=1))), # sigma=Sigma, C = cost
    
    # Kernel Methods -------------------------------------------------------------------------
    # Support Vector Machine with Linear Kernel
    svmLinear = caretModelSpec(method='svmLinear',tuneGrid=c()),
    # Support Vector Machine with Polynomial Kernel
    svmPoly = caretModelSpec(method='svmPoly',tuneGrid=expand.grid(mtry=1:p)),
    # Support Vector Machine with RBF Kernel
    svmRadial = caretModelSpec(method='svmRadial', tuneGrid = expand.grid(mtry=1:p)),
    
    # Tree Methods -----------------------------------------------------------------------------
    # Bagged CART 
    treebag = caretModelSpec(method='treebag',tuneGrid=expand.grid(mtry=1:p)),
    # Random Forest
    rf = caretModelSpec(method='rf', tuneGrid=expand.grid(mtry=1:p)),
    # Conditional Random Forest
    cforest = caretModelSpec(method='cforest', tuneGrid=expand.grid(mtry=1:p,
                                                               coefReg=c(0.1,0.5,1),
                                                               coefImp=T)),
    # Regularized Random Forest 
    RRF = caretModelSpec(method='RRF',
                            tuneGrid= expand.grid(n.trees=seq(1000,3000,by=1000), 
                                                  interaction.depth=seq(5,30,by=5), 
                                                  shrinkage=seq(0.01,0.1,by=0.01),
                                                  n.minobsinnode=1:10)),
    # Quantile Random Forest 
    qrf = caretModelSpec(method='qrf',
                             tuneGrid=expand.grid(nrounds=c(2,10),
                                                  max_depth=c(5,10,15),
                                                  eta=c(0.01,0.1),
                                                  gamma=0:3,
                                                  colsample_bytree=c(0.4,0.7,1),
                                                  min_child_weight=c(0.5,1,1.5),
                                                  subsample=1)),
    # Bayesian Additive Regression Tree 
    bartMachine = caretModelSpec(method='bartMachine',
                           tuneGrid=expand.grid(trials=c(10,20), # boosting iterations
                                                model=c('tree','rules'), # model type
                                                winnow=c(T,F))), # winnow 
    # Stochastic Gradient Boosting 
    gbm = caretModelSpec(method='gbm',
                               tuneGrid = expand.grid(mfinal=500,
                                                      maxdepth=seq(2,20,by=2))),
    # Extreme Gradient Boosting
    xgbTree = caretModelSpec(method='xgbTree',
                           tuneGrid=expand.grid(size=seq(1,9,by=2), 
                                                decay=seq(0.01,2,by=0.2))),
    # Cubist 
    cubist = caretModelSpec(method='cubist',
                             tuneGrid=expand.grid(size=1:10, # number of hidden units
                                                  decay=seq(0.01,2,by=0.2), # weight decay
                                                  bag=c(T,F))), # bagging
    # Neural Network ------------------------------------------------
    # Vanilla Neural Network
    nnet = caretModelSpec(method='nnet',
                          tuneGrid=expand.grid()),
    # Model Averaged Neural Network
    avNNet = caretModelSpec(method='avNNet',
                            tuneGrid=expand.grid()),
    # Neural Network with Feature Extraction 'pcaNNet'
    pcaNNet = caretModelSpec(method='pcaNNet',
                             tuneGrid=expand.grid()),
    # Quantile Regression Neural Network
    qrnn = caretModelSpec(method='qrnn',
                          tuneGrid=expand.grid()),
    # Bayesian Regularized Neural Network
    brnn = caretModelSpec(method='brnn',
                          tuneGrid=expand.grid())
  )

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
  # file_location <- 'C:/Users/jihun/Desktop/'
  saveRDS(modList, paste0(file_location, 'regression_models'))
  
  # Initialize prediction matrices
  i <- 1
  modPred <- matrix(data=0, nrow=nrow(test), ncol=length(modList))
  colnames(modPred) <- names(modList)
  # Initialize best parameter and plot matrices
  metricList <- list()
  pltList <- list()
  bestParamList <- list()
  for (mod  in modList) {
    modPred[,i] <- predict(mod, newdata=test) 
    metricList[[i]] <- postResample(pred=modpred[,i], obs=test)
    pltList[[i]] <- plot(varImp(mod,scale=F))
    bestParamList[[i]] <- mod$bestTune
    i <- i + 1
  }
  
  write.csv(modPred, paste0(file_location, 'modPred.csv'))
  saveRDS(metricList, paste0(file_location, 'metricList'))
  saveRDS(pltList, paste0(file_location, 'pltList'))
  saveRDS(bestParamList, paste0(file_location, 'bestParamList'))
  
  result.comparison <- c()
  for (metric in metricList) {
    result.comparison <- cbind(result.comparison,metric)
  }
  rownames(result.comparison) <- names(postResample(metric))
  colnames(result.comparison) <- names(modList)
  result.comparison <- as.data.frame(result.comparison)
  write.csv(result.comparison, paste0(file_location, 'result_comparison.csv'))
  
  # model selection 
  
  topRMSE <- names(head(sort(result.comparison[1,],decreasing=T),3))
  topRsquared <- names(head(sort(result.comparison[2,],decreasing=T),3))
  topMAE <- names(head(sort(result.comparison[3,],decreasing=T),3))
  
  print(paste('top three models in RMSE are', toString(topRMSE)))
  print(paste0('top three models in Rsquared are ', toString(topRsquared)))
  print(paste0('top three models in MAE are ', toString(topMAE)))
  
}  

# caret ensemble
# caret stacking

# feature selection
# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/

# ---------------------------------
# Create a blank workbook containing:
# 1) summary result of performance
# 2) predictions
# 3) each model: variable importance, summary
# 4) 
OUT <- createWorkbook()
m <- length(modList)
i <- 1
for (mod in modList) {
  addWorksheet(OUT, toString(mod))
  writeData(OUT, sheet = toString(mod), x = pltList[[i]])
  i <- i + 1
}

# --------------------------------------------

# Create a blank workbook
OUT <- createWorkbook()

# Add some sheets to the workbook
addWorksheet(OUT, "Sheet 1 Name")
addWorksheet(OUT, "Sheet 2 Name")

# Write the data to the sheets
writeData(OUT, sheet = "Sheet 1 Name", x = dataframe1)
writeData(OUT, sheet = "Sheet 2 Name", x = dataframe2)

# Reorder worksheets
worksheetOrder(OUT) <- c(2,1)

# Export the file
saveWorkbook(OUT, "My output file.xlsx")

# ------------------------------------
wb = createWorkbook()

sheet = createSheet(wb, "Sheet 1")

addDataFrame(dataframe1, sheet=sheet, startColumn=1, row.names=FALSE)
addDataFrame(dataframe2, sheet=sheet, startColumn=10, row.names=FALSE)

sheet = createSheet(wb, "Sheet 2")

addDataFrame(dataframe3, sheet=sheet, startColumn=1, row.names=FALSE)

saveWorkbook(wb, "My_File.xlsx")



library(xlsx)
write.xlsx(dataframe1, file="filename.xlsx", sheetName="sheet1", row.names=FALSE)
write.xlsx(dataframe2, file="filename.xlsx", sheetName="sheet2", append=TRUE, row.names=FALSE)

write.xlsx(df$sheet1, file = "myfile.xlsx", sheetName="sh1", append=TRUE)
write.xlsx(df$sheet2, file = "myfile.xlsx", sheetName="sh2", append=TRUE)
write.xlsx(df$sheet3, file = "myfile.xlsx", sheetName="sh3", append=TRUE)
# https://cran.r-project.org/web/packages/stargazer/vignettes/stargazer.pdf

require(openxlsx)
list_of_datasets <- list("Name of DataSheet1" = dataframe1, "Name of Datasheet2" = dataframe2)
write.xlsx(list_of_datasets, file = "writeXLSX2.xlsx")





caret(library)
ctrl <- safsControl(functions = rfSA,
                    method = "cv",
                    number = 10)

rf_search <- safs(x = train_data[, -ncol(train_data)],
                  y = train_data$Class,
                  iters = 3,
                  safsControl = ctrl)

many_stats <-
  function(data, lev = levels(data$obs), model = NULL) {
    c(
      twoClassSummary(data = data, lev = levels(data$obs), model),
      prSummary(data = data, lev = levels(data$obs), model),
      mnLogLoss(data = data, lev = levels(data$obs), model),
      defaultSummary(data = data, lev = levels(data$obs), model)
    )
  }

library(caret)
library(tidymodels)
library(doParallel)

sa_funcs <- caretSA
sa_funcs$fitness_extern <- many_stats
sa_funcs$initial <- function(vars, prob = 0.10, ...) 
  sort(sample.int(vars, size = floor(vars * prob) + 1))

nb_grid <- data.frame(usekernel = TRUE, fL = 0, adjust = 1)

# Inner control for each model, use a random 10% validation set
ctrl_rs <- trainControl(
  method = "LGOCV", 
  p = 0.90,
  number = 1,
  summaryFunction = many_stats, 
  classProbs = TRUE,
  allowParallel = FALSE
)

# Outer control for SA
sa_ctrl <- safsControl(
  method = "cv",
  metric = c(internal = "ROC", external = "ROC"),
  maximize = c(internal = TRUE, external = TRUE), 
  functions = sa_funcs, 
  improve = 10,
  returnResamp = "all",
  verbose = TRUE
)

options(digits = 3)

# Run simulated annealing
set.seed(325)
sim_anneal_10_pct <- safs(
  okc_rec,
  data = okc_train,
  iters = 500,
  safsControl = sa_ctrl,
  method = "nb",
  tuneGrid = nb_grid,
  trControl = ctrl_rs,
  metric = "ROC"
)