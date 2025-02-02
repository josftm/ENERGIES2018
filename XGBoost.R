library(rpart.plot)
require(reshape)
library(caret.ts)
library(forecast)
library("TSPred")
library("gbm")
library("mlbench")
library(readr)
#library(ranger)
#library(quantregForest)

XGBoost<-function(numProb,nameData,w,h) {
  #define historical window w and prediction horizon h (according to the preprocessing)
  
  
  #numProb defines the problem to predict, could be read as an argument to this scrit
  #numProb = 1
  
  
  #read data
  #matrix  <- read_delim("Ricerca/Energy/Data UPO/New Data/data as matrix/dataEd1.csv", ";", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)
  matrix  <- read_delim(file=nameData, ";", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)
  
  
  #Split the matrix to data_training and data_test 70% training in this case, change by varying split
  split = 0.7
  corte = floor(split*nrow(matrix))
  data_training = matrix[1:corte,]
  data_test = matrix[(corte+1):nrow(matrix),]
  
  data_training <- as.data.frame(data_training)
  data_test <- as.data.frame(data_test)
  
  #Make sure we only use the historical window
  data_p_training <- data_training[,1:w]
  data_p_test <- data_test [,1:w]
  
  ##################################define the column to predict##################################
  numPredic <- w+numProb
  colPredic <- paste("X",numPredic, sep ="")
  
  data_p_training$prediction <- data_training[[colPredic]]
  data_p_test$prediction <- data_test[[colPredic]]
  ########################XGBOOST PART ##############################################################3
  #Extreme Gradient Boosting
  #print("Random forest part")
  library(xgboost)
  write("XGBoost", stderr())
  xgbGrid <- expand.grid(nrounds = c(100,200),  # 
                         max_depth = c(10, 15, 20, 25),
                         colsample_bytree = seq(0.5, 0.9, length.out = 5),
                         ## The values below are default values in the sklearn-api. 
                         eta = 0.1,
                         gamma=0,
                         min_child_weight = 1
 #                       , subsample = 1
  )
  #xgbGrid <- expand.grid(nrounds = c(1, 10),
  #                     max_depth = c(1, 4),
  #                     eta = c(.1, .4),
  #                     gamma = 0,
  #                     colsample_bytree = .7,
  #                     min_child_weight = 1)#,
##subsample = c(.8, 1))
  modelXGB = train (prediction ~ ., method = "xgbTree", data = data_p_training, tuneGrid = xgbGrid,num.trees = 100)
  
  
  predictTestXGB <- predict(modelXGB,data_p_test)
  
  #prediction on training
  predictTrainXGB = predict(modelXGB, data_p_training)
  
  nameModel <- paste("./modelXGB_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
  save(modelXGB, predictTestXGB, file=nameModel)
  
  #compute evaluation measures 
  resXGB.mre <- (sum(abs(data_p_test$prediction - predictTestXGB) / data_p_test$prediction)) / nrow(data_p_test)*100
  resXGB.MAPE<-MAPE(data_p_test$prediction, predictTestXGB)
  resXGB.sMAPEW<-sMAPE(data_p_test$prediction, predictTestXGB)
  resXGB.mae = (sum(abs(data_p_test$prediction-predictTestXGB)))/nrow(data_p_test)
  resCaret <- caret::postResample(data_p_test$prediction, predictTestXGB)
  resXGB.R2 <- resCaret[2]
  resXGB.rmse = sqrt(MSE(data_p_test$prediction, predictTestXGB))
  
  
  #print the time of start and end of the creation of evtree model and the mre
  resultXGB <- c(resXGB.mre,"MRE: ",resXGB.mre,"MAPE: ",resXGB.MAPE,"sMAPEW: ", resXGB.sMAPEW,"MAE: ",resXGB.mae,"R2: ",resXGB.R2,"rmse: ",resXGB.rmse)
  nameResult <- paste("./resultXGB_HW",w,"PW",h,"_p",numProb,".txt",sep = "")
  write.table(resultXGB, sep =" ", eol="\n", nameResult)
  
  ########################END XGBOOST  PART ##############################################################3
  cat(sprintf("\t\tXGBoost \n"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \n"))
  cat(sprintf("%f \t %f \t %f \t %f \t %f \n",    resXGB.mre, resXGB.R2,resXGB.mae,resXGB.sMAPEW,resXGB.rmse )) 
  
  
  
  
}


#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
if (length(args)<3) {
  stop("must provide building-number historic-window-size prediction-horizon-size")
}
#use of this script
#parameters building-number w h
Ed = as.integer(args[1])
#print(Ed)
w = as.integer(args[2])
h = as.integer(args[3])

#w=15 #historical window
#h=1 #prediction horizon
#Ed=1 #this indicates the building, needed for reading the data
path = "/home/federico/Ricerca/Energy/Data UPO/New Data/data as matrix/dataEd"
nameData <- paste(path,Ed,"w",w,"h",h,".csv",sep = "")
cat(sprintf("Results for building %d, w %d, h %d \n  ",  Ed,w,h))  
for(i in 1:h){
  XGBoost(i,nameData,w,h)
}
