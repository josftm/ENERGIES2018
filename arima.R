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
ScriptForRevision<-function(numProb,nameData,w,h,Ed) {

  #read data
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
  
  
  
  
  #TODO ESTABLISH THE PARAMETERS
  #this is according to arimapar on the whole dataset
  #arimapar(AllYears_NoDates$Edificio1,na.action = na.omit, xreg = NULL)
  #result A numeric vector giving the number of AR, MA, seasonal AR and seasonal MA coefficients, plus the period and the number of non-seasonal and seasonal differences of the automatically fitted ARIMA model. It is also presented the value of the fitted drift constant.
  #      AR Diff MA SeasonalAR SeasonalDiff SeasonalMA Period Drift
  #[1,]  5    0  0          0            0          0      1    NA
  #arima_model(p, d, q, intercept = TRUE, ...)
  #p	Order of auto-regressive (AR) terms. P=AR=5
  #d	Degree of differencing. d=0
  #q	Order of moving-average (MA) terms. q=MA=0
  
  p=5
  d=0
  q=0
  
  if (Ed==1) {
    p=5
    d=0
    q=0
  }
  if (Ed==2) {
    p=3
    d=0
    q=1
  }
  if (Ed==3) {
    p=0
    d=0
    q=0
  }
  if (Ed==6) {
    p=3
    d=0
    q=2
  }
  if (Ed==7) {
    p=1
    d=0
    q=1
  }
  if (Ed==8) {
    p=3
    d=0
    q=4
  }
  if (Ed==10) {
    p=3
    d=0
    q=2
  }
  if (Ed==12) {
    p=2
    d=0
    q=3
  }
  if (Ed==14) {
    p=3
    d=0
    q=1
  }
  if (Ed==15) {
    p=0
    d=0
    q=1
  }
  if (Ed==17) {
    p=2
    d=0
    q=2
  }
  if (Ed==24) {
    p=0
    d=0
    q=1
  }
  if (Ed==25) {
    p=4
    d=0
    q=4
  }
  if (Ed==32) {
    p=0
    d=0
    q=2
  }
  if (Ed==37) {
    p=3
    d=0
    q=2
  }
  if (Ed==42) {
    p=1
    d=0
    q=1
  }
  if (Ed==44) {
    p=4
    d=0
    q=5
  }
  
  arima <- train(prediction ~ ., data = data_p_training, method = arima_model(p, d, q), trControl = trainDirectFit())
  predictTestArima <- predict(arima,data_p_test)
  
  #prediction on training
  predictTrainXGB = predict(arima, data_p_training)

    
  #saving predictions
  nameModel = nameModel <- paste("./predictionARIMA_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
  predictName = paste("./predictionARIMA_HW",w,"PW",h,"_p",numProb,".csv", sep ="")
  save(predictTestArima, file=nameModel)
  write.csv(predictTestArima,file=predictName)
  
  #compute measures
  #compute evaluation measures 
  resArima.mre <- (sum(abs(data_p_test$prediction - predictTestArima) / data_p_test$prediction)) / nrow(data_p_test)*100
  resArima.MAPE<-MAPE(data_p_test$prediction, predictTestArima)
  resArima.sMAPEW<-sMAPE(data_p_test$prediction, predictTestArima)
  resArima.mae = (sum(abs(data_p_test$prediction-predictTestArima)))/nrow(data_p_test)
  resCaret <- caret::postResample(data_p_test$prediction, predictTestArima)
  resArima.R2 <- resCaret[2]
  resArima.rmse = sqrt(MSE(data_p_test$prediction, predictTestArima))
  
  
  
  ########################END XGBOOST  PART ##############################################################3
  cat(sprintf("\t\tXGBoost \n"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \n"))
  cat(sprintf("%f \t %f \t %f \t %f \t %f \n",    resArima.mre, resArima.R2,resArima.mae,resArima.sMAPEW,resArima.rmse )) 

}

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
if (length(args)<3) {
  stop("must provide building-number historic-window-size prediction-horizon-size")
}

write("Arguments: ", stderr())
write(args, stderr())

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
#AllYears_NoDates <- read_csv("Ricerca/Energy/Data UPO/New Data/AllYears - NoDates.csv")
#building = paste("Building",Ed, sep ="")
#TS = paste("AllYears_NoDates$",building, sep ="")
#parameters = arimapar(AllYears_NoDates$Building1,na.action = na.omit, xreg = NULL)

for(i in 1:h){
  ScriptForRevision(i,nameData,w,h,Ed)
}

