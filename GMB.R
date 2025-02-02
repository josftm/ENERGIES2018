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

GBmFunction<-function(numProb,nameData,w,h) {
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
  ########################gmb PART ##############################################################3
  #Extreme Gradient Boosting
  #print("Random forest part")
  library(GBMoost)
  write("GBM", stderr())
  
  gbmGrid <- expand.grid(interaction.depth = 1:2,
                       shrinkage = .1,
                       n.trees = c(10, 50, 100),n.minobsinnode = 10)
  
  #modelGBM = train (prediction ~ ., method = "GBMTree", data = data_p_training, num.trees = 100)
  
  modelGBM = train(prediction ~ ., 
                             method = "gbm", 
                           data = data_p_training,
                             tuneGrid = gbmGrid,verbose = FALSE)
  
  
  
  predictTestGBM <- predict(modelGBM,data_p_test)
  
  #prediction on training
  predictTrainGBM = predict(modelGBM, data_p_training)
  
  nameModel <- paste("./modelGBM_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
  save(modelGBM, predictTestGBM, file=nameModel)
  
  #compute evaluation measures 
  resGBM.mre <- (sum(abs(data_p_test$prediction - predictTestGBM) / data_p_test$prediction)) / nrow(data_p_test)*100
  resGBM.MAPE<-MAPE(data_p_test$prediction, predictTestGBM)
  resGBM.sMAPEW<-sMAPE(data_p_test$prediction, predictTestGBM)
  resGBM.mae = (sum(abs(data_p_test$prediction-predictTestGBM)))/nrow(data_p_test)
  resCaret <- caret::postResample(data_p_test$prediction, predictTestGBM)
  resGBM.R2 <- resCaret[2]
  resGBM.rmse = sqrt(MSE(data_p_test$prediction, predictTestGBM))
  
  
  #print the time of start and end of the creation of evtree model and the mre
  resultGBM <- c(resGBM.mre,"MRE: ",resGBM.mre,"MAPE: ",resGBM.MAPE,"sMAPEW: ", resGBM.sMAPEW,"MAE: ",resGBM.mae,"R2: ",resGBM.R2,"rmse: ",resGBM.rmse)
  nameResult <- paste("./resultGBM_HW",w,"PW",h,"_p",numProb,".txt",sep = "")
  write.table(resultGBM, sep =" ", eol="\n", nameResult)
  
  ########################END GBMOOST  PART ##############################################################3
  cat(sprintf("\t\tGBMoost \n"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \n"))
  cat(sprintf("%f \t %f \t %f \t %f \t %f \n",    resGBM.mre, resGBM.R2,resGBM.mae,resGBM.sMAPEW,resGBM.rmse )) 
  
  
  
  
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
  GBmFunction(i,nameData,w,h)
}
