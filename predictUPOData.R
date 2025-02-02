library(evtree)
library(rpart.plot)
require(reshape)
library(caret.ts)
library(forecast)
library("TSPred")
library("gbm")
library("mlbench")
library("randomForest")
library("nnet")
library(readr)
#library(ranger)
#library(quantregForest)

ScriptForRevision<-function(numProb,nameData,w,h) {
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


#rename columns
#colnames(data_p_training) <- paste("X", 1:ncol(data_p_training), sep="")
#colnames(data_p_test) <- paste("X", 1:ncol(data_p_test), sep="")


########################EVTree #####################################################################
#Define the parameter for evtree
#thre's a method with caret method = 'evtree'
#test_class_cv_model <- train(trainX, trainY, method = "evtree",trControl = cctrl1,control = evc,preProc = c("center", "scale"))

write("EVTree", stderr())
minbucket = 8L
minsplit = 100L
maxdepth = 15L
ntrees = 300L
generations = 800L
controlEv <- evtree.control(minbucket = minbucket, minsplit = minsplit, maxdepth = maxdepth,
                            niterations = generations, ntrees = ntrees, alpha =0.25,
                            operatorprob = list(pmutatemajor = 0.2, pmutateminor = 0.2,
                                                pcrossover = 0.8, psplit= 0.2, pprune = 0.4),
                            seed = NULL)
res <- c()
res <- c(format(Sys.time(), "%d-%b-%Y %H.%M"))

#print("running EVtree")
#create the model on the training data (predict prediction in function of the historical window)
#eval(colPredic), get(colPredic).... to change prediction
Evtree_model_p <- evtree(prediction ~.,data = data_p_training, control = controlEv)


res <- c(res,(format(Sys.time(), "%d-%b-%Y %H.%M")))


#use the model to predict prediction on the test data
Evtree_prediction_p <- predict(Evtree_model_p,data_p_test)
predictTestEVTree = Evtree_prediction_p
nbRow <- nrow(data_p_test)
#prediction on training
predictTrainEVTree = predict(Evtree_model_p, data_p_training)
#compute evaluation measures 
resEVTree.mre <- (sum(abs(data_p_test$prediction - Evtree_prediction_p) / data_p_test$prediction)) / nrow(data_p_test)*100
resEVTree.MAPE<-MAPE(data_p_test$prediction, Evtree_prediction_p)
resEVTree.sMAPEW<-sMAPE(data_p_test$prediction, Evtree_prediction_p)
resEVTree.mae = (sum(abs(data_p_test$prediction-Evtree_prediction_p)))/nrow(data_p_test)

#resEVTree.R2 <- 1 - (sum((data_p_test$prediction-Evtree_prediction_p)^2)/sum((data_p_test$prediction-mean(data_p_test$prediction))^2))
# R-squared is calculated as the square of the correlation between the observed and predicted outcomes.
#R^2 = 1-\frac{∑ (y_i - \hat{y}_i)^2}{∑ (y_i - \bar{y}_i)^2} 
#see https://rdrr.io/cran/caret/man/postResample.html
resCaret <- caret::postResample(data_p_test$prediction, Evtree_prediction_p)
resEVTree.R2 <- resCaret[2]
#resCaret will contain:
#RMSE
#Rsquared
#mean absolute error (MAE)
resEVTree.rmse = sqrt(MSE(data_p_test$prediction, Evtree_prediction_p))


nameModel <- paste("./modelEvtree_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(Evtree_model_p, Evtree_prediction_p, file=nameModel)

#print the time of start and end of the creation of evtree model and the mre
resultEVTree <- c(res,"MRE: ",resEVTree.mre,"MAPE: ",resEVTree.MAPE,"sMAPEW: ", resEVTree.sMAPEW,"MAE: ",resEVTree.mae,"R2: ",resEVTree.R2,"rmse: ",resEVTree.rmse)
nameResult <- paste("./resultEVTree_HW",w,"PW",h,"_p",numProb,".txt",sep = "")
write.table(resultEVTree, sep =" ", eol="\n", nameResult)

########################End EVTree ##################################################################

######################## RANDOM FOREST PART ##############################################################3
#save the RF model, the prediction and the mre in a R object
#print("Random forest part")
write("Random Forest", stderr())
#caret methods rf, ranger, qrf (quantregForest)
#modelRF = train (prediction ~ ., method = "ranger", data = data_p_training, num.trees = 100)


modelRF = randomForest(prediction ~ ., data = data_p_training, ntree=300,maxnodes = 30, prox=TRUE)
predictTest_RF <- predict(modelRF,data_p_test)

#prediction on training
predictTrainRF = predict(modelRF, data_p_training)

nameModel <- paste("./modelRF_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(modelRF, predictTest_RF, file=nameModel)

#compute evaluation measures 
resRF.mre <- (sum(abs(data_p_test$prediction - predictTest_RF) / data_p_test$prediction)) / nrow(data_p_test)*100
resRF.MAPE<-MAPE(data_p_test$prediction, predictTest_RF)
resRF.sMAPEW<-sMAPE(data_p_test$prediction, predictTest_RF)
resRF.mae = (sum(abs(data_p_test$prediction-predictTest_RF)))/nrow(data_p_test)
resCaret <- caret::postResample(data_p_test$prediction, predictTest_RF)
resRF.R2 <- resCaret[2]
resRF.rmse = sqrt(MSE(data_p_test$prediction, predictTest_RF))


#print the time of start and end of the creation of evtree model and the mre
resultRF <- c(res,"MRE: ",resRF.mre,"MAPE: ",resRF.MAPE,"sMAPEW: ", resRF.sMAPEW,"MAE: ",resRF.mae,"R2: ",resRF.R2,"rmse: ",resRF.rmse)
nameResult <- paste("./resultRF_HW",w,"PW",h,"_p",numProb,".txt",sep = "")
write.table(resultRF, sep =" ", eol="\n", nameResult)

########################END RANDOM FOREST PART ##############################################################3

########################XGBOOST PART ##############################################################3
#Extreme Gradient Boosting
#print("Random forest part")
library(xgboost)
write("XGBoost", stderr())
xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)
modelXGB = train (prediction ~ ., method = "xgbTree", data = data_p_training, num.trees = 100)


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

########################END XGBOOST  PART ##############################################################

################### NN part ################################################################
#print("NN part")
write("Neural Network", stderr())

predictors = names(data_p_training)

predictors = predictors[1:(length(predictors)-1)]
#print(predictors)

f <- as.formula(paste("prediction ~", paste(predictors, collapse = " + ")))
modelNN <- train(f, data_p_training,method='nnet',linout=TRUE,trace = FALSE)
#modelNN <- nnet(f, data_p_training,size=50,linout=TRUE, skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=1000)
#print("Parameters of NN")
#summary(modelNN)

predictTestNN = predict(modelNN,data_p_test)

#prediction on training
predictTrainNN = predict(modelNN, data_p_training)

nameModel <- paste("./modelNN_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(modelNN, predictTestNN, file=nameModel)

#compute evaluation measures 
resNN.mre <- (sum(abs(data_p_test$prediction - predictTestNN) / data_p_test$prediction)) / nrow(data_p_test)*100
resNN.MAPE<-MAPE(data_p_test$prediction, predictTestNN)
resNN.sMAPEW<-sMAPE(data_p_test$prediction, predictTestNN)
resNN.mae = (sum(abs(data_p_test$prediction-predictTestNN)))/nrow(data_p_test)
resCaret <- caret::postResample(data_p_test$prediction, predictTestNN)
resNN.R2 <- resCaret[2]
resNN.rmse = sqrt(MSE(data_p_test$prediction, predictTestNN))
#print the time of start and end of the creation of evtree model and the mre
resultNN <- c(res,"MRE: ",resNN.mre,"MAPE: ",resNN.MAPE,"sMAPEW: ", resNN.sMAPEW,"MAE: ",resNN.mae,"R2: ",resNN.R2,"rmse: ",resNN.rmse)
nameResult <- paste("./resultNN_HW",w,"PW",h,"_p",numProb,".txt",sep = "")
write.table(resultNN, sep =" ", eol="\n", nameResult)

###################END NN part ################################################################

###################GBMpart ################################################################
write("GBM", stderr())

model_gbm = gbm(prediction ~ ., data = data_p_training,distribution = "gaussian", n.trees = 300, interaction.depth = 10, shrinkage = 0.9, n.minobsinnode = 1)
predictTestGBM = predict(model_gbm,data_p_test,n.trees = 300)


nameModel <- paste("./modelGBM_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(model_gbm, predictTestGBM, file=nameModel)

#compute evaluation measures 
resGBM.mre <- (sum(abs(data_p_test$prediction - predictTestGBM) / data_p_test$prediction)) / nrow(data_p_test)*100
resGBM.MAPE<-MAPE(data_p_test$prediction, predictTestGBM)
resGBM.sMAPEW<-sMAPE(data_p_test$prediction, predictTestGBM)
resGBM.mae = (sum(abs(data_p_test$prediction-predictTestGBM)))/nrow(data_p_test)
resCaret <- caret::postResample(data_p_test$prediction, predictTestGBM)
resGBM.R2 <- resCaret[2]
resGBM.rmse = sqrt(MSE(data_p_test$prediction, predictTestGBM))
#print the time of start and end of the creation of evtree model and the mre
resultGBM <- c(res,"MRE: ",resGBM.mre,"MAPE: ",resGBM.MAPE,"sMAPEW: ", resGBM.sMAPEW,"MAE: ",resGBM.mae,"R2: ",resGBM.R2,"rmse: ",resGBM.rmse)
nameResult <- paste("./resultGBM_HW",w,"PW",h,"_p",numProb,".txt",sep = "")
write.table(resultGBM, sep =" ", eol="\n", nameResult)
###################GBMpart ################################################################
###################other methods part ################################################################
library(caret.ts)
library(forecast)
library("TSPred")
write("other methods", stderr())


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
arima <- train(prediction ~ ., data = data_p_training, method = arima_model(5, 0, 0), trControl = trainDirectFit())
#arma_model(p, q, intercept = TRUE, ...)
arma <- train(prediction ~ ., data = data_p_training, method = arma_model(5,0), trControl = trainDirectFit())
lmModel <- train(prediction ~ ., data = data_p_training, method = "lm")
rpartModel <- train(prediction ~ ., data = data_p_training, method =  "rpart")

predictArma = predict(arma,data_p_test)
predictArima = predict(arima,data_p_test)
predictLM = predict(lmModel,data_p_test)
predictRPart = predict(rpartModel,data_p_test)
predictTestRPart = predictRPart
predictTrainRPart = predict(rpartModel,data_p_training)

nameModel = nameModel <- paste("./prediction_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(predictArma,predictArima,predictLM,predictRPart, file=nameModel)

#compute measures
mreArma_p <- (sum(abs(data_p_test$prediction - predictArma) / data_p_test$prediction)) / nrow(data_p_test)*100
mreArima_p <- (sum(abs(data_p_test$prediction - predictArima) / data_p_test$prediction)) / nrow(data_p_test)*100
mreLM_p <- (sum(abs(data_p_test$prediction - predictLM) / data_p_test$prediction)) / nrow(data_p_test)*100
mreRPart_p <- (sum(abs(data_p_test$prediction - predictRPart) / data_p_test$prediction)) / nrow(data_p_test)*100
#mreGBM_p <- (sum(abs(data_p_test$prediction - predictGBM) / data_p_test$prediction)) / nrow(data_p_test)*100



MAPE_Arma<-MAPE(data_p_test$prediction, predictArma)
MAPE_Arima<-MAPE(data_p_test$prediction, predictArima)
MAPE_LM<-MAPE(data_p_test$prediction, predictLM)
MAPE_RPart<-MAPE(data_p_test$prediction, predictRPart)
#MAPE_GBM<-MAPE(data_p_test$prediction, predictGBM)


sMAPEW_Arima<-sMAPE(data_p_test$prediction, predictArima)
sMAPEW_Arma<-sMAPE(data_p_test$prediction, predictArma)
sMAPEW_LM<-sMAPE(data_p_test$prediction, predictLM)
sMAPEW_RPart<-sMAPE(data_p_test$prediction, predictRPart)
#sMAPEW_GBM<-sMAPE(data_p_test$prediction, predictGBM)


mae_Arma = (sum(abs(data_p_test$prediction-predictArma)))/nrow(data_p_test)
mae_Arima = (sum(abs(data_p_test$prediction-predictArima)))/nrow(data_p_test)
mae_LM = (sum(abs(data_p_test$prediction-predictLM)))/nrow(data_p_test)
mae_RPart = (sum(abs(data_p_test$prediction-predictRPart)))/nrow(data_p_test)
#mae_GBM = (sum(abs(data_p_test$prediction-predictGBM)))/nrow(data_p_test)

rmse_Arma = sqrt(MSE(data_p_test$prediction, predictArma))
rmse_Arima = sqrt(MSE(data_p_test$prediction, predictArima))
rmse_LM = sqrt(MSE(data_p_test$prediction, predictLM))
rmse_RPart = sqrt(MSE(data_p_test$prediction, predictRPart))

resCaret <- caret::postResample(data_p_test$prediction, predictArma)
R2_Arma <- resCaret[2]

resCaret <- caret::postResample(data_p_test$prediction, predictArima)
R2_Arima <- resCaret[2]

resCaret <- caret::postResample(data_p_test$prediction, predictLM)
R2_LM <- resCaret[2]

resCaret <- caret::postResample(data_p_test$prediction, predictRPart)
R2_RPart <- resCaret[2]

###################Ensemble ################################################################

################################# ADD PREDICTION TO TEST AND TRAINING ####################3
write("Ensemble", stderr())

#add prediction to test data
data_p_test$Pred_EV = predictTestEVTree
data_p_test$Pred_RF = predictTest_RF
#change for rpart
#data_p_test$Pred_NN = predictTestRPart
data_p_test$Pred_NN = predictTestNN


#add prediction to training and test sets for later steps
data_p_training$Pred_EV = predictTrainEVTree
data_p_training$Pred_RF = predictTrainRF
#change for rpart
#data_p_training$Pred_NN = predictTrainRPart
data_p_training$Pred_NN = predictTrainNN


#model_ensemble = lm(prediction ~ Pred_EV + Pred_RF + Pred_NN, data=data_p_training, na.action = NULL)
model_ensemble = gbm(prediction ~ Pred_EV + Pred_RF + Pred_NN, data = data_p_training,distribution = "gaussian", n.trees = 5, interaction.depth = 8, shrinkage = 1, n.minobsinnode = 1)
#predict
predictTest_ensemble <- predict(model_ensemble,data_p_test, n.trees = 5)



nameModel <- paste("./modelEnsemble_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(model_ensemble,file=nameModel)

#compute evaluation measures 
resEnsemble.mre <- (sum(abs(data_p_test$prediction - predictTest_ensemble) / data_p_test$prediction)) / nrow(data_p_test)*100
resEnsemble.MAPE<-MAPE(data_p_test$prediction, predictTest_ensemble)
resEnsemble.sMAPEW<-sMAPE(data_p_test$prediction, predictTest_ensemble)
resEnsemble.mae = (sum(abs(data_p_test$prediction-predictTest_ensemble)))/nrow(data_p_test)
resCaret <- caret::postResample(data_p_test$prediction, predictTest_ensemble)
resEnsemble.R2 <- resCaret[2]
resEnsemble.rmse = sqrt(MSE(data_p_test$prediction, predictTest_ensemble))
#print the time of start and end of the creation of evtree model and the mre
resultEnsemble <- c(res,"MRE: ",resEnsemble.mre,"MAPE: ",resEnsemble.MAPE,"sMAPEW: ", resEnsemble.sMAPEW,"MAE: ",resEnsemble.mae,"R2: ",resEnsemble.R2,"rmse: ",resEnsemble.rmse)
nameResult <- paste("./resultEnsemble_HW",w,"PW",h,"_p",numProb,".txt",sep = "")
write.table(resultEnsemble, sep =" ", eol="\n", nameResult)

###################Ensemble ################################################################
#if(i==1){
  cat(sprintf("\t\tEvTree \t\t\t\t\t RF \t\t\t\t\t NN \t\t\t\t\t GBM \t\t\t\t\t ARMA \t\t\t\t\t ARIMA \t\t\t\t\t LM \t\t\t\t\t rpart \t\t\t\t\t ENSEMBLE \t\t\t\t\t XGBoost \n" ))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \t"))
  cat(sprintf("MRE \t R2 \t MAE \t sMAPE \t RMSE \n"))
  
#}
cat(sprintf("%f \t %f \t %f \t %f \t %f \t  ",  resEVTree.mre, resEVTree.R2,resEVTree.mae,resEVTree.sMAPEW,resEVTree.rmse)) 
cat(sprintf("%f \t %f \t %f \t %f \t %f \t  ",  resRF.mre, resRF.R2,resRF.mae,resRF.sMAPEW,resRF.rmse))  
cat(sprintf("%f \t %f \t %f \t %f \t %f \t  ",  resNN.mre, resNN.R2,resNN.mae,resNN.sMAPEW,resNN.rmse)) 
cat(sprintf("%f \t %f \t %f \t %f \t %f \t  ",  resGBM.mre, resGBM.R2,resGBM.mae,resGBM.sMAPEW,resGBM.rmse)) 



cat(sprintf("%f \t %f \t %f \t %f \t %f \t  ",  mreArma_p, R2_Arma, mae_Arma,  sMAPEW_Arma, rmse_Arma))  
cat(sprintf("%f \t %f \t %f \t %f \t %f \t  ",  mreArima_p,R2_Arima,mae_Arima, sMAPEW_Arima,rmse_Arima )) 
cat(sprintf("%f \t %f \t %f \t %f \t %f \t ",   mreLM_p,   R2_LM,   mae_LM,    sMAPEW_LM,   rmse_LM )) 
cat(sprintf("%f \t %f \t %f \t %f \t %f \t ",   mreRPart_p,R2_RPart,mae_RPart, sMAPEW_RPart,rmse_RPart )) 
cat(sprintf("%f \t %f \t %f \t %f \t %f \t",    resEnsemble.mre, resEnsemble.R2,resEnsemble.mae,resEnsemble.sMAPEW,resEnsemble.rmse )) 
cat(sprintf("%f \t %f \t %f \t %f \t %f \n",    resXGB.mre, resXGB.R2,resXGB.mae,resXGB.sMAPEW,resXGB.rmse )) 


###################ARIMA ARMA part ################################################################

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
for(i in 1:h){
  ScriptForRevision(i,nameData,w,h)
}
