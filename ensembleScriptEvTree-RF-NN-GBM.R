library(readr)
library(evtree)
library(rpart)
library(ggplot2)
library("mlbench")
library("randomForest")
library("nnet")
library('caret')
library("gbm")

#Retrieving arguments that is the problem number and the historical window
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  msg <- "At least two argument must be supplied : the first is the number of the problem the second the historical window.\n"
  stop(msg, call.=FALSE)
}


numProb = as.integer(args[1])
print(numProb)

w = as.integer(args[2])
h = 24
print(w)


#get the matrix corresponding to the historical window
nameData <- paste("./dataHW",w,"PW",h,".csv",sep = "")
matrix <- read.csv(file=nameData,head=TRUE,sep=",",stringsAsFactors=F)
matrix$X<- NULL
#Split the matrix to data_training and data_test
split = 0.7
corte = floor(split*nrow(matrix))
data_evtree_training = matrix[1:corte,]
data_evtree_test = matrix[(corte+1):nrow(matrix),]

data_evtree_training <- as.data.frame(data_evtree_training)
data_evtree_test <- as.data.frame(data_evtree_test)

#Make sure we only use the historical window
data_p_training <- data_evtree_training[,1:w]
data_p_test <- data_evtree_test [,1:w]

#saving the predictors to be used (needed for NN)
predictors = names(data_p_training)
##################################define the column to predict##################################

numPredic <- w+numProb
colPredic <- paste("X",numPredic, sep ="")
data_p_training$X169 <- data_evtree_training[[colPredic]]
data_p_test$X169 <- data_evtree_test[[colPredic]]

#####################load EVtree model #########################################################
#format is like this: modelEvtree_HW24PW24_p5
EVTreeModelName = paste("modelEvtree_HW",w,"PW24_p",numProb,".rda",sep = "")
print(EVTreeModelName)
objectsEVTree=load(EVTreeModelName)
modelEvTree = Evtree_model_p

res <- c()
res <- c(format(Sys.time(), "%d-%b-%Y %H.%M"))
#predict on the training and test sets
nbRowTest <- nrow(data_p_test)
nbRowTrain = nrow(data_p_training)
#use the model to predict X169 on the test data
#Evtree_prediction_p <- predict(Evtree_model_p,data_p_test)
nbRow <- nrow(data_p_test)
predictTrain_evtree = predict(modelEvTree, data_p_training)
predictTest_evtree <- predict(modelEvTree,data_p_test)
MRE_EvTree <- mreEVTREE_p



######################## RANDOM FOREST PART ##############################################################3
#save the RF model, the prediction and the mre in a R object
modelRF = randomForest(X169 ~ ., data = data_p_training, ntree=100,maxnodes = 8, prox=TRUE)
predictTest_RF <- predict(modelRF,data_p_test)
mreRF_p <- (sum(abs(data_p_test$X169 - predictTest_RF) / data_p_test$X169)) / nrow(data_p_test)*100

#prediction on training
predictTrainRF = predict(modelRF, data_p_training)

nameModel <- paste("./modelRF_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(modelRF, predictTest_RF, mreRF_p, file=nameModel)



################### NN part ################################################################
f <- as.formula(paste("X169 ~", paste(predictors, collapse = " + ")))

modelNN <- nnet(f, data_p_training,size=10,linout=TRUE, skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=1000)
predictTestNN = predict(modelNN,data_p_test)
mreNN_p <- (sum(abs(data_p_test$X169 - predictTestNN) / data_p_test$X169)) / nrow(data_p_test)*100

#prediction on training
predictTrainNN = predict(modelNN, data_p_training)

nameModel <- paste("./modelNN_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(modelNN, predictTestNN, mreNN_p, file=nameModel)
################################# ADD PREDICTION TO TEST AND TRAINING ####################3
#add prediction to test data
data_p_test$Pred_EV = predictTest_evtree
data_p_test$Pred_RF = predictTest_RF
data_p_test$Pred_NN = predictTestNN


#add prediction to training and test sets for later steps
data_p_training$Pred_EV = predictTrain_evtree
data_p_training$Pred_RF = predictTrainRF
data_p_training$Pred_NN = predictTrainNN


############################ TOP LAYER OF ENSEMBLE ########################################
#use lm to build emsamble, check it out, it's too simple now
#model_lm = lm(X169 ~ Pred_EV + Pred_RF + Pred_NN, data=data_p_training, na.action = NULL)
model_gbm = gbm(X169 ~ Pred_EV + Pred_RF + Pred_NN, data = data_p_training,distribution = "gaussian", n.trees = 5, interaction.depth = 8, shrinkage = 1, n.minobsinnode = 1)
#predict
predictTest_gbm <- predict(model_gbm,data_p_test, n.trees = 5)
mreGBM_p <- (sum(abs(data_p_test$X169 - predictTest_gbm) / data_p_test$X169)) / nrow(data_p_test)*100

nameModel <- paste("./modelGBM_HW",w,"PW",h,"_p",numProb,".rda", sep ="")
save(model_gbm, predictTest_gbm, mreGBM_p, file=nameModel)

#saving time
res <- c(res,(format(Sys.time(), "%d-%b-%Y %H.%M")))
nameResult <- paste("./result_HW",w,"PW",h,"_p",numProb,".txt",sep = "")
write.table(res, sep =" ", eol="\n", nameResult)
print(mreGBM_p)
print(mreNN_p)
print(mreRF_p)