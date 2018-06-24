
library(rpart)
library(rattle)
library(kknn)
library(e1071)
library(pROC)

germancredit$Default<- as.factor(germancredit$Default)
set.seed(5139)
random<- sample(nrow(germancredit), 0.7*(nrow(germancredit)))
german_train<- germancredit[random,]
german_test<- germancredit[-random,]


#modified confirmation matrix, returns other measures
confmatrix<- function(real, pred){
  matrix<- table(real,pred)
  accuracy<- sum(diag(matrix))/sum(matrix)
  
  FP=matrix[2,1]
  TP=matrix[1,1]
  FN=matrix[1,2]
  TN=matrix[2,2]
  
  acc=(TP+TN)/(TP+TN+FP+FN)
  
  TNR=TN/(TN+FP)
  TPR=TP/(TP+FN)
  FNR=FN/(TP+FN)
  FPR=FP/(FP+TN)
  p=TP/(TP+FP)
  r=TP/(TP+FN)
  F1=2*r*p/(r+p) 
  
  return(list(matrix=matrix, accuracy = accuracy, error = 1-accuracy, TP= TP,
              TN=TN, FP=FP, FN=FN, acc=acc,TPR=TPR, TNR=TNR, FPR=FPR, FNR=FNR,
              p=p, r=r, F1=F1    ))
}
#1) ROC Curves, tree model
german_tree<- rpart(Default~., data = german_train)
fancyRpartPlot(german_tree)
predict_tree<-predict(german_tree, newdata= german_test, "class")
confmatrix(german_test$Default, predict_tree)$accuracy

#phat
tree_phat<- predict(german_tree, newdata= german_test)[,2]
plot(roc(german_test$Default=="1",tree_phat), xlab="Default", 
     ylab="prob. default", main="ROC Tree")


#weighted nearest neighbor
fit.glass<- train.kknn(Default~., data = german_train, kmax= 15,
                       kernel =c("rectangular","triangular", "epanechnikov",  "biweight",
                                 "triweight","cos","inv", "gaussian" ,"optimal"), 
                       distance = 2)
fit.glass$best.parameters

german_kknn<- kknn(Default~., train = german_train, test = german_test, k=15,
                   kernel= "optimal", distance = 2)

confmatrix(german_test$Default,german_kknn$fitted.values)$accuracy
kknn_phat<- german_kknn$prob[,2]

# ROC curve for weighted KKNN
plot(roc(german_test$Default=="1", kknn_phat), xlab="Default", 
     ylab="prob. default", main="ROC kknn")


#Naive bayes model
german_naive<-naiveBayes(Default~., data = german_train )
predict_naive<- predict(german_naive, newdata = german_test)
confmatrix(german_test$Default, predict_naive)$accuracy
naive_phat<- predict(german_naive, newdata = german_test, type ="raw")[,2]
#ROC curve for Naive Bayes
plot(roc(german_test$Default =="1", naive_phat),xlab="Default", 
     ylab="prob. default", main="ROC Naive Bayes")

#2)
###cost tree    predicting that they will default(false positive) when they do not
p0= 5000/(5000+20000)    #.2   False positive/(false positive+True negative)
p0    #P0 threshold is .2    FPR  

#modify prediction data set
treephatp0<- tree_phat
treephatp0[tree_phat>=0.2]="0"
treephatp0[tree_phat<0.2]="1"

cmatrixtreep0<- confmatrix(german_test$Default, treephatp0)
costtreep0<- 5000*cmatrixtreep0$FP+ 20000*cmatrixtreep0$FN
costtreep0

#cost KNN  (Weighted)
kknnphatp0<- kknn_phat
kknnphatp0[kknn_phat>=0.2]="0"
kknnphatp0[kknn_phat<0.2]="1"

cmatrixkknnp0<- confmatrix(german_test$Default, kknnphatp0)
costkknnp0<- 5000*cmatrixkknnp0$FP+ 20000*cmatrixkknnp0$FN
costkknnp0

#Naive Bayes

naivephatp0<- naive_phat
naivephatp0[naive_phat>=0.2]="0"
naivephatp0[naive_phat<0.2]="1"

cmatrixnaivep0<- confmatrix(german_test$Default, naivephatp0)
costnaivep0<- 5000*cmatrixnaivep0$FP+ 20000*cmatrixnaivep0$FN
costnaivep0

#C) Threshold of .5 

treephat.5<- tree_phat
treephat.5[tree_phat>=0.5]="0"
treephat.5[tree_phat<0.5]="1"

#cost tree

cmatrixtree.5<- confmatrix(german_test$Default, treephat.5)
costtree.5<- 5000*cmatrixtree.5$FP+ 20000*cmatrixtree.5$FN
costtree.5

#cost kknn
kknnphat.5<- kknn_phat
kknnphat.5[kknn_phat>=0.5]="0"
kknnphat.5[kknn_phat<0.5]="1"


cmatrixkknn.5<- confmatrix(german_test$Default, kknnphat.5)
costkknn.5<- 5000*cmatrixkknn.5$FP+ 20000*cmatrixkknn.5$FN
costkknn.5

#Cost for Naive Bayes 
naivephat.5<- naive_phat
naivephat.5[naive_phat>=0.5]="0"
naivephat.5[naive_phat<0.5]="1"


cmatrixnaive.5<- confmatrix(german_test$Default, naivephat.5)
costnaive.5<- 5000*cmatrixnaive.5$FP+ 20000*cmatrixnaive.5$FN
costnaive.5
