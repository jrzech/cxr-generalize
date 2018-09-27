#install.packages("pROC")
library("pROC")
#install.packages("PerfMeas")
#library("PerfMeas")

#load
#        img_id train_group pred_group     label       pred actual
main=read.csv("manu-main-experiment/rollup_probs_nopivot.csv")
#   img_id                 train_group pred_group     label       pred actual
engineered=read.csv("manu-imbalance-experiment/rollup_probs_nopivot.csv")
print(nrow(main))
print(nrow(engineered))

#filter
main = subset(main, label=="Pneumonia")
engineered = subset(engineered, label=="Pneumonia")
print(nrow(main))

#define fn
auc_ci=function(data,train,pred)
{
  rocs = list()  
  results = data.frame(train=character(),test=character(),auc_lower=double(),auc=double(),auc_upper=double(),count=integer(),tp=integer(),fp=integer(),tn=integer(),fn=integer(),sens=double(),spec=double(),acc=double(), ppv=double(), npv=double())
  results$train=as.character(results$train)
  results$test=as.character(results$test)
  
  for(t_iter in train)
  {
    for(p_iter in pred)
    {
      x = subset(data,train_group==t_iter)
      if(p_iter=="all")
      {
        print("...")
      }
      else if(p_iter=="msh_nih")
      {
        x = subset(x,((pred_group=="msh") | (pred_group=="nih"))) 
      }
      else
      {
        x = subset(x,pred_group==p_iter)
      }
      
      #one iteration here
      print(paste("\n\ntrain ",t_iter," pred ",p_iter," with rows ",nrow(x)))
      r=roc(x$actual, x$pred)
      labelstr=paste("t_",t_iter,"_p_",p_iter,sep="")
      rocs[[labelstr]]=r
      print(r)
      c=ci.auc(r, conf.level=0.95, method="delong") #this could  be bootstrap; boot.n = 2000, boot.stratified = TRUE
      print(c)
      #need thresh
      target_positive = ceiling(min(0.95*(sum(x$actual=="True")),length(x$actual)))
      x = x[order(-x$pred),]  #pred descending
      #find cutoff
      tempsum=0
      row=1
      step = 1#round(length(x$actual)/2)
      print(paste("step ",step))
      while(tempsum!=target_positive)
      {
        tempsum=tempsum + (x$actual[row]=="True")
        row = row+step
        #step=round(step/2)
        #print(paste("row ",row," tempsum ",tempsum," target_positive ",target_positive))
      }
      cutoff=x$pred[row]
      print(paste("cutoff=",cutoff))
      tp=sum(x$pred>=cutoff & x$actual=="True")
      tn=sum(x$pred<cutoff & x$actual=="False")
      fp=sum(x$pred>=cutoff & x$actual=="False")
      fn=sum(x$pred<cutoff & x$actual=="True")
      
      auprc=NA
      acc = (tp+tn)/(tp+tn+fp+fn)
      ppv = tp/(tp+fp)
      npv = tn/(tn+fn)
      
      results[nrow(results) + 1,] = list(t_iter,p_iter,c[1],c[2],c[3],nrow(x),tp,fp,tn,fn,(tp/(tp+fn)),(tn/(tn+fp)),acc,ppv,npv
      )
    }
  }         
  return_me=list()
  return_me$results=results
  return_me$rocs=rocs
  return(return_me)
}


#MAIN

#for main - iter over pred: nih, msh, iu, both msh and iu, all
#train: msh, nih, msh_nih
train<-list('nih','msh','msh_nih')
pred<-list('nih','msh','iu','msh_nih','all')
results = auc_ci(main, train,pred)
print(results$results)
write.csv(results$results,"manu-results/main_result_table.csv")

sink("manu-results/main_experiment_stat_analysis.txt", append=FALSE, split=FALSE)
print("MSH-NIH JOINT TRAIN PERFORMANCE VERSUS SINGLE SITE TRAIN")
#joint performance vs single site
p_iter_1="msh_nih"
p_iter_2="nih"
t_iter_1="msh_nih"
t_iter_2="msh_nih"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

p_iter_1="msh_nih"
p_iter_2="msh"
t_iter_1="msh_nih"
t_iter_2="msh_nih"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)

print("SINGLE SITE MSH OR NIH TRAIN ON ALL EXTERNAL COMPARISONS")
#single site internal > external
t_iter_1="msh"
t_iter_2="msh"
p_iter_1="msh"
p_iter_2="nih"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

t_iter_1="msh"
t_iter_2="msh"
p_iter_1="msh"
p_iter_2="iu"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)


t_iter_1="nih"
t_iter_2="nih"
p_iter_1="nih"
p_iter_2="msh"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

t_iter_1="nih"
t_iter_2="nih"
p_iter_1="nih"
p_iter_2="iu"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)


#train mshnih pred  iu vs train nih pred iu
print("TRAIN MSH-NIH JOINT, COMPARE PRED NIH VS IU, MSH VS IU, MSHNIH VS IU")
t_iter_1="msh_nih"
t_iter_2="nih"
p_iter_1="iu"
p_iter_2="iu"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

#train mshnih pred  iu vs train msh pred iu
t_iter_1="msh_nih"
t_iter_2="msh"
p_iter_1="iu"
p_iter_2="iu"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)

#COMPARISON - WORSE INTERNAL THAN EXTERNAL COMPARISON

#train mshnih pred  iu vs train msh pred iu
t_iter_1="msh_nih"
t_iter_2="msh_nih"
p_iter_1="msh_nih"
p_iter_2="iu"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

#is train MSH test NIH vs train NIH test NIH Different

#train mshnih pred  iu vs train msh pred iu
t_iter_1="msh"
t_iter_2="nih"
p_iter_1="nih"
p_iter_2="nih"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)


#five fully external comparisons for opening
print("five fully external comparisons")
print("train msh")
t_iter_1="msh"
t_iter_2="msh"
p_iter_1="msh"
p_iter_2="nih"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

t_iter_1="msh"
t_iter_2="msh"
p_iter_1="msh"
p_iter_2="iu"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

print("nih")

t_iter_1="nih"
t_iter_2="nih"
p_iter_1="nih"
p_iter_2="msh"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

t_iter_1="nih"
t_iter_2="nih"
p_iter_1="nih"
p_iter_2="iu"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)

print("nih-msh")

t_iter_1="msh_nih"
t_iter_2="msh_nih"
p_iter_1="msh_nih"
p_iter_2="iu"
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
print(z)








sink()


print("-------------------ENGINEERED-----------------------")
#ENGINEERED
unique(engineered$train_group)
unique(engineered$pred_group)

#for main - iter over pred: nih, msh, iu, both msh and iu, all
#train: msh, nih, msh_nih
train<-list('msh_nih_bal_balanced','msh_nih_bal_msh_mild','msh_nih_bal_msh_severe','msh_nih_bal_nih_mild','msh_nih_bal_nih_severe')
pred<-list('iu','msh_nih')
results = auc_ci(engineered, train,pred)
print(results$results)
write.csv(results$results,"manu-results/engineered_table.csv")


sink("manu-results/engineered_experiment_stat_analysis.txt", append=FALSE, split=FALSE)


t_iter_1="msh_nih_bal_balanced"
t_iter_2="msh_nih_bal_nih_severe"
p_iter_1="iu"
p_iter_2="iu"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)

t_iter_1="msh_nih_bal_balanced"
t_iter_2="msh_nih_bal_msh_severe"
p_iter_1="iu"
p_iter_2="iu"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)

t_iter_1="msh_nih_bal_balanced"
t_iter_2="msh_nih_bal_nih_severe"
p_iter_1="msh_nih"
p_iter_2="msh_nih"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)

t_iter_1="msh_nih_bal_balanced"
t_iter_2="msh_nih_bal_msh_severe"
p_iter_1="msh_nih"
p_iter_2="msh_nih"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)

print("extremes internal vs external")

t_iter_1="msh_nih_bal_msh_severe"
t_iter_2="msh_nih_bal_msh_severe"
p_iter_1="msh_nih"
p_iter_2="iu"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)


t_iter_1="msh_nih_bal_nih_severe"
t_iter_2="msh_nih_bal_nih_severe"
p_iter_1="msh_nih"
p_iter_2="iu"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)


t_iter_1="msh_nih_bal_msh_mild"
t_iter_2="msh_nih_bal_msh_mild"
p_iter_1="msh_nih"
p_iter_2="iu"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)


t_iter_1="msh_nih_bal_nih_mild"
t_iter_2="msh_nih_bal_nih_mild"
p_iter_1="msh_nih"
p_iter_2="iu"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)

t_iter_1="msh_nih_bal_balanced"
t_iter_2="msh_nih_bal_balanced"
p_iter_1="msh_nih"
p_iter_2="iu"
print(paste("question: is train ",t_iter_1," pred ",p_iter_1," > train ",t_iter_2," pred ",p_iter_2,"?"))
z=roc.test(results$rocs[[paste("t_",t_iter_1,"_p_",p_iter_1,sep="")]],results$rocs[[paste("t_",t_iter_2,"_p_",p_iter_2,sep="")]])
print(z)


sink()#
