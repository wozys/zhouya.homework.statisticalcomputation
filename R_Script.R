# Generate Data
n = 100
p = 500
sigma_noise = 0.5
beta = rep(0, p)
beta[1:6] = c(5,10,3,80,90,10)
XData = matrix(rnorm(n*p,sd=10), nrow = n, ncol=p)
X=XData
for (i in 1:500){
  X[,i]=((X[,i]-sum(X[,i])/n))/((var(X[,i])*99/100)^0.5)
}
YData = X %*% beta + rnorm(n, sd = sigma_noise)
YData=YData-sum(YData)/100
XData=X
library(zhouyahomework1)
# Coordinate Descent for LASSO
betaHat1list = CoordinateDescent(XData,YData,1)
betaHat1=unlist(betaHat1list["betahat"])
diff1 = betaHat1 - beta
# L2 loss
L2Loss1 = sqrt(sum(diff1^2))
# True Positive Rate
TP1 = sum(abs(beta)>0 & abs(betaHat1)>0) / sum(abs(beta)>0)
# True Negative Rate
TN1 = sum(abs(beta)==0 & abs(betaHat1)==0) / sum(abs(beta)==0)
# Numbers of Iteration
ite1=unlist(betaHat1list["iteration"])
# Values of Objective Function
f1=unlist(betaHat1list["object function"])

# Proximal Operator for LASSO
betaHat2list=ProximalOperator(XData, YData, 1)
betaHat2 = unlist(betaHat2list["betahat"])
diff2 = betaHat2 - beta
# L2 loss
L2Loss2 = sqrt(sum(diff2^2))
# True Positive Rate
TP2 = sum(abs(beta)>0 & abs(betaHat2)>0) / sum(abs(beta)>0)
# True Negative Rate
TN2 = sum(abs(beta)==0 & abs(betaHat2)==0) / sum(abs(beta)==0)
# Numbers of Iteration
ite2=unlist(betaHat2list["iteration"])
# Values of Objective Function
f2=unlist(betaHat2list["object function"])

cat("the L2 loss for Coordinate Descent and Proximal Operator are  :",L2Loss1,L2Loss2)
cat("the True Positive Rate for Coordinate Descent and Proximal Operator are:",TP1,TP2)
cat("the True Negative Rate for Coordinate Descent and Proximal Operator are:",TN1,TN2)
cat("the numbers of iteration for Coordinate Descent and Proximal 
    Operator are:",ite1,ite2)

system.time(CoordinateDescent(XData,YData,1))
system.time(ProximalOperator(XData, YData, 1))


f11=log10(f1[1:ite1])
f22=log10(f2[1:ite2])

#Notation: the case is lambda=1
#plot log10 values of object functions and numbers of iteration
plot(1:ite2,f22,xlab="iteration",
     ylab="log object function",
     main="Coordinate Descent VS Proximal Operator ",
     type="s",col=2)
lines(1:ite1,f11,col=1)
legend("topright",inset=0.05,
       c("Proximal Operator","Coordinate Descent"),
       lty=c(1,1),col=c("red","black"))






