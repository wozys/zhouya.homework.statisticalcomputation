
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
# Coordinate Descent for LASSO
betaHat1list = CoordinateDescent(XData,YData,1)
betaHat1=unlist(betaHat1list["betahat"])
diff1 = betaHat1 - beta
# L2 loss
L2Loss1 = sqrt(sum(diff1^2))
# Proximal Operator for LASSO
betaHat2list=ProximalOperator(XData, YData, 1)
betaHat2 = unlist(betaHat2list["betahat"])
diff2 = betaHat2 - beta
# L2 loss
L2Loss2 = sqrt(sum(diff2^2))
cat("L2 Loss are respectively:",L2Loss1,L2Loss2)