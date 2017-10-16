#include <RcppArmadillo.h>
#include "soft_th.hpp"
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

//' The algorithm of Proximal Operator for LASSO
//' @param X the observation vector
//' @param y the response vector 
//' @param lambda the smoothing parameter
//' @return the list which concludes the estimator beta, 
//' the corresponding values of objective functions and 
//' the numbers of iteration 
//' @examples
//' require(zhouyahomework1)
//' # Generate Data
//' n = 100
//' p = 500
//' sigma_noise = 0.5
//' beta = rep(0, p)
//' beta[1:6] = c(5,10,3,80,90,10)
//' XData = matrix(rnorm(n*p,sd=10), nrow = n, ncol=p)
//' X=XData
//' for (i in 1:500){
//' X[,i]=((X[,i]-sum(X[,i])/n))/((var(X[,i])*99/100)^0.5)
//' }
//' YData = X %*% beta + rnorm(n, sd = sigma_noise)
//' YData=YData-sum(YData)/100
//' XData=X
//' # Proximal Operator for LASSO
//' betaHat2list=ProximalOperator(YData, XData, 1)
//' betaHat2 = unlist(betaHat2list["betahat"])
// [[Rcpp::export]]
Rcpp::List ProximalOperator(const arma::mat& X, const arma::colvec& y,double lambda)
{
  //Let the max iteration equal 1000 by default
  int Maxiteration=1000;
  int n = X.n_rows, p=X.n_cols;
  //use "the max eigen value +1" as M
  double M=eig_sym(X.t()*X).max()/n+1;
  //use the solution of least square as initial value of beta
  arma::colvec beta=arma::solve(X, y);
  arma::colvec XTyXXBeta=beta;
  arma::colvec betaold=beta;
  arma::colvec f(Maxiteration);
  f=f.fill(0);
  int ite=0;
  
  for(int i=0;i<=Maxiteration-1;i++)
  {
    betaold=beta;
    XTyXXBeta=X.t()*(y-X*beta);
    for(int j=0;j<=p-1;j++){
      beta(j)=soft_th(beta(j)+XTyXXBeta(j)/(M*n),lambda/M);
    }
    ite=ite+1;
    //to calculate the object funtion for every beta
    f(i)=(norm(y-X*beta,2))*(norm(y-X*beta,2))/(2*n)+lambda*norm(beta,1);
    //if the diffenrence bwtween new beta and old beta in L1 norm less than 0.1 ,break and return
    if (norm(beta-betaold,1)<0.1){
    break;} 
    
  }
  //print the iteration number
  return Rcpp::List::create(Rcpp::Named("betahat")=beta,
                            Rcpp::Named("object function")=f,
                            Rcpp::Named("iteration")=ite);
}

