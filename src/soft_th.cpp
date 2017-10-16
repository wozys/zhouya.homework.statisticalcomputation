
double soft_th(const double& z, const double& lambda){
  if(z>lambda) return z-lambda;
  else{
    if(z<-lambda) return z+lambda;
    else return 0;
  }
}
