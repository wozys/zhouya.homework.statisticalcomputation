// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// CoordinateDescent
Rcpp::List CoordinateDescent(const arma::mat& X, const arma::colvec& y, double lambda);
RcppExport SEXP zhouyahomework1_CoordinateDescent(SEXP XSEXP, SEXP ySEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(CoordinateDescent(X, y, lambda));
    return rcpp_result_gen;
END_RCPP
}
// ProximalOperator
Rcpp::List ProximalOperator(const arma::mat& X, const arma::colvec& y, double lambda);
RcppExport SEXP zhouyahomework1_ProximalOperator(SEXP XSEXP, SEXP ySEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(ProximalOperator(X, y, lambda));
    return rcpp_result_gen;
END_RCPP
}