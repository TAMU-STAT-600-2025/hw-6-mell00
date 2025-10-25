#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Soft-thresholding function, returns scalar
// [[Rcpp::export]]
double soft_c(double a, double lambda){
  if (a >  lambda) return a - lambda;
  if (a < -lambda) return a + lambda;
  return 0.0;
}

// Lasso objective function, returns scalar
// [[Rcpp::export]]
double lasso_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& beta, double lambda){
  const double n = static_cast<double>(Xtilde.n_rows);
  arma::colvec r = Ytilde - Xtilde * beta;
  double data_fit = 0.5 * arma::dot(r, r) / n; // 1/(2n) * ||r||^2
  double l1 = arma::sum(arma::abs(beta));
  return data_fit + lambda * l1;
}

// helper function, one-lambda CD with scaling from lasso_c
// z_j = (1/n) * sum x_ij^2, rho = (1/n) * x_j^T r_added_back
static arma::colvec cd_one_lambda_scaled(const arma::mat& X, const arma::colvec& y, double lambda,
                                         const arma::colvec& beta_start, double eps){
  
  const arma::uword p = X.n_cols;
  const double n = static_cast<double>(X.n_rows);

  arma::colvec beta = beta_start; // working copy
  arma::colvec r = y - X * beta;  // residual, current
  arma::rowvec z = arma::sum(arma::square(X), 0) / n; // z_j
  
  // initial objective funct
  double f_prev = 0.5 * arma::dot(r, r) / n + lambda * arma::sum(arma::abs(beta));

  const int max_iter = 100000;
  for (int it = 0; it < max_iter; ++it) {
  
    for (arma::uword j = 0; j < p; ++j) {
      double zj = z[j];
      if (zj == 0.0) continue; // skip all-zero standardized column
    
      double bj_old = beta[j];
      const arma::colvec xj = X.col(j);
    
      // readd old contribution
      r += xj * bj_old;
    
      // rho = (1/n) x_j^T r
      double rho = arma::dot(xj, r) / n;
    
      // update
      double bj_new = soft_c(rho, lambda) / zj;
      beta[j] = bj_new;
      
      // remove new contribution
      r -= xj * bj_new;
    }
    
    // objective and convergence check (stop when first time diff < eps)
    double f_curr = 0.5 * arma::dot(r, r) / n + lambda * arma::sum(arma::abs(beta));
    if (std::abs(f_prev - f_curr) < eps) break;
    f_prev = f_curr;
    }
  return beta;
}

// Lasso coordinate-descent on standardized data with one lamdba. Returns a vector beta.
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, double lambda, const arma::colvec& beta_start, double eps = 0.001){
  return cd_one_lambda_scaled(Xtilde, Ytilde, lambda, beta_start, eps);
}  

// helper function analog to .cd_solve_precomp (from LassoFunctions.R)
static void cd_solve_precomp_c(const arma::mat& Xtilde, arma::colvec& r, arma::colvec& beta, const arma::rowvec& z,
                               const std::vector<arma::uword>& active, const double lambda, const double eps){
  
  if (active.empty()) return;

  const int max_iter = 100000;
  const double n = static_cast<double>(Xtilde.n_rows);

  for (int it = 0; it < max_iter; ++it) {
    double max_change = 0.0;
  
    for (arma::uword t = 0; t < active.size(); ++t) {
      const arma::uword j = active[t];
      const double zj = z[j];
      if (zj == 0.0) continue;
    
      const double bj_old = beta[j];
      const arma::colvec xj = Xtilde.col(j);
    
      // readd old contribution
      r += xj * bj_old;
    
      // rho_j = (1/n) x_j^T r
      const double rho = arma::dot(xj, r) / n;
    
      // soft-threshold update
      const double bj_new = soft_c(rho, lambda) / zj;
    
      // remove new contribution and store
      r -= xj * bj_new;
      beta[j] = bj_new;
    
      const double d = std::abs(bj_new - bj_old);
      if (d > max_change) max_change = d;
    }
    
    if (max_change < eps) break;
  }
}

// Lasso coordinate-descent on standardized data with supplied lambda_seq. 
// You can assume that the supplied lambda_seq is already sorted from largest to smallest, and has no negative values.
// Returns a matrix beta (p by number of lambdas in the sequence)
// [[Rcpp::export]]
arma::mat fitLASSOstandardized_seq_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& lambda_seq, double eps = 0.001){
  const arma::uword p = Xtilde.n_cols;
  const arma::uword L = lambda_seq.n_elem;
  
  arma::mat betas(p, L);
  arma::colvec beta = arma::zeros<arma::colvec>(p); // warm start from 0s

  for (arma::uword k = 0; k < L; ++k) {
    beta = cd_one_lambda_scaled(Xtilde, Ytilde, lambda_seq[k], beta, eps);
    betas.col(k) = beta;
  }
  return betas;
}