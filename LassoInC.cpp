#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

inline void axpy_inplace(double a, const double* x, double* y, arma::uword n) {
  for (arma::uword i = 0; i < n; ++i) y[i] += a * x[i];
}

inline double dot_raw(const double* x, const double* y, arma::uword n) {
  double s = 0.0;
  for (arma::uword i = 0; i < n; ++i) s += x[i] * y[i];
  return s;
}

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

static void cd_solve_precomp_c(const arma::mat& Xtilde, arma::colvec& r, arma::colvec& beta, const arma::rowvec& z,
                               const std::vector<const double*>& colptrs, const std::vector<arma::uword>& active,
                               const double lambda, const double eps)
{
  if (active.empty()) return;

  const arma::uword n = Xtilde.n_rows;
  const double invn = 1.0 / static_cast<double>(n);
  double* rptr = r.memptr();

  // Keep L1 in sync with beta (incremental updates)
  double l1 = arma::sum(arma::abs(beta));

  const int max_iter = 100000;
  for (int it = 0; it < max_iter; ++it) {
    // f_prev = (1/(2n))*||r||^2 + lambda*||beta||_1
    double sse = 0.0;
    for (arma::uword i = 0; i < n; ++i) sse += rptr[i] * rptr[i];
    const double f_prev = 0.5 * sse * invn + lambda * l1;
  
    for (arma::uword t = 0; t < active.size(); ++t) {
      const arma::uword j = active[t];
      const double zj = z[j];
      if (zj == 0.0) continue;
    
      const double bj_old = beta[j];
      const double* xj = colptrs[j];
    
      // r += xj * bj_old
      if (bj_old != 0.0) axpy_inplace(bj_old, xj, rptr, n);
    
      // rho = (1/n) * xj^T r
      const double rho = dot_raw(xj, rptr, n) * invn;
    
      // soft-threshold update
      const double bj_new = soft_c(rho, lambda) / zj;
    
      // L1 update, r -= xj * bj_new, store
      l1 += std::abs(bj_new) - std::abs(bj_old);
      if (bj_new != 0.0) axpy_inplace(-bj_new, xj, rptr, n);
      beta[j] = bj_new;
    }
  
    // f_curr, stopping
    double sse2 = 0.0;
    for (arma::uword i = 0; i < n; ++i) sse2 += rptr[i] * rptr[i];
    const double f_curr = 0.5 * sse2 * invn + lambda * l1;
  
    if (std::abs(f_prev - f_curr) < eps) break;
  }
}

// Lasso coordinate-descent on standardized data with one lamdba. Returns a vector beta.
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, double lambda, const arma::colvec& beta_start, double eps = 0.001){
  return cd_one_lambda_scaled(Xtilde, Ytilde, lambda, beta_start, eps);
}  

// helper function analog to .cd_solve_precomp (from LassoFunctions.R)
static void cd_solve_precomp_c(const arma::mat& Xtilde, arma::colvec& r, arma::colvec& beta, const arma::rowvec& z,
                               const std::vector<arma::uword>& active, const double lambda, const double eps)
{
  if (active.empty()) return;

  const int max_iter = 100000;
  const double n = static_cast<double>(Xtilde.n_rows);

  // Track L1 term for objective; sync with beta updates
  double l1 = arma::sum(arma::abs(beta));

  for (int it = 0; it < max_iter; ++it) {
    // f_prev = (1/(2n)) * ||r||^2 + lambda * ||beta||_1
    double f_prev = 0.5 * arma::dot(r, r) / n + lambda * l1;
  
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
    
      // update L1 incrementally
      l1 += std::abs(bj_new) - std::abs(bj_old);
    
      // remove new contribution and store
      r -= xj * bj_new;
      beta[j] = bj_new;
    }
  
    double f_curr = 0.5 * arma::dot(r, r) / n + lambda * l1;
    if (std::abs(f_prev - f_curr) < eps) break;  // match R's stopping rule
  }
}

// Lasso coordinate-descent on standardized data with supplied lambda_seq. 
// You can assume that the supplied lambda_seq is already sorted from largest to smallest, and has no negative values.
// Returns a matrix beta (p by number of lambdas in the sequence)
// [[Rcpp::export]]
arma::mat fitLASSOstandardized_seq_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& lambda_seq,
                                     double eps = 0.001)
{
  const arma::uword p = Xtilde.n_cols;
  const arma::uword L = lambda_seq.n_elem;
  const double n = static_cast<double>(Xtilde.n_rows);
  const double kkt_tol = 1e-7;

  arma::mat  beta_mat(p, L); // p x |lambda_seq|
  arma::colvec beta(p, arma::fill::zeros); // warm start
  arma::colvec r = Ytilde; // residual at beta = 0
  arma::rowvec z = arma::sum(arma::square(Xtilde), 0) / n;

  // gradient at current solution (g = (1/n) X^T r)
  arma::colvec g_prev = (Xtilde.t() * r) / n;

  std::vector<unsigned char> in_active(p, 0);
  std::vector<arma::uword>   active;
  active.reserve(std::min<arma::uword>(p, 512));

  for (arma::uword k = 0; k < L; ++k) {
    const double lambda    = lambda_seq[k];
    const double lambda_pr = (k == 0 ? lambda : lambda_seq[k - 1]);
    const double strong_thr = std::max(2.0 * lambda - lambda_pr, 0.0); // strong rule
  
    // strong set with previous support
    active.clear();
    std::fill(in_active.begin(), in_active.end(), 0);
    for (arma::uword j = 0; j < p; ++j) {
      if (std::abs(beta[j]) > 0.0 || std::abs(g_prev[j]) >= strong_thr) {
        in_active[j] = 1;
        active.push_back(j);
      }
    }
    
    // solve on active set
    cd_solve_precomp_c(Xtilde, r, beta, z, active, lambda, eps);
    
    // KKT check on complement; pull violators and re-solve if needed
    while (true) {
      arma::colvec g = (Xtilde.t() * r) / n; // gradient at current beta
      bool any_violation = false;

      for (arma::uword j = 0; j < p; ++j) {
        if (in_active[j]) continue;
        if (std::abs(g[j]) > lambda + kkt_tol) {
          in_active[j] = 1;
          active.push_back(j);
          any_violation = true;
        }
      }
    
      if (!any_violation) {
        g_prev = std::move(g); // cache for next lambda
        break;
      }
      cd_solve_precomp_c(Xtilde, r, beta, z, active, lambda, eps);
    }
  
    beta_mat.col(k) = beta; // store solution at this lambda
    // r already corresponds to current beta; reused for next step
  }

  return beta_mat;
}