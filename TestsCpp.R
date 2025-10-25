
# Header for Rcpp and RcppArmadillo
library(Rcpp)
library(RcppArmadillo)

# Source your C++ funcitons
sourceCpp("LassoInC.cpp")

# Source your LASSO functions from HW4 (make sure to move the corresponding .R file in the current project folder)
source("LassoFunctions.R")

set.seed(123)
cat("\n=== Begin Tests (LASSO C++) ===\n")
n_ok <- 0L

# Small standardization helper for R
std_xy <- function(X, Y) {
  s <- standardizeXY(X, Y)
  list(X = s$Xtilde, Y = s$Ytilde)
}

# Do at least 2 tests for soft-thresholding function below. You are checking output agreements on at least 2 separate inputs
#################################################

{
  test_name <- "soft_c matches soft (mixed signs, lambda = 1.2)"
  a <- c(-3, -0.5, 0, 0.5, 3)
  lam <- 1.2
  r_out <- sapply(a, soft, lambda = lam)
  c_out <- sapply(a, soft_c, lambda = lam)
  if (!isTRUE(all.equal(r_out, c_out, tolerance = 1e-12)))
    stop(test_name, " (mismatch)")
  n_ok <- n_ok + 1L
  cat("TEST PASSED: ", test_name, "\n")
}

{
  test_name <- "soft_c matches soft (random vector, multiple lambdas)"
  a <- rnorm(100)
  for (lam in c(0, 0.01, 2.5, 10)) {
    r_out <- sapply(a, soft, lambda = lam)
    c_out <- sapply(a, soft_c, lambda = lam)
    if (!isTRUE(all.equal(r_out, c_out, tolerance = 1e-12)))
      stop(test_name, " (lambda = ", lam, ")")
  }
  n_ok <- n_ok + 1L
  cat("TEST PASSED: ", test_name, "\n")
}

# Do at least 2 tests for lasso objective function below. You are checking output agreements on at least 2 separate inputs
#################################################

{
  test_name <- "lasso_c matches lasso (random beta, lambda = 0.7)"
  n <- 60; p <- 15
  X <- matrix(rnorm(n * p), n, p); Y <- rnorm(n)
  s <- std_xy(X, Y); Xs <- s$X; Ys <- s$Y
  beta <- rnorm(p); lam <- 0.7
  r_val <- lasso(Xs, Ys, beta, lam)
  c_val <- lasso_c(Xs, Ys, beta, lam)
  if (!isTRUE(all.equal(r_val, c_val, tolerance = 1e-10)))
    stop(test_name, " (mismatch)")
  n_ok <- n_ok + 1L
  cat("TEST PASSED: ", test_name, "\n")
}

{
  test_name <- "lasso_c matches lasso (0 beta, lambda = 0.05)"
  n <- 60; p <- 15
  X <- matrix(rnorm(n * p), n, p); Y <- rnorm(n)
  s <- std_xy(X, Y); Xs <- s$X; Ys <- s$Y
  beta0 <- rep(0, p); lam <- 0.05
  r_val <- lasso(Xs, Ys, beta0, lam)
  c_val <- lasso_c(Xs, Ys, beta0, lam)
  if (!isTRUE(all.equal(r_val, c_val, tolerance = 1e-10)))
    stop(test_name, " (mismatch)")
  n_ok <- n_ok + 1L
  cat("TEST PASSED: ", test_name, "\n")
}

# Do at least 2 tests for fitLASSOstandardized function below. You are checking output agreements on at least 2 separate inputs
#################################################

{
  test_name <- "fitLASSOstandardized_c matches fitLASSOstandardized (zero start)"
  n <- 80; p <- 25
  X <- matrix(rnorm(n * p), n, p); Y <- rnorm(n)
  s <- std_xy(X, Y); Xs <- s$X; Ys <- s$Y
  lam <- 0.3; beta0 <- rep(0, p)
  r_beta <- fitLASSOstandardized(Xs, Ys, lam, beta0, eps = 1e-7)$beta
  c_beta <- fitLASSOstandardized_c(Xs, Ys, lam, beta0, eps = 1e-7)
  if (!isTRUE(all.equal(as.numeric(r_beta), as.numeric(c_beta), tolerance = 1e-6)))
    stop(test_name, " (mismatch)")
  n_ok <- n_ok + 1L
  cat("TEST PASSED: ", test_name, "\n")
}

{
  test_name <- "fitLASSOstandardized_c matches fitLASSOstandardized (random warm start)"
  n <- 80; p <- 25
  X <- matrix(rnorm(n * p), n, p); Y <- rnorm(n)
  s <- std_xy(X, Y); Xs <- s$X; Ys <- s$Y
  lam <- 0.3; beta1 <- rnorm(p, sd = 0.1)
  r_beta <- fitLASSOstandardized(Xs, Ys, lam, beta1, eps = 1e-7)$beta
  c_beta <- fitLASSOstandardized_c(Xs, Ys, lam, beta1, eps = 1e-7)
  if (!isTRUE(all.equal(as.numeric(r_beta), as.numeric(c_beta), tolerance = 1e-6)))
    stop(test_name, " (mismatch)")
  n_ok <- n_ok + 1L
  cat("TEST PASSED: ", test_name, "\n")
}

# Do microbenchmark on fitLASSOstandardized vs fitLASSOstandardized_c
######################################################################

library(microbenchmark)

cat("\n microbenchmark: fitLASSOstandardized vs fitLASSOstandardized_c \n")
{
  n <- 200; p <- 80
  X <- matrix(rnorm(n * p), n, p); Y <- rnorm(n)
  s <- std_xy(X, Y); Xs <- s$X; Ys <- s$Y
  lam <- 0.25; beta0 <- rep(0, p)
  print(
    microbenchmark(
      R_fit = fitLASSOstandardized(Xs, Ys, lam, beta0, eps = 1e-6)$beta,
      C_fit = fitLASSOstandardized_c(Xs, Ys, lam, beta0, eps = 1e-6),
      times = 30
    )
  )
}

# Do at least 2 tests for fitLASSOstandardized_seq function below. You are checking output agreements on at least 2 separate inputs
#################################################

{
  test_name <- "fitLASSOstandardized_seq_c matches fitLASSOstandardized_seq (25 lambdas)"
  n <- 100; p <- 30
  X <- matrix(rnorm(n * p), n, p); Y <- rnorm(n)
  s <- std_xy(X, Y); Xs <- s$X; Ys <- s$Y
  n_cur <- nrow(Xs)
  lam_max <- max(abs(drop(crossprod(Xs, Ys))) / n_cur)
  lam_seq <- seq(lam_max, lam_max * 0.1, length.out = 25)
  
  r_fit <- fitLASSOstandardized_seq(Xs, Ys, lambda_seq = lam_seq,
                                    n_lambda = length(lam_seq), eps = 1e-7)
  r_mat <- r_fit$beta_mat
  c_mat <- fitLASSOstandardized_seq_c(Xs, Ys, lam_seq, eps = 1e-7)
  
  if (!identical(dim(r_mat), dim(c_mat))) stop(test_name, " (dim mismatch)")
  if (!isTRUE(all.equal(as.numeric(r_mat), as.numeric(c_mat), tolerance = 1e-6)))
    stop(test_name, " (values mismatch)")
  n_ok <- n_ok + 1L
  cat("TEST PASSED: ", test_name, "\n")
}

{
  test_name <- "fitLASSOstandardized_seq_c matches fitLASSOstandardized_seq
                (12 lambdas AND new dims)"
  n <- 120; p <- 18
  X <- matrix(rnorm(n * p), n, p); Y <- rnorm(n)
  s <- std_xy(X, Y); Xs <- s$X; Ys <- s$Y
  n_cur <- nrow(Xs)
  lam_max <- max(abs(drop(crossprod(Xs, Ys))) / n_cur)
  lam_seq2 <- seq(lam_max, lam_max * 0.2, length.out = 12)
  
  r_fit2 <- fitLASSOstandardized_seq(Xs, Ys, lambda_seq = lam_seq2,
                                     n_lambda = length(lam_seq2), eps = 1e-7)
  r_mat2 <- r_fit2$beta_mat
  c_mat2 <- fitLASSOstandardized_seq_c(Xs, Ys, lam_seq2, eps = 1e-7)
  
  if (!identical(dim(r_mat2), dim(c_mat2))) stop(test_name, " (dim mismatch)")
  if (!isTRUE(all.equal(as.numeric(r_mat2), as.numeric(c_mat2), tolerance = 1e-6)))
    stop(test_name, " (values mismatch)")
  n_ok <- n_ok + 1L
  cat("TEST PASSED: ", test_name, "\n")
}

# Do microbenchmark on fitLASSOstandardized_seq vs fitLASSOstandardized_seq_c
######################################################################

# Tests on riboflavin data
##########################
require(hdi) # this should install hdi package if you don't have it already; otherwise library(hdi)
data(riboflavin) # this puts list with name riboflavin into the R environment, y - outcome, x - gene erpression

# Make sure riboflavin$x is treated as matrix later in the code for faster computations
class(riboflavin$x) <- class(riboflavin$x)[-match("AsIs", class(riboflavin$x))]

# Standardize the data
out <- standardizeXY(riboflavin$x, riboflavin$y)

# This is just to create lambda_seq, can be done faster, but this is simpler
outl <- fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, n_lambda = 30)

# The code below should assess your speed improvement on riboflavin data
microbenchmark(
  fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, outl$lambda_seq),
  fitLASSOstandardized_seq_c(out$Xtilde, out$Ytilde, outl$lambda_seq),
  times = 10
)
