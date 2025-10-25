
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

# Do at least 2 tests for lasso objective function below. You are checking output agreements on at least 2 separate inputs
#################################################


# Do at least 2 tests for fitLASSOstandardized function below. You are checking output agreements on at least 2 separate inputs
#################################################

# Do microbenchmark on fitLASSOstandardized vs fitLASSOstandardized_c
######################################################################

# Do at least 2 tests for fitLASSOstandardized_seq function below. You are checking output agreements on at least 2 separate inputs
#################################################

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
