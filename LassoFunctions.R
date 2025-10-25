# Standardize X and Y: center both X and Y; scale centered X
# X - n x p matrix of covariates
# Y - n x 1 response vector
standardizeXY <- function(X, Y){
  
  # Basic checks
  if (is.null(dim(X))) stop("X must be a 2D matrix")
  if (!is.numeric(X)) stop("X must be numeric")
  if (!is.numeric(Y)) stop("Y must be numeric")
  n <- nrow(X); p <- ncol(X)
  if (length(Y) != n) stop("length of Y must match number of rows in X")
  if (n < 1 || p < 1) stop("X must have positive dimensions")
  if (anyNA(X) || anyNA(Y)) stop("missing inputs not supported")
  
  # Center Y
  Ymean <- mean(Y)
  Ytilde <- as.numeric(Y - Ymean)
  
  # Center X and compute weights per column
  Xmeans  <- colMeans(X)
  Xtilde  <- matrix(0, n, p)
  weights <- numeric(p)
  
  for (j in seq_len(p)) {
    xcj <- X[, j] - Xmeans[j]
    ssj <- sum(xcj * xcj) # == 0 for constant column after centering
    if (ssj == 0) {
      weights[j] <- 1 # EXACTLY 1 for constant columns
      Xtilde[, j] <- 0
    } else {
      wj <- sqrt(ssj / n)
      weights[j] <- wj
      Xtilde[, j] <- xcj / wj
    }
  }
  
  # Return:
  # Xtilde - centered and appropriately scaled X
  # Ytilde - centered Y
  # Ymean - the mean of original Y
  # Xmeans - means of columns of X (vector)
  # weights - defined as sqrt(X_j^{\top}X_j/n) after centering of X but before scaling
  return(list(Xtilde = Xtilde, Ytilde = Ytilde, Ymean = Ymean, Xmeans = Xmeans, weights = weights))
}

# Soft-thresholding of a scalar a at level lambda 
# [OK to have vector version as long as works correctly on scalar; will only test on scalars]
soft <- function(a, lambda){
  if (!is.numeric(a) || !is.numeric(lambda) || length(lambda) != 1L)
    stop("inputs must be numeric & lambda must be length 1")
  if (lambda < 0) stop("lambda must be non-negative")
  sign(a) * pmax(abs(a) - lambda, 0)
}

# Calculate objective function of lasso given current values of Xtilde, Ytilde, beta and lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba - tuning parameter
# beta - value of beta at which to evaluate the function
lasso <- function(Xtilde, Ytilde, beta, lambda){
  if (is.null(dim(Xtilde))) stop("Xtilde must be a 2D matrix")
  if (!is.numeric(Xtilde) || !is.numeric(Ytilde) || !is.numeric(beta))
    stop("all inputs must be numeric")
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if (length(Ytilde) != n) stop("length of Ytilde must = number of rows in Xtilde")
  if (length(beta) != p) stop("length of beta must = number of columns in Xtilde")
  if (!is.numeric(lambda) || length(lambda) != 1L || lambda < 0)
    stop("lambda must be a non-negative numeric scalar")
  
  resid <- as.numeric(Ytilde - Xtilde %*% beta)
  (as.numeric(crossprod(resid)) / (2 * n)) + lambda * sum(abs(beta))
}

# Fit LASSO on standardized data for a given lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1 (vector)
# lamdba - tuning parameter
# beta_start - p vector, an optional starting point for coordinate-descent algorithm
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized <- function(Xtilde, Ytilde, lambda, beta_start = NULL, eps = 0.001){
  #Check that n is the same between Xtilde and Ytilde
  if (is.null(dim(Xtilde))) stop("Xtilde must be a 2D matrix")
  if (!is.numeric(Xtilde) || !is.numeric(Ytilde))
    stop("Xtilde and Ytilde must be numeric")
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if (length(Ytilde) != n) stop("length of Ytilde must match number of rows in Xtilde")
  if (anyNA(Xtilde) || anyNA(Ytilde)) stop("missing values are not supported")
  
  # Check that lambda is non-negative
  
  if (!is.numeric(lambda) || length(lambda) != 1L || lambda < 0)
    stop("lambda must be a non-negative numeric scalar.")
  
  # Check for starting point beta_start. 
  # If none supplied, initialize with a vector of zeros.
  # If supplied, check for compatibility with Xtilde in terms of p
  
  if (is.null(beta_start)) {
    beta <- numeric(p)
  } else {
    if (!is.numeric(beta_start) || length(beta_start) != p)
      stop("beta_start must be a numeric vector of length ncol(Xtilde)")
    beta <- as.numeric(beta_start)
  }
  
  # Pre-compute column norms z_j = (1/n) * sum x_{ij}^2
  z <- colSums(Xtilde * Xtilde) / n
  
  # Initialize residual and objective
  r <- as.numeric(Ytilde - Xtilde %*% beta)
  l1 <- sum(abs(beta))
  f_prev <- (as.numeric(crossprod(r)) / (2 * n)) + lambda * l1
  
  # Cache columns to reduce repeated [, j] extraction overhead
  Xcols <- lapply(seq_len(p), function(j) Xtilde[, j])
  
  # Coordinate-descent implementation. 
  # Stop when the difference between objective functions is less than eps for the first time.
  # For example, if you have 3 iterations with objectives 3, 1, 0.99999,
  # your should return fmin = 0.99999, and not have another iteration
  
  repeat {
    for (j in seq_len(p)) {
      if (z[j] == 0) { next }  # skip if column is all 0s
      bj_old <- beta[j]
      xj <- Xcols[[j]]
      
      # re-add old contribution
      r <- r + xj * bj_old
      
      # partial residual correlation
      rho <- sum(xj * r) / n
      
      # soft-threshold update
      bj_new <- soft(rho, lambda) / z[j]
      beta[j] <- bj_new
      
      # update l1 incrementally
      l1 <- l1 + abs(bj_new) - abs(bj_old)
      
      # remove new contribution
      r <- r - xj * bj_new
    }
    
    # compute current objective
    f_curr <- (as.numeric(crossprod(r)) / (2 * n)) + lambda * l1
    
    # convergence check: stop at the first time the difference < eps
    if (abs(f_prev - f_curr) < eps) {
      fmin <- f_curr
      break
    }
    f_prev <- f_curr
  }
  
  # Return 
  # beta - the solution (a vector)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta, fmin = fmin))
}

# helper function to perform fast coordinate-descent on possibly restricted active set
.cd_solve_precomp <- function(X, Xcols, Ytilde, z, lambda, beta_start, eps,
                              active = NULL, kkt_tol = 1e-7) {
  n  <- nrow(X); p <- ncol(X)
  beta <- as.numeric(beta_start)
  
  # initialize residual and l1
  r <- as.numeric(Ytilde - X %*% beta)
  l1 <- sum(abs(beta))
  
  # indices to sweep this round
  idx <- if (is.null(active)) seq_len(p) else sort(unique(active))
  
  repeat {
    f_prev <- (as.numeric(crossprod(r)) / (2 * n)) + lambda * l1
    for (j in idx) {
      if (z[j] == 0) next
      bj_old <- beta[j]
      xj <- Xcols[[j]]
      
      # re-add old contribution
      r <- r + xj * bj_old
      
      # partial residual correlation
      rho <- as.numeric(crossprod(xj, r)) / n
      
      # soft-threshold update
      bj_new <- soft(rho, lambda) / z[j]
      beta[j] <- bj_new
      
      # update l1 incrementally
      l1 <- l1 + abs(bj_new) - abs(bj_old)
      
      # remove new contribution
      r <- r - xj * bj_new
    }
    f_curr <- (as.numeric(crossprod(r)) / (2 * n)) + lambda * l1
    if (abs(f_prev - f_curr) < eps) break
  }
  
  # KKT check for variables outside idx
  if (length(idx) < p) {
    g <- as.numeric(crossprod(X, r)) / n # (1/n) X^T r
    outside <- setdiff(seq_len(p), idx)
    viol <- outside[ which(abs(g[outside]) > (lambda + kkt_tol)) ]
    if (length(viol)) {
      
      # expand active set and resolve
      idx <- sort(unique(c(idx, viol)))
      return(.cd_solve_precomp(X, Xcols, Ytilde, z, lambda, beta, eps,
                               active = idx, kkt_tol = kkt_tol))
    }
  }
  
  list(beta = beta,
       fmin = (as.numeric(crossprod(r)) / (2 * n)) + lambda * l1,
       r = r)
}

# Fit LASSO on standardized data for a sequence of lambda values. Sequential version of a previous function.
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence,
#             is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized_seq <- function(Xtilde, Ytilde, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  
  # Check that n is the same between Xtilde and Ytilde
  if (is.null(dim(Xtilde))) stop("Xtilde must be a 2D matrix")
  if (!is.numeric(Xtilde) || !is.numeric(Ytilde))
    stop("Xtilde and Ytilde must be numeric")
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if (length(Ytilde) != n) stop("length of Ytilde must match number of rows in Xtilde")
  if (anyNA(Xtilde) || anyNA(Ytilde)) stop("missing values not supported")
  if (!is.numeric(n_lambda) || length(n_lambda) != 1L || n_lambda < 1)
    stop("n_lambda must be a positive integer")
  
  # helper function to sanitize/sort lambda vectors
  .finalize_lambda <- function(v) {
    v <- v[is.finite(v) & v >= 0]        # keep finite, non-negative
    v <- sort(as.numeric(v), decreasing = TRUE) # enforce decreasing order
    v <- unique(v)                        # drop duplicates
    v
  }
  
  # Check for the user-supplied lambda-seq (see below)
  # If lambda_seq is supplied, only keep values that are >= 0,
  # and make sure the values are sorted from largest to smallest.
  # If none of the supplied values satisfy the requirement,
  # print the warning message and proceed as if the values were not supplied.
  
  used_supplied <- FALSE
  if (!is.null(lambda_seq)) {
    if (!is.numeric(lambda_seq)) stop("supplied lambda_seq must be numeric")
    lambda_seq <- .finalize_lambda(lambda_seq) # sort supplied sequence
    if (length(lambda_seq) == 0L) {
      warning("No non-negative values in supplied lambda_seq; computing lambda_seq")
    } else {
      used_supplied <- TRUE
    }
  }
  
  # If lambda_seq is not supplied, calculate lambda_max 
  # (the minimal value of lambda that gives zero solution),
  # and create a sequence of length n_lambda as
  # lambda_seq = exp(seq(log(lambda_max), log(0.01), length = n_lambda))
  
  if (!used_supplied) {
    
    # lambda_max = max_j |(1/n) Xtilde_j^T Ytilde|
    # compute cross-products
    lam_candidates <- abs(drop(crossprod(Xtilde, Ytilde))) / n
    lambda_max <- suppressWarnings(max(lam_candidates))
    
    
    if (!is.finite(lambda_max) || lambda_max < .Machine$double.eps) {
      # fallback -> all zeros path
      lambda_seq <- rep(0, n_lambda)
    } else {
      # geometric path relative to lambda_max (1% of lambda_max)
      lambda_min <- lambda_max * 0.01
      lambda_seq <- exp(seq(log(lambda_max), log(lambda_min), length.out = n_lambda))
    }
    
    
    lambda_seq <- .finalize_lambda(lambda_seq) # clean generated seq
    if (length(lambda_seq) == 0L) lambda_seq <- rep(0, n_lambda) # final fallback
  }
  
  
  # Ensure descending sort
  lambda_seq <- sort(lambda_seq, decreasing = TRUE) 
  
  # Apply fitLASSOstandardized going from largest to smallest lambda 
  # (make sure supplied eps is carried over). 
  # Use warm starts strategy discussed in class for setting the starting values.
  
  m <- length(lambda_seq)
  beta_mat <- matrix(0, nrow = p, ncol = m)
  fmin_vec <- numeric(m)
  beta_start <- rep(0, p)
  
  # Precompute once for the whole path
  z     <- colSums(Xtilde * Xtilde) / n
  Xcols <- lapply(seq_len(p), function(j) Xtilde[, j])
  r_prev <- as.numeric(Ytilde)  # residual at beta = 0
  g_prev <- as.numeric(crossprod(Xtilde, r_prev)) / n
  
  for (t in seq_len(m)) {
    lam <- lambda_seq[t]
    if (t == 1) {
      # for the first lambda, use support from zero-solution screening
      active <- which(abs(g_prev) >= lam)
      fit_t <- .cd_solve_precomp(Xtilde, Xcols, Ytilde, z, lam, beta_start, eps,
                                 active = active, kkt_tol = 1e-7)
    } else {
      lam_prev <- lambda_seq[t - 1L]
      # keep j with |g_prev_j| >= 2*lam - lam_prev, or previously active
      strong_keep <- which(abs(g_prev) >= (2 * lam - lam_prev))
      active <- union(which(beta_start != 0), strong_keep)
      fit_t <- .cd_solve_precomp(Xtilde, Xcols, Ytilde, z, lam, beta_start, eps,
                                 active = active, kkt_tol = 1e-7)
    }
    
    beta_mat[, t] <- fit_t$beta
    fmin_vec[t]   <- fit_t$fmin
    beta_start    <- fit_t$beta
    r_prev        <- fit_t$r
    g_prev        <- as.numeric(crossprod(Xtilde, r_prev)) / n
  }
  
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value
  # fmin_vec - length(lambda_seq) vector of corresponding objective function values at solution
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, fmin_vec = fmin_vec))
}

# Fit LASSO on original data using a sequence of lambda values
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  if (is.null(dim(X))) stop("X must be a 2D matrix")
  if (!is.numeric(X) || !is.numeric(Y))
    stop("X and Y must be numeric")
  n <- nrow(X); p <- ncol(X)
  if (length(Y) != n) stop("length of Y must = number of rows in X")
  if (anyNA(X) || anyNA(Y)) stop("missing values not supported")
  # Center and standardize X,Y based on standardizeXY function
  
  std <- standardizeXY(X, Y)
  Xtilde <- std$Xtilde; Ytilde <- std$Ytilde
  
  # Fit Lasso on a sequence of values using fitLASSOstandardized_seq
  # (make sure the parameters carry over)
  
  seq_fit <- fitLASSOstandardized_seq(Xtilde, Ytilde, lambda_seq = lambda_seq,
                                      n_lambda = n_lambda, eps = eps)
  
  # check—lambda sequence length must match solution columns
  if (length(seq_fit$lambda_seq) != ncol(seq_fit$beta_mat)) {
    stop(sprintf("internal error: length(lambda_seq)=%d != ncol(beta_mat)=%d",
                 length(seq_fit$lambda_seq), ncol(seq_fit$beta_mat)))
  }
  
  # Perform back scaling and centering to get original intercept and coefficient vector
  # for each lambda
  
  betat <- seq_fit$beta_mat
  m <- ncol(betat)
  beta_mat <- betat
  
  # back-scale: beta_original = beta_tilde / weights
  beta_mat <- sweep(beta_mat, 1, std$weights, FUN = "/")
  
  # intercept: beta0 = Ymean - sum_j Xmean_j * beta_j
  beta0_vec <- as.numeric(std$Ymean - crossprod(std$Xmeans, beta_mat))
  
  # isolate output lambda sequence
  lambda_seq_out <- seq_fit$lambda_seq
  
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  return(list(lambda_seq = lambda_seq_out, beta_mat = beta_mat, beta0_vec = beta0_vec))
}


# Fit LASSO and perform cross-validation to select the best fit
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# k - number of folds for k-fold cross-validation, default is 5
# fold_ids - (optional) vector of length n specifying the folds assignment (from 1 to max(folds_ids)), if supplied the value of k is ignored 
# eps - precision level for convergence assessment, default 0.001
cvLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, k = 5, fold_ids = NULL, eps = 0.001){
  
  if (is.null(dim(X))) stop("X must be a 2D matrix")
  if (!is.numeric(X) || !is.numeric(Y))
    stop("X and Y must be numeric")
  n <- nrow(X); p <- ncol(X)
  if (length(Y) != n) stop("length of Y must = number of rows in X")
  if (anyNA(X) || anyNA(Y)) stop("missing values not supported")
  if (!is.null(lambda_seq) && !is.numeric(lambda_seq)) stop("supplied lambda_seq must be numeric")
  if (!is.numeric(n_lambda) || length(n_lambda) != 1L || n_lambda < 1)
    stop("n_lambda must be a positive integer")
  if (!is.null(fold_ids)) {
    if (length(fold_ids) != n) stop("fold_ids must have length n")
    if (any(is.na(fold_ids))) stop("fold_ids must not contain NA")
    if (any(fold_ids < 1)) stop("fold_ids must be positive integers starting at 1")
    k <- max(fold_ids)
    if (k < 2) stop("at least 2 folds are required")
    
    # ensure no empty folds
    for (ff in 1:k) {
      if (!any(fold_ids == ff)) stop("each fold id from 1..max(fold_ids) must appear at least once")
    }
  } else {
    if (!is.numeric(k) || length(k) != 1L || k < 2) stop("k must be an integer >= 2")
    if (k > n) stop("k cannot exceed n")
    
    # random, roughly equal folds
    fold_ids <- sample(rep(1:k, length.out = n))
  }
  
  # Fit Lasso on original data using fitLASSO
  
  full_fit <- fitLASSO(X, Y, lambda_seq = lambda_seq, n_lambda = n_lambda, eps = eps)
  lambda_seq_used <- full_fit$lambda_seq
  m <- length(lambda_seq_used)
  
  # If fold_ids is NULL, split the data randomly into k folds.
  # If fold_ids is not NULL, split the data according to supplied fold_ids.
  
  if (is.null(fold_ids)) {
    if (!is.numeric(k) || length(k) != 1L || k < 2 || k != as.integer(k))
      stop("k must be an integer >= 2.")
    if (k > n) stop("k cannot exceed n.")
    
    # Balanced random assignment of folds (sizes differ by at most 1)
    perm <- sample.int(n)
    fold_ids <- integer(n)
    base_size <- n %/% k
    extras <- n %% k
    start <- 1L
    for (f in seq_len(k)) {
      take <- base_size + as.integer(f <= extras)
      if (take > 0L) {
        idx <- perm[start:(start + take - 1L)]
        fold_ids[idx] <- f
        start <- start + take
      }
    }
  } else {
    
    # Validate supplied fold_ids
    if (is.factor(fold_ids)) fold_ids <- as.integer(fold_ids)
    if (!is.numeric(fold_ids) || length(fold_ids) != n)
      stop("fold_ids must be a numeric vector of length n")
    if (any(is.na(fold_ids))) stop("fold_ids must not contain NA")
    if (any(fold_ids < 1) || any(fold_ids != floor(fold_ids)))
      stop("fold_ids must be positive integers starting at 1")
    
    k <- max(fold_ids)
    if (k < 2) stop("at least 2 folds are required")
    
    # Ensure every fold label 1..k appears at least once
    missing_levels <- setdiff(seq_len(k), sort(unique(fold_ids)))
    if (length(missing_levels) > 0L)
      stop("each fold id from 1..k must appear at least once")
  }
  
  # Calculate LASSO on each fold using fitLASSO,
  # and perform any additional calculations needed for CV(lambda) and SE_CV(lambda)
  
  fold_means <- matrix(NA_real_, nrow = k, ncol = m)
  
  for (fold in 1:k) {
    val_idx <- which(fold_ids == fold)
    tr_idx  <- setdiff(seq_len(n), val_idx)
    
    Xtr <- X[tr_idx, , drop = FALSE]
    Ytr <- Y[tr_idx]
    Xval <- X[val_idx, , drop = FALSE]
    Yval <- Y[val_idx]
    
    fit_tr <- fitLASSO(Xtr, Ytr, lambda_seq = lambda_seq_used,
                       n_lambda = length(lambda_seq_used), eps = eps)
    
    # predictions at each lambda on validation fold
    preds <- sweep(Xval %*% fit_tr$beta_mat, 2, fit_tr$beta0_vec, FUN = "+")
    se_mat <- sweep(preds, 1, Yval, FUN = "-") ^ 2
    
    # fold-average MSE for each lambda
    fold_means[fold, ] <- colMeans(se_mat)
  }
  
  cvm  <- colMeans(fold_means) # CV(lambda)
  cvse <- apply(fold_means, 2, sd) / sqrt(k) # SE_CV(lambda)
  
  # Find lambda_min
  idx_min <- which.min(cvm)
  lambda_min <- lambda_seq_used[idx_min]
  
  # Find lambda_1SE
  thresh <- cvm[idx_min] + cvse[idx_min]
  idx_1se <- which(cvm <= thresh)
  
  # choose the largest lambda that satisfies the 1SE rule
  lambda_1se <- lambda_seq_used[min(idx_1se)]
  
  # Return output
  # Output from fitLASSO on the whole data
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  # fold_ids - used splitting into folds from 1 to k (either as supplied or as generated in the beginning)
  # lambda_min - selected lambda based on minimal rule
  # lambda_1se - selected lambda based on 1SE rule
  # cvm - values of CV(lambda) for each lambda
  # cvse - values of SE_CV(lambda) for each lambda
  return(list(lambda_seq = lambda_seq_used, beta_mat = full_fit$beta_mat,
        beta0_vec  = full_fit$beta0_vec, fold_ids   = fold_ids, lambda_min = lambda_min, 
        lambda_1se = lambda_1se, cvm = cvm, cvse = cvse))
}
