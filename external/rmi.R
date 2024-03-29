# Wrapper script for estimating mutual information.
#
# Use as:
# $ Rscript FILE DIMX DIMY [optional hyperparameters]
#
# To install the requirements run:
# $ R
# > install.packages(c("argparse", "dplyr", "rmi", "BNSL"))

parser <- argparse::ArgumentParser()

parser$add_argument("FILE", help="Path to the CSV file.")
parser$add_argument("DIMX", type="integer", help="Dimension of the X variable.")
parser$add_argument("DIMY", type="integer", help="Dimension of the Y variable.")
parser$add_argument("--method", default="KSG1", help="Method to be used. Allowed: KSG1, KSG2, LNN, BNSL.")
parser$add_argument("--neighbors", type="integer", default=10, help="Number of neighbors (k) to be used.")
parser$add_argument("--alpha", type="double", default=0.65, help="Hyperparameter of the LNC method. It's ignored for KSG methods.")
parser$add_argument("--truncation", type="integer", default=30, help="Order of truncation for the LNN method.")
parser$add_argument("--proc", type="integer", default=0, help="The `proc` argument for the BNSL algorithm. Either 0, 1, 2 or 10.")

# Parse arguments
args <- parser$parse_args()

# Validate the arguments
if ((args$proc > 2 && args$proc != 10) || args$proc < 0) {
    stop(paste0("--proc must be 0, 1, 2 or 10, but was ", args$proc))
}

# Read the data frame
data <- read.csv(args$FILE)

splits <- c(args$DIMX, args$DIMY)

# TODO(Pawel): Validate the number of columns.

# Cast the data to matrix
data_matrix <- data.matrix(data)

# Construct the right list of hyperparameters
if (args$method == "LNC") {
    # We don't use LNC until we figure out how to pass the right alpha
    # hyperparameters and what their meaning is, as it's a vector
    stop("LNC method currently is not supported.")

    options <- list(method = args$method, k = args$neighbors, alpha = args$alpha)
    mi_estimate <- rmi::knn_mi(data_matrix, splits, options)
} else if (args$method == "KSG1" || args$method == "KSG2") {
    options <- list(method = args$method, k = args$neighbors)
    mi_estimate <- rmi::knn_mi(data_matrix, splits, options)
} else if (args$method == "LNN") {
    mi_estimate <- rmi::lnn_mi(data_matrix, splits, k = args$neighbors, tr = args$truncation)
} else if (args$method == "BNSL") {
    idx1 <- args$DIMX
    idx2 <- idx1 + 1
    idx3 <- args$DIMX + args$DIMY
    
    mi_estimate <- BNSL::mi(data_matrix[,1:idx1], data_matrix[,idx2:idx3], proc=args$proc)
} else {
    stop(paste0("Method ", args$method, " not recognized."))
}

cat(mi_estimate)
cat("\n")
