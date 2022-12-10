#  Use as
# $ Rscript SEED FILE DIMX DIMY [optional hyperparameters]
library("rmi")

parser <- argparse::ArgumentParser()

parser$add_argument("SEED", type="integer", help="Seed to be taken.")
parser$add_argument("FILE", help="Path to the CSV file.")
parser$add_argument("DIMX", type="integer", help="Dimension of the X variable.")
parser$add_argument("DIMY", type="integer", help="Dimension of the Y variable.")
parser$add_argument("--method", default="KSG1", help="Method to be used. Allowed: KSG1 KSG2 LNC")
parser$add_argument("--neighbors", type="integer", default=10, help="Number of neighbors (k) to be used.")
parser$add_argument("--alpha", type="double", default=0.65, help="Hyperparameter of the LNC method. It's ignored for KSG methods.")

# Parse arguments
args <- parser$parse_args()

# Read the data frame and filter the data corresponding to the right seed
data <- read.csv(args$FILE)
data <- dplyr::filter(data, data$seed==args$SEED)

splits <- c(args$DIMX, args$DIMY)

# Drop the seed column
data$seed <- NULL

# TODO(Pawel): Validate the number of columns.

# Cast the data to matrix
data_matrix <- data.matrix(data)

# Construct the right list of hyperparameters
if (args$method == "LNC") {
    options <- list(method = args$method, k = args$neighbors, alpha = args$alpha)
} else {
    # TODO(Pawel): Check whether it's KSG1 or KSG2. Otherwise, raise an exception.
    options <- list(method = args$method, k = args$neighbors)
}

mi_estimate <- rmi::knn_mi(data_matrix, splits, options)

cat(mi_estimate)
cat("\n")
