# ===================================================================================
# A mutual information estimator using TransferEntropy.jl
# library in Julia
# Use as:
# $ julia external/mi_estimator.jl path_to_file_with_samples.csv dim_x dim_y
# Note that the CSV must be formatted as X1, ..., Xn, Y1, ..., Yn
#
# To install the dependencies run:
# $ julia
# > using Pkg
# > Pkg.add(["ArgParse", "CSV", "DataFrames", "TransferEntropy"])
# ===================================================================================
using ArgParse
using CSV
using DataFrames
using TransferEntropy


function get_args()
    allowed_estimators = ["KSG1", "KSG2", "Histogram", "Transfer", "Kernel"]

    settings = ArgParseSettings()

    @add_arg_table settings begin
    "samples"
        help = "The path to samples CSV in the format: X1, ..., Xm, Y1, ..., Yn."
        required = true
    "dim_x"
        help = "The dimension of the X variable."
        arg_type = Int
        required = true
    "dim_y"
        help = "The dimension of the Y variable."
        arg_type = Int
        required = true
    "--verbose"
        help = "Prints out additional information."
        action = :store_true
    "--neighbors"
        help = "The number of nearest neighbors to be used."
        arg_type = Int
        default = 10
    "--estimator"
        help = "Estimator type to be used. Allowed: $(allowed_estimators)"
        default = "KSG1"
    "--bins"
        help = "Number of bins per dimension for the histogram- and transfer- based estimators."
        arg_type = Int
        default = 10
    "--bandwidth"
        help = "Bandwidth for the kernel-based estimator."
        arg_type = Float64
        default = 1.0
    end

    args = parse_args(settings)

    # *** Validate the arguments related to the data ***
    # Validate whether the file exists
    if ! isfile(args["samples"])
        throw(ArgumentError("file does not seem to exist"))
    end
    # Validate the dimensions
    if min(args["dim_x"], args["dim_y"]) < 1
        throw(ArgumentError("the dimensions must be at least 1"))
    end

    # *** Validate the arguments related to the estimators ***
    # Validate the neighbors
    if args["neighbors"] < 1
        throw(ArgumentError("the neighbors argument must be at least 1"))
    end

    # Check whether the type of the estimator is right
    if ! (args["estimator"] in allowed_estimators)
        throw(ArgumentError("the estimator type is wrong"))
    end

    return args
end


function get_estimator(args::Dict, verbose::Bool)
    name = args["estimator"]
    k = args["neighbors"]
    bins = args["bins"]
    bandwidth = args["bandwidth"]

    if name == "KSG1"
        return Kraskov1(k)
    elseif name == "KSG2"
        return Kraskov2(k)
    elseif name == "Histogram"
        return VisitationFrequency(RectangularBinning(bins))
    elseif name == "Transfer"
        return TransferOperator(RectangularBinning(bins))
    elseif name == "Kernel"
        return NaiveKernel(bandwidth)
    else
        throw(ArgumentError("estimator not recognized"))
    end
end


function get_samples(filename::String, dim_x::Int, dim_y::Int, verbose::Bool)
    # Parse the file into a data frame
    df = CSV.File(filename) |> DataFrame

    if verbose
        println("Read $(size(df, 1)) rows from file $filename")
    end

    # Convert the right columns to Dataset
    x = Dataset(Matrix(df[:, 1:dim_x]))
    y = Dataset(Matrix(df[:, 1+dim_x: dim_x+dim_y]))

    if verbose
        println("X variable: $(size(x)). Y variable: $(size(y))")
    end

    return (x, y)
end


function main()
    args = get_args()

    # Flag whether to print auxiliary information or not
    verbose = args["verbose"]

    if verbose
        println("Settings:")
        println(args)
    end

    # Parse the samples
    x, y = get_samples(args["samples"], args["dim_x"], args["dim_y"], verbose)

    # Run the right mutual information estimator, basing on provided hyperparameters
    estimator = get_estimator(args, verbose)
    mi = mutualinfo(x, y, estimator, base=MathConstants.e)

    if verbose
        println("Estimated mutual information: $mi")
    else
        println(mi)
    end
end

main()

# est = VisitationFrequency(RectangularBinning(0.1))
# mi = mutualinfo(x_samples, y_samples, est)
# println(mi)


# est = VisitationFrequency(RectangularBinning(0.2))
# mi = mutualinfo(x_samples, y_samples, est)
# println(mi)


# est = VisitationFrequency(RectangularBinning(0.5))
# mi = mutualinfo(x_samples, y_samples, est)
# println(mi)

# est = VisitationFrequency(RectangularBinning(1))
# mi = mutualinfo(x_samples, y_samples, est)
# println(mi)

