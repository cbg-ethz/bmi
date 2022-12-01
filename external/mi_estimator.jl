# A mutual information estimator using TransferEntropy.jl
# library in Julia
# Use as:
# julia external/mi_estimator.jl path_to_file_with_samples.csv seed dim_x dim_y
# Note that the CSV must be formatted as seed, X1, ..., Xn, Y1, ..., Yn
using CSV
using DataFrames
using TransferEntropy

samples_filename = ARGS[1]
seed = parse(Int, ARGS[2])
dim_x = parse(Int, ARGS[3])
dim_y = parse(Int, ARGS[4])


df = CSV.File(samples_filename) |> DataFrame

df = df[df.seed .== seed, :] 

x_samples = Dataset(Matrix(df[:, 2:1+dim_x]))
y_samples = Dataset(Matrix(df[:, 2+dim_x: 1+dim_x+dim_y]))

# show(x_samples)
# show(y_samples)


# mi = mutualinfo(x_samples, y_samples, Kraskov())
# println(mi)


mi = mutualinfo(x_samples, y_samples, Kraskov1())
println(mi)


# mi = mutualinfo(x_samples, y_samples, Kraskov2())
# println(mi)


# mi = mutualinfo(x_samples, y_samples, KozachenkoLeonenko())
# println(mi)


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

