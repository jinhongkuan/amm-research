import Pkg;
Pkg.add("DataFrames");
Pkg.add("StatsModels");
Pkg.add("GLM");
Pkg.add("StatsBase");
Pkg.add("Printf");

using StatsModels, DataFrames, GLM, StatsBase, Printf

begin

    f = @formula(y ~ x1 + x2)

    df = DataFrame(x1=rand(100), x2=rand(100))
    df.y = 1.0 .+ 2.0 * df.x1 + 3.0 * df.x2 + randn(100) * 0.1

    lm = fit(LinearModel, f, df)

    println("r^2: ", r2(lm))
    println("adj. r^2: ", adjr2(lm))
    println("information criterion (AIC): ", aic(lm))
    cm = DataFrame(vcov(lm), :auto)
    coef_names = coefnames(lm)
    rename!(cm, Symbol.(:, coef_names))
    insertcols!(cm, 1, :Variable => repeat(coef_names, inner=size(cm, 1) ÷ length(coef_names)))
    println("Covariance matrix table:")
    println(cm[2:end, 3:end])
end
