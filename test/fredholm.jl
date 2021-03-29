@testset "Fredholm.jl" begin
    Random.seed!(42)    
    F(t) = exp(-(t - 2)^2 / (2 * 0.3^2)) + exp(-(t - 3)^2 / (2 * 0.3^2))
    y(s) =  quadgk(t -> F(t) * exp(-t * s), 0, Inf, rtol=1e-6)[1] 

    s = 10.0.^(-2:0.05:1)  # generate discrete example data
    ys = map(y, s)         # from this is we want to approximate F(t)
    noise = (randn(length(ys))) * 0.001 

    ti = 0:0.05:5 |> collect # define the t-domain for the solution
    α = 6e-4
    regularizations = [Tikhonov, SecondDerivative]
    for r in regularizations
        t, yt = invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), r(α))
        @test norm(F.(ti) .- yt) < 1.5
        t, yt = invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), LCurve(r(α)))
        @test norm(F.(ti) .- yt) < 1.5
        t, yt = invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), XuPei(r(1), 0.95))
        @test norm(F.(ti) .- yt) < 1.5
    end

end


using Random, QuadGK, LinearAlgebra
    Random.seed!(42)    
    F(t) = exp(-(t - 2)^2 / (2 * 0.3^2)) + exp(-(t - 3)^2 / (2 * 0.3^2))
    y(s) =  quadgk(t -> F(t) * exp(-t * s), 0, Inf, rtol=1e-6)[1] 

    s = 10.0.^(-2:0.05:1)  # generate discrete example data
    ys = map(y, s)         # from this is we want to approximate F(t)
    noise = (randn(length(ys))) * 0.001 

    ti = 0:0.05:5 |> collect # define the t-domain for the solution
    α = 6.0e-4
    regularizations = [Tikhonov, SecondDerivative]
    for r in regularizations
        t, yt = invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), r(α))
        @test norm(F.(ti) .- yt) < 1.5
        t, yt = invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), LCurve(r(α)))
        @test norm(F.(ti) .- yt) < 1.5
        t, yt = invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), XuPei(r(1), 0.95))
        @test norm(F.(ti) .- yt) < 1.5
    end

using Logging
logger = SimpleLogger(stdout, Logging.Debug)
old_logger = global_logger(logger)

Fredholm. invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), Fredholm.LCurve(Fredholm.SecondDerivative()))