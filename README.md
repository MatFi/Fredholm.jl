# Fredholm


## Usage
 As an example, consider input data of the following form

 ```julia
using Fredholm, QuadGK, Random
Random.seed!(1234);

F(t) = exp(-(t - 2)^2 / (2 * 0.3^2)) + exp(-(t - 3)^2 / (2 * 0.3^2))
y(s) =  quadgk(t -> F(t) * exp(-t * s), 0, Inf, rtol=1e-6)[1] 

s = 10.0.^(-2:0.05:1)  # generate discrete example data
ys = map(y, s)         # from this we want to approximate F(t)
noise = (randn(length(ys))) * 0.001 

ti = 0:0.01:5|> collect #define the t-domain for the solution
α = 1.2e-4
t, yt, ss, yss= invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), Tikhonov(α))
```
The solution `yt` at discrete `t` will very much depend on the choice of the regularization parameter `α`. If more noise is present in the data a higher `α` should be picked and vice versa. The variables `ss` and `yss` contain the regularized form of `s` and `ys`, where `ss[end]` contians the `y-offset`. If `invert` is called with the keyword `yoffset=false` `ss` and `s` will be equal. 

![example](example.png)

To allow the solution to take also negative amplitudes use the `tdomain = :real` keyword

```julia
t, yt = invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), Tikhonov(α), tdomain=:real)
```

In cases that the regularization parameter `α` is not known beforehand one can estimate it via the L-Curve method by calling
```julia
t, yt = invert(s, ys .+ noise, ti, (t, s) -> exp(-t * s), LCurve(Tikhonov(α)), tdomain=:real)
```





