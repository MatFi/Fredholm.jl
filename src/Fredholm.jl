module Fredholm
    using LinearAlgebra: I, ⋅, norm, Diagonal, mul!

    using NNLS
    using ForwardDiff
    abstract type AutoRegMethod end
    abstract type AbstractRegularization{T} end  

    include("regularizations.jl")
    include("autoreg.jl")
    include("solve.jl")
    
    export invert, XuPei,LCurve,Tikhonov,SecondDerivative
end # module