mutable struct Tikhonov{T} <: AbstractRegularization{T}
    α::T
end
Tikhonov() =  Tikhonov

get_regularization_matrix(regop::Tikhonov{T},N,ds) where {T} =   Matrix(I(N))

mutable struct SecondDerivative{T} <: AbstractRegularization{T}
    α::T
end
SecondDerivative()=SecondDerivative

function get_regularization_matrix(regop::SecondDerivative{T},N,ds) where {T}
    α=regop.α
    L=zeros( N+2,N)
    for j in 3:N
        L[j,j]= 1/(ds[j]*(ds[j]+ds[j-1]))#
        L[j,j-1]= -2*1/(ds[j-1]*(ds[j]))#
        L[j,j-2]= 1/(ds[j-1]*(ds[j]+ds[j-1]))#
    end

    # if regop.lower_bc
         L[1,1]=1/ds[1]
         L[2,1]=-2*1/(ds[1]*(ds[2]))
         L[2,2]=1/ds[2]*(ds[1]+ds[2])
    # end
    # if regop.upper_bc
         L[N+1,N-1]=1/ds[N-1]*(ds[N]+ds[N-1])
         L[N+1,N]=-2*1/((ds[N]*ds[N-1]))
         L[N+2,N]=1/ds[N]
    # end
    return L
    #L[:,end] .=0  #Do not regularize on the y offset   
end