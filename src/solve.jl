function rilt(t,y,smin,smax,N,α=missing;kwargs...)
    s = smin * (smax / smin).^range(0, 1, length=N + 1)
   invert(t,y,s,(t,s) -> exp(-t*s),Tikhonov(α),kwargs...)
end
function invert(t,y,s,kernel::Function,aamethod::AutoRegMethod;kwargs...)
    reg = auto_reg(t,y,s,kernel::Function,aamethod;kwargs...)
    invert(t,y,s,kernel,reg;kwargs...)
end

"""
    invert(s,y,t,k::Function,reg;yoffset=true,tdomain=:realplus)

Compute the discretized form a(s) in y(t) = ∫a(s)k(t,s)ds. `s` and `y` represents 
the data wich we want to invert on discrete points `t`. 

# Examples
```julia-repl
julia> bar([1, 2], [1, 2])
1
```
"""

function invert(s,y,t,k::Function,reg;yoffset=true,kwargs...)
    AR,yr = build_ar(s,y,t,k,reg,yoffset)
    yt=get_yt(AR,yr;kwargs...)
    reshape
    return (t,yt[1:length(t),1],s,(AR*yt)[1:length(s),1])

end

function get_yt(AR,yr;tdomain=:realplus)

    if tdomain ==:real
        sy = (AR \ yr)
        return  sy
    end
    return nonneg_lsq(AR,yr;alg=:fnnls) 
 
end


function build_ar(t,y,s,kernel,reg,yoffset)
    #TODO: whole function does allocate like hell when AutoRegMethod is present
    #   cacheing must be implemented
    # central points
   
    sc=  (s[1:end - 1] + s[2:end]) / 2
    N=length(sc)
    ds = diff(s)
    dd=ones(length(ds))
    L = get_regularization_matrix(reg,N,dd)
    
    #Build Matrix approximation of kernel
    sType = eltype(s)
    lType = eltype(L)
    aType =promote_type(sType,lType,eltype(t),eltype(ds))
    A = aType[kernel(i, sc[j])*ds[j] for i in t, j in eachindex(sc)]

    if reg.α isa AbstractArray
        L= L*(Diagonal(reg.α))
    else    
        L= L*(Diagonal(reg.α*ones(aType,length(sc))))
    end

    if yoffset
        #add a ofset column
        A= hcat(A,ones(size(A,1)))
        #Do not regularize the y-offset
        L= hcat(L,zeros(size(L,1)))
       # push!(ds, )
        push!(sc, 0)
    end
 
    AR = vcat(A, L)

    # add  entry to y to store the regularization 
    yr = vcat(y, zeros(size(L,1 )))
    yr=convert.((eltype(AR),),yr) #allows for automatic dfferentation  NNLS

    return (AR,yr)
end

