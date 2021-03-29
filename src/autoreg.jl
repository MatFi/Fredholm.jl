
struct LCurve{T}  <: AutoRegMethod
    reg::T
end

function auto_reg(t,y,s,kernel::Function,method::LCurve;yoffset=true,tdomain=:realplus)
    creg = typeof(method.reg).name.wrapper
  
    if method.reg isa UnionAll
        λ_ini = 1
        creg = method.reg
    else
        λ_ini = method.reg.α
        creg = typeof(method.reg).name.wrapper
    end

    function obj(λ)
        reg=creg(λ)
        
        AR,yr = build_ar(t,y,s,kernel,reg,yoffset)
        #TODO: could reduce allocs , problem with forward diff
        # p=load!(o.problem, AR,yr)  
        sy = get_yt(AR,yr;tdomain=tdomain)
        ρ = (AR*sy)[1:length(y)].-y |>norm
        η =  sy[1:end-1] |> norm
        return (ρ ,η)
    end       
    #calculate curvatur
    function k(λ)
        dd=ForwardDiff.Dual{:a}(λ,one(1.))
        o=obj(dd)
        η =  o[2].value
        ρ =  o[1].value
        dη = o[2].partials[1]
        c= -2*η*ρ/dη*(λ^2*dη*ρ + 2*λ*η*ρ + λ^4*η*dη)/((λ^2*η^2+ ρ^2)^3/2 )
        @debug "L-curvature" κ=c λ=λ  
       # return c
        #supress negative curvatures
        if c<0 
            c=exp(c)
        else
            c=c+1
        end

        return c
    end

   # return k(λ_ini)
    #optimize on log scales
    λ_opt = 10. ^gradient_decent(x ->log(1/k(10.0 ^x)),ini=log10(λ_ini),d_ini=log10(λ_ini))
    @debug "found λ" λ=λ_opt
    return creg( λ_opt)
end

function gradient_decent(f;ini=1e-5,d_ini=ini,maxiters=40)
    k=f
    α_old =ini
    α_cur =ini+1
    α_new =ini+1
    dual=ForwardDiff.Dual{:c}(α_old,one(1.))
    kevel= k(dual)
    Δ_old = kevel.partials[1]
    dual=ForwardDiff.Dual{:c}(α_new,one(1.))
    kevel= k(dual)
    Δ_cur = kevel.partials[1]
    k_best= kevel.value
    k_cur=k_best
    α_opt=α_old
 
   for i  in 1:maxiters
    
        α_cur=α_new
        dual=ForwardDiff.Dual{:c}(α_cur,one(1.))
        kn= k(dual)

        Δ_cur = kn.partials[1]
        k_cur= kn.value

        # save best values
         if  k_cur< k_best
            k_best=k_cur
            α_opt=α_cur
        end
        γ=(abs(α_old-α_cur)*abs(Δ_cur-Δ_old))/((Δ_cur-Δ_old)^2+eps(Float64))#+1e-4
        α0=1
        α_new= clamp(α_cur-γ*Δ_cur,α_cur-α0,α_cur+α0)
        #α_new=clamp(α_cur-Δ_cur,α_cur-α0,α_cur+α0)
        α_old=α_cur
        Δ_old=Δ_cur
        i>1 &&γ<0.01 && abs(Δ_cur)<0.01 &&break
    end

    return α_opt
end


struct XuPei{T,S} <: AutoRegMethod
    reg::T
    λ::S  #shrinking parameter
end
XuPei(r::AbstractRegularization{T}) where {T}  = XuPei(r,0.9)

#from http://dx.doi.org/10.1016/j.flowmeasinst.2016.05.004
function auto_reg(t,y,s,kernel,method::XuPei;yoffset=true,tdomain=:realplus)
   
    creg = typeof(method.reg).name.wrapper
  
    λ =method.λ
    #define regularization vector
    if method.reg isa UnionAll
        Λ=ones(length(s)-1)*1e-3 
        creg = method.reg
    else
        Λ=ones(length(s)-1)*method.reg.α
        creg = typeof(method.reg).name.wrapper
    end
   
    res = Inf
    for i in 1:100
        reg=creg(Λ)
        AR,yr = build_ar(t,y,s,kernel,reg,yoffset)
        sy= get_yt(AR,yr;tdomain=:real)
        sy[sy[1:end].<0][1:end-1].=0
        res_new= norm((AR*sy)[1:length(y)]-y)
        reg_new = norm((AR*sy)[length(y)+1:end-1])

        if  res < res_new 
            break
        end

        res=res_new
        sy=sy[1:end-1]
      
        Λ[sy.>0] .=Λ[sy.>0]*λ
    end
    return creg(Λ)
end