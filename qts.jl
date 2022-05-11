function samplepoint_will(t::Float64)
    #println("t: ", t)
    sinht = sinh(t)
    ϕ = tanh(sinht*π/2)
    ϕ′ = (cosh(t)*π/2)/cosh(sinht*π/2)^2
    return ϕ, ϕ′
end


using LinearAlgebra: norm
function estimate_error_will(prevI, I)
    ε = eps(Float64)
    M = 20
    return M*norm(I - prevI)^2 + norm(I)*ε
end


function qts_will(f::Function, h0::Float64, maxlevel::Integer=6)

    #@assert maxlevel > 0
    #@assert h0 > 0
    
    T = Float64
    
    x0, w0 = samplepoint_will(0.0)  #origin
    Σ = f(x0)*w0
   
    k = 1
    while true
        t = k*h0
        xk, wk = samplepoint_will(t)  
        1 - xk ≤ eps(T) && break   # xk is too close to 1, the upper bound of integral
        wk ≤ floatmin(T) && break  # wk is too small, series trucated
        
        Σ += (f(xk) + f(-xk)) * wk
        k += 1    # step is either 1 (for level = 0) or 2

    end    
    I = h0*Σ
    E = zero(eltype(I))
    
    for level in 1:maxlevel
        
        k = 1
        h = h0/2^level
        while true
            t = k*h
            xk, wk = samplepoint_will(t)  
            1 - xk ≤ eps(T) && break   # xk is too close to 1, the upper bound of integral
            wk ≤ floatmin(T) && break  # wk is too small, series trucated

            Σ += (f(xk) + f(-xk)) * wk
            k += 2     # step is either 1 (for level = 0) or 2

        end     
        
        prevI = I
        I = h*Σ
        E = estimate_error_will(prevI, I)
        ###tol = max(norm(I)*rtol, atol)
        tol = norm(I)*sqrt(eps(T))
        !(E > tol) && break  
        #println("level:", level)
    end
        

    #h = h0/2^maxlevel
    #return h*Σ
    return I
    #return I, E

end


# In contrast to qts_will which integrates from -1 to 1
# QTS_will integrates from a to b
function QTS_will(f::Function, a::Float64, b::Float64, M::Int)
    s = (b + a)/2
    t = (b - a)/2

    #print("t:", t)
    #I, E = q.qts(u -> f(s + t*u); atol=atol/t, rtol=rtol)
    #I*t, E*t
    
    h0 = 1.0
    #maxlevel = 12 # A large maxlevel may be slow if the integral does not converge...
    I = qts_will(u-> f(s+t*u), h0, M)
    return I*t   

end 