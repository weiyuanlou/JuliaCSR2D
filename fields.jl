########################
######## Case B ########
########################
function f_root_case_B(a::Float64, z::Float64, x::Float64, beta::Float64) 
    return a - beta/2 * sqrt(x^2 + 4*(1+x)*sin(a)^2) - z
end

function alpha_exact_case_B_brentq(z::Float64, x::Float64, beta::Float64)
    f_root_case_B_x(a) = f_root_case_B(a, z, x, beta)
    return brentq(f_root_case_B_x, -0.1, 1)
end

function Es_case_B(z::Float64, x::Float64, gamma::Float64)
    #if z == 0 && x == 0
    #    return 0  
    #end
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)

    alp = alpha_exact_case_B_brentq(z, x, beta)
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)
    
    kap = 2*(alp - z)/beta # kappa for case B
    
    N1 = cos2a - (1+x)
    N2 = (1+x)*sin2a - beta*kap
    #println(N1)
    #println(N2)
    
    D = kap - beta*(1+x)*sin2a
    #println(D)
    return N1*N2/D^3
end

function Fx_case_B(z::Float64, x::Float64, gamma::Float64)
    #if z == 0 && x == 0
    #    return 0  
    #end
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)

    alp = alpha_exact_case_B_brentq(z, x, beta)
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)
    
    kap = 2*(alp - z)/beta # kappa for case D
    
    N1 = sin2a - beta*kap
    N2 = (1+x)*sin2a - beta*kap
    D = kap - beta*(1+x)*sin2a
    
    
    ### SC term with prefactor 1/(gamma*beta)^2 = 1/(gamma^2-1)
    #NSC = (1 + beta2 - beta*kap*sin2a + x - cos2a*(1 + beta2*(1 + x)) ) / (gamma^2-1) 
    #out =  (1+x)*(N1*N2 + NSC)/D^3
    
    return (1+x)*N1*N2/D^3
end


function Fx_case_B_SC(z::Float64, x::Float64, gamma::Float64)
    #if z == 0 && x == 0
    #    return 0  
    #end
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)

    alp = alpha_exact_case_B_brentq(z, x, beta)
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)
    
    kap = 2*(alp - z)/beta # kappa for case D
    
    N1 = sin2a - beta*kap
    N2 = (1+x)*sin2a - beta*kap
    D = kap - beta*(1+x)*sin2a
    
    #return (1+x)*N1*N2/D^3
    
    ### SC term with prefactor 1/(gamma*beta)^2 = 1/(gamma^2-1)
    NSC = (1 + beta2 - beta*kap*sin2a + x - cos2a*(1 + beta2*(1 + x)) ) / (gamma^2-1) 
    out =  (1+x)*(N1*N2 + NSC)/D^3
    
    return out
end

########################
######## Case D ########
########################
function f_root_case_D(a::Float64, z::Float64, x::Float64, beta::Float64, lamb::Float64) 
    return a + 1/2 * (lamb - beta* sqrt(lamb^2 + x^2 + 4*(1+x)*sin(a)^2 + 2*lamb*sin(2*a))) - z
end


function alpha_exact_case_D_brentq(z::Float64, x::Float64, beta::Float64, lamb::Float64)
    f_root_case_D_x(a) = f_root_case_D(a, z, x, beta, lamb)
    return brentq(f_root_case_D_x, -0.1, 1)
end


function Es_case_D(z::Float64, x::Float64, gamma::Float64, lamb::Float64)
    #if z == 0 && x == 0
    #    return 0  
    #end
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)

    alp = alpha_exact_case_D_brentq(z, x, beta, lamb)
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)
    
    kap = (2*(alp - z) + lamb)/beta # kappa for case D
    
    N1 = cos2a - (1+x)
    N2 = lamb*cos2a + (1+x)*sin2a - beta*kap
    #println(N1)
    #println(N2)
    
    D = kap - beta*(lamb*cos2a + (1+x)*sin2a)
    #println(D)
    return N1*N2/D^3
end


function Fx_case_D(z::Float64, x::Float64, gamma::Float64, lamb::Float64)
    #if z == 0 && x == 0
    #    return 0  
    #end
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)

    alp = alpha_exact_case_D_brentq(z, x, beta, lamb)
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)
    
    kap = (2*(alp - z) + lamb)/beta # kappa for case D
    
    N1 = lamb + sin2a - beta*kap
    N2 = lamb*cos2a + (1+x)*sin2a - beta*kap
    D = kap - beta*(lamb*cos2a + (1+x)*sin2a)
    
    return (1+x)*N1*N2/D^3
end


########################
######## Case A ########
########################

function eta_case_A(z::Float64, x::Float64, beta2::Float64, alp::Float64)
    """
    Eq.(?) from Ref[1] slide 11
    "eta" here is H/rho, not to be confused with the eta function.
    "alp" here is half of the bending angle, not the alpha function.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    sin2a = sin(2*alp)
    
    a = (1-beta2)/4
    b = alp - z - beta2*(1+x)*sin2a/2
    c = alp^2 - 2*alp*z + z^2 - beta2*x^2/4 - beta2*(1+x)*sin(alp)^2
    
    return (-b + sqrt(b^2 - 4*a*c)) / (2*a)
    
end


function Es_case_A(z::Float64, x::Float64, gamma::Float64, alp::Float64)
    """
    Eq.(?) from Ref[2] with no constant factor e/gamma**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    'alp' is half the observational angle here.
    """

    #if z == 0 and x == 0 and alp==0:
    #    return 0
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 
    
    eta = eta_case_A(z, x, beta2, alp)
    kap = (2*(alp - z) + eta)/beta   # kappa for case A
    #kap = sqrt( eta**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*eta*(1+x)*sin2a) 
    
    N = sin2a + (eta - beta*kap)*cos2a
    D = kap - beta*(eta + (1+x)*sin2a)
    
    return N/D^3
    
end


function Fx_case_A(z::Float64, x::Float64, gamma::Float64, alp::Float64)
    """
    Eq.(?) from Ref[2] with no constant factor e/gamma**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    'alp' is half the observational angle here.
    """

    #if z == 0 and x == 0 and alp==0:
    #    return 0
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 
    
    eta = eta_case_A(z, x, beta2, alp)
    kap = (2*(alp - z) + eta)/beta   # kappa for case A
    #kap = sqrt( eta**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*eta*(1+x)*sin2a) 
    
    NEx = 1+x - cos2a + (eta - beta*kap)*sin2a 
    NBy = beta*( (1+x)*cos2a - 1)
    D = kap - beta*(eta + (1+x)*sin2a)
    
    return (1+x)*(NEx - beta*NBy)/D^3
end



########################
######## Case C ########
########################


function eta_case_C(z::Float64, x::Float64, beta2::Float64, alp::Float64, lamb::Float64)
    """
    Eq.(?) from Ref[1] slide 11
    "eta" here is H/rho, not to be confused with the eta function.
    "alp" here is half of the bending angle, not the alpha function.
    "lamb" is L/rho, where L is the bunch center location down the bending exit.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    """
    sin2a = sin(2*alp)
    cos2a = cos(2*alp)
    
    a = (1-beta2)/4
    b = alp - z + lamb/2 - lamb*beta2*cos2a/2 - beta2*(1+x)*sin2a/2
    c = alp^2 + alp*lamb + (1-beta2)*lamb^2/4 - 2*alp*z - lamb*z + z^2 - beta2*x^2/4 - beta2*(1+x)*sin(alp)^2 - lamb*beta2*sin2a/2
    
    return (-b + sqrt(b^2 - 4*a*c)) / (2*a)
end


function Es_case_C(z::Float64, x::Float64, gamma::Float64, alp::Float64, lamb::Float64)
    """
    Eq.(?) from Ref[2] with no constant factor e/gamma**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    'alp' is half the observational angle here.
    """

    #if z == 0 and x == 0 and alp == 0:
    #    return 0
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 
    eta = eta_case_C(z, x, beta2, alp, lamb)
    
    kap = (2*(alp - z) + eta + lamb)/beta # kappa for case C
    #kap = sqrt( lamb**2 + eta**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*(lamb + eta*(1+x))*sin2a + 2*lamb*eta*cos2a) 
    
    N = lamb + sin2a + (eta - beta*kap)*cos2a
    D = kap - beta*(eta + lamb*cos2a + (1+x)*sin2a)
    
    return N/D^3
        
end


function Fx_case_C(z::Float64, x::Float64, gamma::Float64, alp::Float64, lamb::Float64)
    """
    Eq.(?) from Ref[2] with no constant factor e/gamma**2/rho**2.
    Note that 'x' here corresponds to 'chi = x/rho', 
    and 'z' here corresponds to 'xi = z/2/rho' in the paper. 
    'alp' is half the observational angle here.
    """

    #if z == 0 and x == 0 and alp == 0:
    #    return 0
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)
    
    sin2a = sin(2*alp)
    cos2a = cos(2*alp) 
    eta = eta_case_C(z, x, beta2, alp, lamb)
    
    kap = (2*(alp - z) + eta + lamb)/beta # kappa for case C
    #kap = sqrt( lamb**2 + eta**2 + x**2 + 4*(1+x)*sin(alp)**2 + 2*(lamb + eta*(1+x))*sin2a + 2*lamb*eta*cos2a) 
    
    NEx = 1+x - cos2a + (eta - beta*kap)*sin2a 
    NBy = beta*( (1+x)*cos2a - 1 - lamb*sin2a )
    D = kap - beta*(eta + lamb*cos2a + (1+x)*sin2a)
    
    return (1+x)*(NEx - beta*NBy)/D^3
        
end

########################
######## Case E ########
########################


function Es_case_E(z::Float64, x::Float64, gamma::Float64)
    """
    Eq.(B5) from Ref[1] with no constant factor e**2/gamma**2.
    """
  
    #if z == 0 and x == 0:
    #    return 0
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)
    
    L = (z + beta*sqrt(x^2*(1-beta2) + z^2))/(1-beta2)
    
    S = sqrt(x^2 + L^2)
    N1 = L - beta * S
    D = S-beta*L
  
    return N1/D^3

end
    
function psi_s_case_E(z::Float64, x::Float64, gamma::Float64)
    """
    Eq.(B5) from Ref[1] with no constant factor 1/gamma**2.
    """
  
    #if z == 0 and x == 0:
    #    return 0
    
    beta2 = 1-1/gamma^2
    beta = sqrt(beta2)
    
    L = (z + beta*sqrt(x^2*(1-beta2) + z^2))/(1-beta2)
    
    return 1/(sqrt(x^2 + L^2) - beta*L)
end 