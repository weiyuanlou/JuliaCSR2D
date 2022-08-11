# Integrator for case ABCD

include("qts.jl")
include("brentq.jl")
include("fields.jl")
include("interp.jl")


function QTS_case_D(z_ob::Float64, x_ob::Float64, 
        gamma::Float64, rho::Float64, phi_m::Float64, lamb::Float64, xp::Float64, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension)
    
    beta = (1-1/gamma^2)^(1/2)
    
    lamb_b(z::Float64, x::Float64) = interp_will(z, x, charge_grid, zmin, zmax, xmin, xmax)
    
    if dimension == 1
        iii = z -> Es_case_D((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma, lamb)*lamb_b( z, xp )
    elseif dimension == 2
        iii = z -> Fx_case_D((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma, lamb)*lamb_b( z, xp )
    end

    ## boundary conditions
    chi = ( x_ob - xp )/rho
    zid = rho*(phi_m + lamb - beta*sqrt(lamb^2 + chi^2 + 4*(1 + chi)*sin(phi_m/2)^2 + 2*lamb*sin(phi_m)))
    zod = rho*(lamb - beta*sqrt(lamb^2 + chi^2))
    
#####################################
    # find critical alpha

    # These came from Es_case_D
    kap(alp::Float64) = sqrt(lamb^2 + chi^2 + 4*(1+chi)*sin(alp)^2 + 2*lamb*sin(2*alp))
    N2(alp::Float64) = lamb*cos(2*alp) + (1+chi)*sin(2*alp) - beta * kap(alp)    
    
    alp_crit2_found = true
    
    alp_crit2 = find_root_Will(N2, 0.0, 0.03, 2000)
    
    if alp_crit2 == -1.0
       alp_crit2_found = false
    end

    #println("alp_crit2: ", alp_crit2)
    
    alp_crit2_usable = false

    if alp_crit2_found == true
        z_crit2  = z_ob - 2*rho*(alp_crit2 + (lamb - beta*kap(alp_crit2))/2) 
        if (z_crit2 > z_ob - zid) && (z_crit2 < z_ob - zod)
            alp_crit2_usable = true
        end
    end        
        
    # TESTING
    if (alp_crit2_usable == true)
        i1 = QTS_will(iii, z_ob - zid, z_crit2, M)  
        i2 = QTS_will(iii, z_crit2, z_ob - zod, M) 
        return i1+i2 
    else
        if (- zid < 0.0) && (0.0 < - zod)      
            i1 = QTS_will(iii, z_ob - zid, z_ob, M) 
            i2 = QTS_will(iii, z_ob, z_ob - zod, M)
            return i1+i2
        else
            return QTS_will(iii, z_ob - zid, z_ob - zod, M) 
        end     
    end
end 


function QTS_case_B(z_ob::Float64, x_ob::Float64, 
        gamma::Float64, rho::Float64, phi::Float64, xp::Float64, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension)
    
    beta = (1-1/gamma^2)^(1/2)
    
    #zmin, xmin = mins
    #zmax, xmax = maxs
      
    lamb_b(z::Float64, x::Float64) = interp_will(z, x, charge_grid, zmin, zmax, xmin, xmax)
    
    if dimension == 1
        iii = z -> Es_case_B((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma)*lamb_b( z, xp )
    elseif dimension == 2
        iii = z -> Fx_case_B((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma)*lamb_b( z, xp )
    end

    #iii(z::Float64) =  Es_case_B((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma)*lamb_b( z, xp )
    
    ## boundary conditions
    chi = ( x_ob - xp )/rho
    zi = rho*(phi - beta*sqrt(chi^2 + 4*(1 + chi)*sin(phi/2)^2))
    zo = -beta*abs(x_ob - xp)

    # These came from Es_case_B
    kap(alp::Float64) = sqrt(chi^2 + 4*(1+chi)*sin(alp)^2)
    N2(alp::Float64) = (1+chi)*sin(2*alp) - beta*kap(alp)

    alp_crit2_found = true
    
    alp_crit2 = find_root_Will(N2, 0.0, 0.03, 2000)
    
    if alp_crit2 == -1.0
       alp_crit2_found = false
    end
 
    alp_crit2_usable = false

    if alp_crit2_found == true
        z_crit2  = z_ob - 2*rho*(alp_crit2 - beta*kap(alp_crit2)/2)  
        if (z_crit2 > z_ob - zi) && (z_crit2 < z_ob - zo)
            alp_crit2_usable = true
        end
    end        
    
    if (alp_crit2_usable == true)

        i1 = QTS_will(iii, z_ob - zi, z_crit2, M)  
        i2 = QTS_will(iii, z_crit2, z_ob - zo, M) 
        
        return i1+i2
        
    else
        if (- zi < 0.0) && (0.0 < - zo)
            
            i1 = QTS_will(iii, z_ob - zi, z_ob, M) 
            i2 = QTS_will(iii, z_ob, z_ob - zo, M) 
            return i1 + i2
            
        else
            #println("USING NO BP!!!")
            return QTS_will(iii, z_ob - zi, z_ob - zo, M) 
        end
    end
end 


function QTS_case_A(z_ob::Float64, x_ob::Float64, 
        gamma::Float64, rho::Float64, phi::Float64, xp::Float64, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)
    
    beta = (1-1/gamma^2)^(1/2)
    alp_ob = phi/2
    
    lamb_b(z::Float64, x::Float64) = interp_will(z, x, charge_grid, zmin, zmax, xmin, xmax)
    
    #sigma_x = 50E-6
    #sigma_z = 50E-6
    #lamb_2d(z::Float64, x::Float64) = 1/(2*pi*sigma_x*sigma_z)* exp(-z^2 / 2 / sigma_z^2 - x^2 / 2 / sigma_x^2)
    
    if dimension == 1
        iii = z -> Es_case_A((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma, alp_ob)*lamb_b(z,xp)
    elseif dimension == 2
        iii = z -> Fx_case_A((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma, alp_ob)*lamb_b(z,xp)
    end

    ## boundary conditions
    chi = ( x_ob - xp )/rho
    zi = rho*(phi - beta*sqrt(chi^2 + 4*(1 + chi)*sin(phi/2)^2))
    
    return QTS_will(iii, zmin, z_ob - zi, M)
    
end 


function QTS_case_C(z_ob::Float64, x_ob::Float64, 
        gamma::Float64, rho::Float64, phi_m::Float64, lamb::Float64, xp::Float64, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)
    
    beta = (1-1/gamma^2)^(1/2)
    alp_m = phi_m/2
    
    lamb_b(z::Float64, x::Float64) = interp_will(z, x, charge_grid, zmin, zmax, xmin, xmax)
    
    if dimension == 1
        iii = z -> Es_case_C((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma, alp_m, lamb)*lamb_b(z,xp)
    elseif dimension == 2
        iii = z -> Fx_case_C((z_ob-z)/2/rho, (x_ob-xp)/rho, gamma, alp_m, lamb)*lamb_b(z,xp)
    end

    ## boundary conditions
    chi = ( x_ob - xp )/rho
    zid = rho*(phi_m + lamb - beta*sqrt(lamb^2 + chi^2 + 4*(1 + chi)*sin(phi_m/2)^2 + 2*lamb*sin(phi_m)))
    
    return QTS_will(iii, zmin, z_ob - zid, M)
    
end 

## Currently obsolete
function QTS_case_E(z_ob::Float64, x_ob::Float64, 
        gamma::Float64, rho::Float64, lamb::Float64, xp::Float64, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)

    beta = (1-1/gamma^2)^(1/2)
    
    lamb_b(z::Float64, x::Float64) = interp_will(z, x, charge_grid, zmin, zmax, xmin, xmax)    

    
    if dimension == 1
        iii = z -> Es_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_b(z,xp)
    elseif dimension == 2
        iii = z -> Fx_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_b(z,xp)
    end
    
    ## integral    
    # iii(z::Float64) = psi_s_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_2d_dz(z,xp)
 
    ## boundary conditions
    chi = ( x_ob - xp )/rho
    zod = rho*(lamb - beta*sqrt(lamb^2 + chi^2))
    z_near = -beta*abs(x_ob - xp)

    # if z_near is too close to zero, the integral blows up
    if z_near > -1E-16
        z_near = -1E-16
    end
    
    i1 = QTS_will(iii, z_ob - zod, z_ob-1E-16, M)
    i2 = QTS_will(iii, z_ob + 1E-16, z_ob - z_near, M)
  
    return (i1+i2)
    
    #return (i1+i2)*(-1.0/gamma^2)
end 


####################################################

####################################################

function compute_wake_case_D(z_ob::Float64, x_ob::Float64; 
        gamma::Float64, rho::Float64, phi_m::Float64, lamb::Float64, nxp::Int, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)

    xp_min = xmin
    xp_max = xmax
    dxp = (xp_max - xp_min)/(nxp-1)
    xp_vec = xp_min:dxp:xp_max
    
    sum = 0.0
    
    for i in 1:1:length(xp_vec)

        sum += QTS_case_D(z_ob, x_ob, gamma, rho, phi_m, lamb, xp_vec[i], M,
                            charge_grid, zmin, zmax, xmin, xmax, dimension)
        
    end
    
    beta2 = 1-1/gamma^2
    
    return sum* dxp*beta2/rho^2
end


function compute_wake_case_B(z_ob::Float64, x_ob::Float64; 
        gamma::Float64, rho::Float64, phi::Float64, nxp::Int, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)
    
    xp_max = xmax
    xp_min = xmin
    dxp = (xp_max - xp_min)/(nxp-1)
    xp_vec = xp_min:dxp:xp_max
    
    sum = 0.0
    
    for i in 1:1:length(xp_vec)
        sum += QTS_case_B(z_ob, x_ob, gamma, rho, phi, xp_vec[i], M,
                            charge_grid, zmin, zmax, xmin, xmax, dimension) 
    end
    
    beta2 = 1-1/gamma^2
    
    return sum* dxp*beta2/rho^2
end





function compute_Ws_case_B(z_ob::Float64, x_ob::Float64; 
        gamma::Float64, rho::Float64, phi::Float64, nxp::Int, M::Int, 
        charge_grid, zmin, zmax, xmin, xmax)
    
    xp_max = xmax
    xp_min = xmin
    dxp = (xp_max - xp_min)/(nxp-1)

    xp_vec = xp_min:dxp:xp_max
    
    sum = 0.0
    
    for i in 1:1:length(xp_vec)
        sum += QTS_case_B_Es(z_ob, x_ob, gamma, rho, phi, xp_vec[i], M,
                            charge_grid, zmin, zmax, xmin, xmax) 
    end
    
    beta2 = 1-1/gamma^2
    
    return sum* dxp*beta2/rho^2
end



function compute_wake_case_A(z_ob::Float64, x_ob::Float64; 
        gamma::Float64, rho::Float64, phi::Float64, nxp::Int, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)

    #xp_min = -5*50E-6 
    #xp_max = 5*50E-6
    xp_min = xmin 
    xp_max = xmax
    
    dxp = (xp_max - xp_min)/(nxp-1)
    xp_vec = xp_min:dxp:xp_max
    
    sum = 0.0
    
    for i in 1:1:length(xp_vec)
        sum += QTS_case_A(z_ob, x_ob, gamma, rho, phi, xp_vec[i], M,
                    charge_grid, zmin, zmax, xmin, xmax, dimension)   
    end
    
    return sum* dxp/rho^2/gamma^2
end



function compute_wake_case_C(z_ob::Float64, x_ob::Float64; 
        gamma::Float64, rho::Float64, phi_m::Float64, lamb::Float64, nxp::Int, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)

    xp_min = xmin 
    xp_max = xmax
    dxp = (xp_max - xp_min)/(nxp-1)
    xp_vec = xp_min:dxp:xp_max
    
    sum = 0.0
    
    for i in 1:1:length(xp_vec)
        sum += QTS_case_C(z_ob, x_ob, gamma, rho, phi_m, lamb, xp_vec[i], M,
                    charge_grid, zmin, zmax, xmin, xmax, dimension)   
    end
    
    return sum* dxp/rho^2/gamma^2
end


# Include both boundary terms and integral term
function compute_wake_case_E(z_ob::Float64, x_ob::Float64; 
        gamma::Float64, rho::Float64, lamb::Float64, nxp::Int, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)

    xp_min = xmin 
    xp_max = xmax
    dxp = (xp_max - xp_min)/(nxp-1)
    xp_vec = xp_min:dxp:xp_max
    
    sum = 0.0
    
    for i in 1:1:length(xp_vec)
        sum += QTS_case_E(z_ob, x_ob, gamma, rho, lamb, xp_vec[i], M,
                    charge_grid, zmin, zmax, xmin, xmax, dimension)   
    end
    
    return sum* (1/gamma^2)
end


function compute_wake_case_E_boundary(z_ob::Float64, x_ob::Float64; 
        gamma::Float64, rho::Float64, lamb::Float64, nxp::Int, M::Int,
        charge_grid, zmin, zmax, xmin, xmax, dimension::Int)
    
    beta = (1-1/gamma^2)^(1/2)
    
    lamb_b(z::Float64, x::Float64) = interp_will(z, x, charge_grid, zmin, zmax, xmin, xmax)    
    
  ## integral    
    #iii1(z::Float64) =  psi_s_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_2d_dz(z,xp)
 #   if dimension == 1
 #       iii = z -> Es_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_b(z,xp)
 #   elseif dimension == 2
 #       iii = z -> Fx_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_b(z,xp)
 #   end    
    
    chi(xp) = ( x_ob - xp )/rho
    zod(xp)  = rho*(lamb - beta*sqrt(lamb^2 + chi(xp)^2))
    z_near(xp) = -beta*abs(x_ob - xp) 
    
    L_far = rho*lamb
    psi_s_case_E_far(x) = 1/(sqrt(x^2 + L_far^2) - beta*L_far)
    #psi_s_case_E_near(x) = 1/abs(x)
    
    
    psi_x_case_E_far(x) = ( beta*x^2 - (1-beta^2)*L_far*sqrt(x^2 + L_far^2) )/ ( x*(x^2 + L_far^2*(1-beta^2)) )
    
    #sigma_x = 50E-6
    #sigma_z = 50E-6
    #lamb_2d(z, x) = 1/(2*pi*sigma_x*sigma_z)* exp(-z^2 / 2 / sigma_z^2 - x^2 / 2 / sigma_x^2) 
    
    if dimension == 1
        iii2 = xp -> psi_s_case_E_far( x_ob-xp )*lamb_b(z_ob-zod(xp), xp)
    elseif dimension == 2
        return 0.0
        #iii2 = xp -> psi_x_case_E_far( x_ob-xp )*lamb_b(z_ob-zod(xp), xp)
    end
        
    #iii2(xp) = psi_s_case_E_far( x_ob-xp )*lamb_b(z_ob-zod(xp), xp)
    #iii3(xp) = -psi_s_case_E_near( x_ob-xp )*lamb_2d(z_ob-z_near(xp), xp)
    
    
    ifar_1 = QTS_will(iii2, xmin, x_ob, M)       
    ifar_2 = QTS_will(iii2, x_ob, xmax, M)     
    
    #ifar = quadgk(xp -> iii2(xp), -5*sigma_x, 5*sigma_x,  rtol=1e-4)[1]
    
    #inear_1 = QTS_will(iii3, xmin, x_ob, 6)       
    #inear_2 = QTS_will(iii3, x_ob, xmax, 6)       
    
    #return (ifar_1 + ifar_2)*(-1.0/gamma^2)    
    #return (inear_1 + inear_2)*(-1.0/gamma^2)  
    #return (ifar_1 + ifar_2 + inear_1 + inear_2)*(-1.0/gamma^2)  
    
    return (ifar_1 + ifar_2)*(-1.0/gamma^2)  

end

# Simple version for ws only
function compute_Ws_case_E_simple(z_ob::Float64, x_ob::Float64; 
        gamma::Float64, rho::Float64, lamb::Float64)
    
    beta = (1-1/gamma^2)^(1/2)
    
    #lamb_b(z::Float64, x::Float64) = interp_will(z, x, charge_grid, zmin, zmax, xmin, xmax)    
    
  ## integral    
    #iii1(z::Float64) =  psi_s_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_2d_dz(z,xp)
 #   if dimension == 1
 #       iii = z -> Es_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_b(z,xp)
 #   elseif dimension == 2
 #       iii = z -> Fx_case_E((z_ob-z), (x_ob-xp), gamma)*lamb_b(z,xp)
 #   end    
    
    chi(xp) = ( x_ob - xp )/rho
    zod(xp)  = rho*(lamb - beta*sqrt(lamb^2 + chi(xp)^2))
    z_near(xp) = -beta*abs(x_ob - xp) 
    
    L_far = rho*lamb
    psi_s_case_E_far(x) = 1/(sqrt(x^2 + L_far^2) - beta*L_far)
    psi_s_case_E_near(x) = 1/abs(x)
    
    sigma_x = 50E-6
    sigma_z = 50E-6
    lamb_2d(z, x) = 1/(2*pi*sigma_x*sigma_z)* exp(-z^2 / 2 / sigma_z^2 - x^2 / 2 / sigma_x^2) 
    
    
    iii2(xp) = psi_s_case_E_far( x_ob-xp )*lamb_2d(z_ob-zod(xp), xp)
    iii3(xp) = -psi_s_case_E_near( x_ob-xp )*lamb_2d(z_ob-z_near(xp), xp)
    
    
    ifar_1 = QTS_will(iii2, -5*sigma_x, x_ob, 6)       
    ifar_2 = QTS_will(iii2, x_ob, 5*sigma_x, 6)     
    
    #ifar = quadgk(xp -> iii2(xp), -5*sigma_x, 5*sigma_x,  rtol=1e-4)[1]
    
    #inear_1 = QTS_will(iii3, -5*sigma_x, x_ob-1E-16, 6)       
    #inear_2 = QTS_will(iii3, x_ob+1E-16, 5*sigma_x, 6)       
    
    #return (ifar_1 + ifar_2)*(-1.0/gamma^2)    
    #return (inear_1 + inear_2)*(-1.0/gamma^2)  
    #return (ifar_1 + ifar_2 + inear_1 + inear_2)*(-1.0/gamma^2)  
    
    return (ifar_1 + ifar_2)*(-1.0/gamma^2)  

end