using CUDA
include("integrator.jl")
include("deposit.jl")


function compute_entrance_wake_GPU!(A::CuDeviceArray, Δ, lambda_grid::CuDeviceArray, 
                            gamma::Real, rho::Real, phi::Real, 
                            nxp::Int, M::Int,
                            zmin, zmax, xmin, xmax, dimension::Int)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx = size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]
   
        z = zmin + dz*(ij[1]-1)
        x = zmin + dx*(ij[2]-1)
        
        wA = compute_wake_case_A(z, x, 
            gamma=gamma, rho=rho, phi=phi, nxp=nxp, M=M-1, 
            lambda_grid=lambda_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
        
        wB = compute_wake_case_B(z, x, 
            gamma=gamma, rho=rho, phi=phi, nxp=nxp, M=M, 
            lambda_grid=lambda_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
        
        @inbounds A[i] = wA + wB
        
    end
    
end


## Using only the boundary term from case E
## OBSOLETE
function compute_exit_wake_GPU_boundary!(A::CuDeviceArray, Δ, lambda_grid::CuDeviceArray, 
                            gamma::Real, rho::Real, phi_m::Real, lamb::Real, 
                            nxp::Int, M::Int,
                            zmin, zmax, xmin, xmax, dimension::Int, include_case_C::Bool)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx = size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]
        
        z = zmin + dz*(ij[1]-1)
        x = zmin + dx*(ij[2]-1)
        
        wD = compute_wake_case_D(z, x, 
            gamma=gamma, rho=rho, phi_m=phi_m, lamb=lamb, nxp=nxp, M=M, 
            lambda_grid=lambda_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
        
        # Case E does NOT take phi_m
        wE = compute_wake_case_E_boundary_far(z, x, 
            gamma=gamma, rho=rho, lamb=lamb, nxp=nxp, M=M, 
            lambda_grid=lambda_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)

        
        if include_case_C
            wC = compute_wake_case_C(z, x, 
                gamma=gamma, rho=rho, phi_m=phi_m, lamb=lamb, nxp=nxp, M=M, 
                lambda_grid=lambda_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
            
            @inbounds A[i] = wC + wD + wE
            
        else
            @inbounds A[i] = wD + wE
        end
        
        #+compute_Ws_case_E_simple(z, x, gamma=gamma, rho=rho, lamb=lamb)  
    end
    
end



function compute_exit_wake_GPU!(A::CuDeviceArray, Δ, lambda_grid::CuDeviceArray, 
                            gamma::Real, rho::Real, phi_m::Real, lamb::Real, 
                            nxp::Int, M::Int,
                            zmin, zmax, xmin, xmax, dimension::Int, include_case_C::Bool)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx = size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]
        
        z = zmin + dz*(ij[1]-1)
        x = zmin + dx*(ij[2]-1)
           
        wD = compute_wake_case_D(z, x, 
            gamma=gamma, rho=rho, phi_m=phi_m, lamb=lamb, nxp=nxp, M=M, 
            lambda_grid=lambda_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
        
        # Case E does NOT take phi_m
        wE = compute_wake_case_E(z, x, 
            gamma=gamma, rho=rho, lamb=lamb, nxp=nxp, M=M, 
            lambda_grid=lambda_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
        
        if include_case_C
            wC = compute_wake_case_C(z, x, 
                gamma=gamma, rho=rho, phi_m=phi_m, lamb=lamb, nxp=nxp, M=M, 
                lambda_grid=lambda_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
            
            @inbounds A[i] = wC + wD + wE
            
        else
            @inbounds A[i] = wD + wE
        end

    end
    
end

#############################################

#############################################


function csr2d_kick_calc_entrance(z_b, x_b, weight;
    gamma::Real, rho::Real, phi::Real,
    nz::Int, nx::Int, nz_cg::Int, nx_cg::Int, M::Int, nxp::Int, 
    reverse_bend::Bool=false)
    
    if reverse_bend
        x_b = - x_b
    end
    
    #zmin = -5*sigma_z
    zmin = minimum(z_b)
    zmax = maximum(z_b)
    xmin = minimum(x_b)
    xmax = maximum(x_b)
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    zv = LinRange(zmin, zmax, nz)
    xv = LinRange(xmin, xmax, nx)

    ##### Charge grid calculation #####
    dz_cg = (zmax - zmin) / (nz_cg - 1)
    dx_cg = (xmax - xmin) / (nx_cg - 1)

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz_cg, zmin, zmax, nx_cg, xmin, xmax)
    
    # Normalize charge grid
    nn = sum(charge_grid) *dz_cg*dx_cg
    lambda_grid = charge_grid ./ nn
    
    ###################################
    
    
    ##### Applying GPU ####
    println(" Computing Case A+B wake grid(s) via GPU...")
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    Ncu = nz * nx
    
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    Wx_grid = CuArray{Float64}(undef, nz, nx);
    
    kernel = @cuda launch=false compute_entrance_wake_GPU!(Ws_grid, Δ, lambda_grid_gpu, 
                                        gamma, rho, phi, nxp, M,
                                        zmin, zmax, xmin, xmax, 1)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi, nxp, M, zmin, zmax, xmin, xmax, 1; threads, blocks)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi, nxp, M, zmin, zmax, xmin, xmax, 2; threads, blocks)
    
    Ws_grid_cpu = Array(Ws_grid)
    Wx_grid_cpu = Array(Wx_grid)
    
    #########################
    
    println(" Interpolating wake value at the particle positions...")
    
    Np = length(z_b)
    
    # Overall factor
    Nb = sum(weight) / e_charge
    kick_factor = r_e * Nb / gamma  # in m
    
    eval_Ws(id) = interp_will(z_b[id], x_b[id], Ws_grid_cpu, zmin, zmax, xmin, xmax)
    eval_Wx(id) = interp_will(z_b[id], x_b[id], Wx_grid_cpu, zmin, zmax, xmin, xmax)
    
    ddelta_ds = kick_factor * map(eval_Ws, collect(1:Np))
    dxp_ds = kick_factor * map(eval_Wx, collect(1:Np))
    
    if reverse_bend
        dxp_ds = -dxp_ds
    end
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
        "zv" => zv,  "xv" => xv, 
        "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, 
        "lambda_grid"=> lambda_grid )
    
    return dd
    
end


## OBSOLETE
function csr2d_kick_calc_exit_boundary(z_b, x_b, weight;
    gamma::Real, rho::Real, phi_m::Real, lamb::Real,
    nz::Int, nx::Int, nz_cg::Int, nx_cg::Int, M::Int, nxp::Int, 
    reverse_bend::Bool=false, include_case_C::Bool=true)

    if reverse_bend
        x_b = - x_b
    end

    zmin = minimum(z_b)
    zmax = maximum(z_b)
    xmin = minimum(x_b)
    xmax = maximum(x_b)
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    zv = LinRange(zmin, zmax, nz);
    xv = LinRange(xmin, xmax, nx);

    ##### Charge grid calculation #####
    dz_cg = (zmax - zmin) / (nz_cg - 1)
    dx_cg = (xmax - xmin) / (nx_cg - 1)

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz_cg, zmin, zmax, nx_cg, xmin, xmax)
    
    # Normalize charge grid
    nn = sum(charge_grid) *dz_cg*dx_cg
    lambda_grid = charge_grid ./ nn
    
    ###################################
    
    
    ##### Applying GPU ####
    if include_case_C
        println(" Computing Case C+D+E wake grid(s) via GPU...")
    else
        println(" Computing Case D+E wake grid(s) via GPU...")
    end
    
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    #Ncu = length(Ws_grid)
    Ncu = nz * nx
    
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    kernel = @cuda launch=false compute_exit_wake_GPU_boundary!(Ws_grid, Δ, lambda_grid_gpu, 
                                        gamma, rho, phi_m, lamb, nxp, M,
                                        zmin, zmax, xmin, xmax, 1, include_case_C)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi_m, lamb, nxp, M,
                                zmin, zmax, xmin, xmax, 1, include_case_C; threads, blocks)
    Ws_grid_cpu = Array(Ws_grid)
    

    Wx_grid = CuArray{Float64}(undef, nz, nx);

    #kernel = @cuda launch=false compute_wake_case_B_GPU!(Wx_grid, Δ, lambda_grid_gpu, 
    #                                    gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 2)
    #config = launch_configuration(kernel.fun)
    #threads = min(Ncu, config.threads)
    #blocks = cld(Ncu, threads)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi_m, lamb, nxp, M,
                                zmin, zmax, xmin, xmax, 2, include_case_C; threads, blocks)
    Wx_grid_cpu = Array(Wx_grid)
    
    #########################
    
    println(" Interpolating wake value at the particle positions...")
    
    Np = length(z_b)
    
    # Overall factor
    Nb = sum(weight) / e_charge
    kick_factor = r_e * Nb / gamma  # in m
    
    eval_Ws(id) = interp_will(z_b[id], x_b[id], Ws_grid_cpu, zmin, zmax, xmin, xmax)
    eval_Wx(id) = interp_will(z_b[id], x_b[id], Wx_grid_cpu, zmin, zmax, xmin, xmax)
    
    ddelta_ds = kick_factor * map(eval_Ws, collect(1:Np))
    dxp_ds = kick_factor * map(eval_Wx, collect(1:Np))
    
    if reverse_bend
        dxp_ds = -dxp_ds
    end
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
        "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, "zv" => zv,  "xv" => xv, 
        "lambda_grid"=> lambda_grid )
    
    return dd
    
end


function csr2d_kick_calc_exit(z_b, x_b, weight;
    gamma::Real, rho::Real, phi_m::Real, lamb::Real,
    nz::Int, nx::Int, nz_cg::Int, nx_cg::Int, M::Int, nxp::Int,
    reverse_bend::Bool=false, include_case_C::Bool=true)

    if reverse_bend
        x_b = - x_b
    end
    
    zmin = minimum(z_b)
    zmax = maximum(z_b)
    xmin = minimum(x_b)
    xmax = maximum(x_b)
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    zv = LinRange(zmin, zmax, nz)
    xv = LinRange(xmin, xmax, nx)

    ##### Charge grid calculation #####
    dz_cg = (zmax - zmin) / (nz_cg - 1)
    dx_cg = (xmax - xmin) / (nx_cg - 1)

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz_cg, zmin, zmax, nx_cg, xmin, xmax)
    
    # Normalize charge grid
    nn = sum(charge_grid) *dz_cg*dx_cg
    lambda_grid = charge_grid ./ nn
    
    
    
    
    ##### Applying GPU ####
    if include_case_C
        println(" Computing Case C+D+E wake grid(s) via GPU...")
    else
        println(" Computing Case D+E wake grid(s) via GPU...")
    end
    
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    Ncu = nz * nx
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    Wx_grid = CuArray{Float64}(undef, nz, nx);
    
    kernel = @cuda launch=false compute_exit_wake_GPU!(Ws_grid, Δ, lambda_grid_gpu, 
                                        gamma, rho, phi_m, lamb, nxp, M,
                                        zmin, zmax, xmin, xmax, 1, include_case_C)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi_m, lamb, nxp, M,
                                zmin, zmax, xmin, xmax, 1, include_case_C; threads, blocks)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi_m, lamb, nxp, M,
                                zmin, zmax, xmin, xmax, 2, include_case_C; threads, blocks)
    
    Ws_grid_cpu = Array(Ws_grid)
    Wx_grid_cpu = Array(Wx_grid)
    
    #########################
    
    println(" Interpolating wake value at the particle positions...")
    
    Np = length(z_b)
    
    # Overall factor
    Nb = sum(weight) / e_charge
    kick_factor = r_e * Nb / gamma  # in m
    
    eval_Ws(id) = interp_will(z_b[id], x_b[id], Ws_grid_cpu, zmin, zmax, xmin, xmax)
    eval_Wx(id) = interp_will(z_b[id], x_b[id], Wx_grid_cpu, zmin, zmax, xmin, xmax)
    
    ddelta_ds = kick_factor * map(eval_Ws, collect(1:Np))
    dxp_ds = kick_factor * map(eval_Wx, collect(1:Np))
    
    if reverse_bend
        dxp_ds = -dxp_ds
    end
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
            "zv" => zv,  "xv" => xv, 
            "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, 
            "lambda_grid"=> lambda_grid )
    
    return dd
    
end