using CUDA
include("integrator.jl")
include("deposit.jl")


function compute_wake_case_B_GPU!(A::CuDeviceArray, Δ, charge_grid::CuDeviceArray, 
                            gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, dimension)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx =  size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]

        #z = dz*(ij[1]-1 - (nz-1)/2)
        #x = dx*(ij[2]-1 - (nx-1)/2)
        
        z = zmin + dz*(ij[1]-1)
        x = zmin + dx*(ij[2]-1)
         
        @inbounds A[i] = compute_wake_case_B(z, x, 
            gamma=gamma, rho=rho, phi=phi, nxp=nxp, 
            charge_grid=charge_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
    end
    
end


function compute_wake_case_A_GPU!(A::CuDeviceArray, Δ, charge_grid::CuDeviceArray, 
                            gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, dimension)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx =  size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]

        # Chris assumes grid symmetry about (0.0)
        #z = dz*(ij[1]-1 - (nz-1)/2)
        #x = dx*(ij[2]-1 - (nx-1)/2)
        
        z = zmin + dz*(ij[1]-1)
        x = zmin + dx*(ij[2]-1)
        
        
        @inbounds A[i] = compute_wake_case_A(z, x, 
            gamma=gamma, rho=rho, phi=phi, nxp=nxp, 
            charge_grid=charge_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
    end
    
end


function compute_wake_case_D_GPU!(A::CuDeviceArray, Δ, charge_grid::CuDeviceArray, 
                            gamma, rho, phi_m, lamb, nxp, zmin, zmax, xmin, xmax, dimension)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx =  size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]

        #z = dz*(ij[1]-1 - (nz-1)/2)
        #x = dx*(ij[2]-1 - (nx-1)/2)
        
        z = zmin + dz*(ij[1]-1)
        x = zmin + dx*(ij[2]-1)
        
        @inbounds A[i] = compute_wake_case_D(z, x, 
            gamma=gamma, rho=rho, phi_m=phi_m, lamb=lamb, nxp=nxp, 
            charge_grid=charge_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
    end
    
end


function compute_wake_case_C_GPU!(A::CuDeviceArray, Δ, charge_grid::CuDeviceArray, 
                            gamma, rho, phi_m, lamb, nxp, zmin, zmax, xmin, xmax, dimension)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx =  size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]

        #z = dz*(ij[1]-1 - (nz-1)/2)
        #x = dx*(ij[2]-1 - (nx-1)/2)
        
        z = zmin + dz*(ij[1]-1)
        x = zmin + dx*(ij[2]-1)
        
        @inbounds A[i] = compute_wake_case_C(z, x, 
            gamma=gamma, rho=rho, phi_m=phi_m, lamb=lamb, nxp=nxp, 
            charge_grid=charge_grid, zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax, dimension=dimension)
    end
    
end


#############################################


#############################################


function csr2d_kick_calc_case_B(z_b, x_b, weight,
    gamma, rho, phi,
    nz, nx, nxp)
    
    #zmin = -5*sigma_z
    zmin = minimum(z_b)
    zmax = maximum(z_b)
    xmin = minimum(x_b)
    xmax = maximum(x_b)
    
    #zmin = -5*50E-6
    #zmax = 5*50E-6
    #xmin = -5*50E-6
    #xmax = 5*50E-6
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    zv = LinRange(zmin, zmax, nz);
    xv = LinRange(xmin, xmax, nx);

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz, zmin, zmax, nx, xmin, xmax)
    
    nn = sum(charge_grid) *dz*dx
    lambda_grid = charge_grid ./ nn
    
    
    ##### Applying GPU ####
    println(" Computing Case B wake grid(s) via GPU...")
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    #Ncu = length(Ws_grid)
    Ncu = nz * nx
    
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    kernel = @cuda launch=false compute_wake_case_B_GPU!(Ws_grid, Δ, lambda_grid_gpu, 
                                        gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 1)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 1; threads, blocks)
    Ws_grid_cpu = Array(Ws_grid)
    

    Wx_grid = CuArray{Float64}(undef, nz, nx);

    #kernel = @cuda launch=false compute_wake_case_B_GPU!(Wx_grid, Δ, lambda_grid_gpu, 
    #                                    gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 2)
    #config = launch_configuration(kernel.fun)
    #threads = min(Ncu, config.threads)
    #blocks = cld(Ncu, threads)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 2; threads, blocks)
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
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
        "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, "zv" => zv,  "xv" => xv, 
        "lambda_grid"=> lambda_grid )
    
    return dd
    
end



function csr2d_kick_calc_case_D(z_b, x_b, weight,
    gamma, rho, phi_m, lamb,
    nz, nx, nxp)
    
    zmin = minimum(z_b)
    zmax = maximum(z_b)
    xmin = minimum(x_b)
    xmax = maximum(x_b)
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    zv = LinRange(zmin, zmax, nz);
    xv = LinRange(xmin, xmax, nx);

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz, zmin, zmax, nx, xmin, xmax)
    
    nn = sum(charge_grid) *dz*dx
    lambda_grid = charge_grid ./ nn
    
    
    ##### Applying GPU ####
    println(" Computing Case D wake grid(s) via GPU...")
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    #Ncu = length(Ws_grid)
    Ncu = nz * nx
    
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    kernel = @cuda launch=false compute_wake_case_D_GPU!(Ws_grid, Δ, lambda_grid_gpu, 
                                        gamma, rho, phi_m, lamb, nxp, zmin, zmax, xmin, xmax, 1)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi_m, lamb, nxp, zmin, zmax, xmin, xmax, 1; threads, blocks)
    Ws_grid_cpu = Array(Ws_grid)
    

    Wx_grid = CuArray{Float64}(undef, nz, nx);

    #kernel = @cuda launch=false compute_wake_case_B_GPU!(Wx_grid, Δ, lambda_grid_gpu, 
    #                                    gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 2)
    #config = launch_configuration(kernel.fun)
    #threads = min(Ncu, config.threads)
    #blocks = cld(Ncu, threads)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi_m, lamb, nxp, zmin, zmax, xmin, xmax, 2; threads, blocks)
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
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
        "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, "zv" => zv,  "xv" => xv, 
        "lambda_grid"=> lambda_grid )
    
    return dd
    
end




function csr2d_kick_calc_case_A(z_b, x_b, weight,
    gamma, rho, phi,
    nz, nx, nxp)
    
    zmin = minimum(z_b)
    zmax = maximum(z_b)
    xmin = minimum(x_b)
    xmax = maximum(x_b)
    
    #zmin = -5*50E-6
    #zmax = 5*50E-6
    #xmin = -5*50E-6
    #xmax = 5*50E-6
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    zv = LinRange(zmin, zmax, nz);
    xv = LinRange(xmin, xmax, nx);

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz, zmin, zmax, nx, xmin, xmax)
    
    nn = sum(charge_grid) *dz*dx
    lambda_grid = charge_grid ./ nn
    
    
    ##### Applying GPU ####
    println(" Computing Case A wake grid(s) via GPU...")
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    Ncu = nz*nx
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    kernel = @cuda launch=false compute_wake_case_A_GPU!(Ws_grid, Δ, lambda_grid_gpu, 
                                            gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 1)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 1; threads, blocks);
    Ws_grid_cpu = Array(Ws_grid)
    

    Wx_grid = CuArray{Float64}(undef, nz, nx);

    #kernel = @cuda launch=false compute_Wx_case_A_GPU!(Wx_grid, Δ, lambda_grid_gpu, 
    #                                        gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 2)
    #config = launch_configuration(kernel.fun)
    #threads = min(Ncu, config.threads)
    #blocks = cld(Ncu, threads)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 2; threads, blocks);
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
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
        "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, "zv" => zv,  "xv" => xv, 
        "lambda_grid"=> lambda_grid )
    
    return dd
    
end




function csr2d_kick_calc_case_C(z_b, x_b, weight,
    gamma, rho, phi_m, lamb,
    nz, nx, nxp)
    
    zmin = minimum(z_b)
    zmax = maximum(z_b)
    xmin = minimum(x_b)
    xmax = maximum(x_b)
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    zv = LinRange(zmin, zmax, nz);
    xv = LinRange(xmin, xmax, nx);

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz, zmin, zmax, nx, xmin, xmax)
    
    nn = sum(charge_grid) *dz*dx
    lambda_grid = charge_grid ./ nn
    
    
    ##### Applying GPU ####
    println(" Computing Case C wake grid(s) via GPU...")
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    #Ncu = length(Ws_grid)
    Ncu = nz * nx
    
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    kernel = @cuda launch=false compute_wake_case_C_GPU!(Ws_grid, Δ, lambda_grid_gpu, 
                                        gamma, rho, phi_m, lamb, nxp, zmin, zmax, xmin, xmax, 1)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi_m, lamb, nxp, zmin, zmax, xmin, xmax, 1; threads, blocks)
    Ws_grid_cpu = Array(Ws_grid)
    

    Wx_grid = CuArray{Float64}(undef, nz, nx);

    #kernel = @cuda launch=false compute_wake_case_B_GPU!(Wx_grid, Δ, lambda_grid_gpu, 
    #                                    gamma, rho, phi, nxp, zmin, zmax, xmin, xmax, 2)
    #config = launch_configuration(kernel.fun)
    #threads = min(Ncu, config.threads)
    #blocks = cld(Ncu, threads)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, 
                                gamma, rho, phi_m, lamb, nxp, zmin, zmax, xmin, xmax, 2; threads, blocks)
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
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
        "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, "zv" => zv,  "xv" => xv, 
        "lambda_grid"=> lambda_grid )
    
    return dd
    
end
