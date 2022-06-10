include("integrator.jl")
include("deposit.jl")

function compute_Ws_case_B_GPU!(A::CuDeviceArray, Δ, charge_grid::CuDeviceArray, gamma, rho, phi, nxp, sigma_z, sigma_x)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx =  size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]

        z = dz*(ij[1]-1 - (nz-1)/2)
        x = dx*(ij[2]-1 - (nx-1)/2)
     
        #@inbounds A[i] = compute_Ws_case_D(z, x, 
        #    gamma=5000.0, rho = 1.5, phi_m=0.2, lamb=0.02, nxp=101, 
        #    charge_grid=charge_grid, sigma_z=50E-6, sigma_x=50E-6)
        
        @inbounds A[i] = compute_Ws_case_B(z, x, 
            gamma=gamma, rho=rho, phi=phi, nxp=nxp, 
            charge_grid=charge_grid, sigma_z=sigma_z, sigma_x=sigma_x)
    end
    
end

function compute_Ws_case_A_GPU!(A::CuDeviceArray, Δ, charge_grid::CuDeviceArray, gamma, rho, phi, nxp, sigma_z, sigma_x)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx =  size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]

        z = dz*(ij[1]-1 - (nz-1)/2)
        x = dx*(ij[2]-1 - (nx-1)/2)
        
        @inbounds A[i] = compute_Ws_case_A(z, x, 
            gamma=gamma, rho=rho, phi=phi, nxp=nxp, 
            charge_grid=charge_grid, sigma_z=sigma_z, sigma_x=sigma_x)
    end
    
end


function compute_Wx_case_B_GPU!(A::CuDeviceArray, Δ, charge_grid::CuDeviceArray, gamma, rho, phi, nxp, sigma_z, sigma_x)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx =  size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]

        z = dz*(ij[1]-1 - (nz-1)/2)
        x = dx*(ij[2]-1 - (nx-1)/2)
        
        @inbounds A[i] = compute_Wx_case_B(z, x, 
            gamma=gamma, rho=rho, phi=phi, nxp=nxp, 
            charge_grid=charge_grid, sigma_z=sigma_z, sigma_x=sigma_x)
    end
    
end

function compute_Wx_case_A_GPU!(A::CuDeviceArray, Δ, charge_grid::CuDeviceArray, gamma, rho, phi, nxp, sigma_z, sigma_x)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    
    dz, dx = Δ
    nz, nx =  size(A) 
    for i = index:stride:length(A)
        ij = @inbounds CartesianIndices(A)[i]

        z = dz*(ij[1]-1 - (nz-1)/2)
        x = dx*(ij[2]-1 - (nx-1)/2)
     
        @inbounds A[i] = compute_Wx_case_A(z, x, 
            gamma=gamma, rho=rho, phi=phi, nxp=nxp, 
            charge_grid=charge_grid, sigma_z=sigma_z, sigma_x=sigma_x)
    end
    
end






function csr2d_kick_calc_case_B(z_b, x_b, weight,
    gamma, rho, phi,
    nz, nx,
    sigma_z, sigma_x)
    
    zmin = -5*sigma_z
    zmax = 5*sigma_z
    
    xmin = -5*sigma_x
    xmax = 5*sigma_x
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    #zv = LinRange(-5*sigma_z, 5*sigma_z, nz);
    #xv = LinRange(-5*sigma_x, 5*sigma_x, nx);

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz, zmin, zmax, nx, xmin, xmax)
    
    norm = sum(charge_grid) *dz*dx
    lambda_grid = charge_grid ./ norm
    
    
    ##### Applying GPU ####
    println(" Computing Case B wake grid(s) via GPU...")
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    Ncu = length(Ws_grid)
    kernel = @cuda launch=false compute_Ws_case_B_GPU!(Ws_grid, Δ, lambda_grid_gpu, gamma, rho, phi, 201, 50E-6, 50E-6)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, gamma, rho, phi, 201, 50E-6, 50E-6; threads, blocks);
    Ws_grid_cpu = Array(Ws_grid)
    

    Wx_grid = CuArray{Float64}(undef, nz, nx);
    Ncu = length(Wx_grid)
    kernel = @cuda launch=false compute_Wx_case_B_GPU!(Ws_grid, Δ, lambda_grid_gpu, gamma, rho, phi, 201, 50E-6, 50E-6)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, gamma, rho, phi, 201, 50E-6, 50E-6; threads, blocks);
    Wx_grid_cpu = Array(Wx_grid)
    
    #########################
    
    println(" Interpolating wake value at the particle positions...")
    
    Np = length(z_b)
    
    # Overall factor
    Nb = sum(weight) / e_charge
    kick_factor = r_e * Nb / gamma  # in m
    
    eval_Ws(id) = interp_will(z_b[id], x_b[id], Ws_grid_cpu, -5*sigma_z, 5*sigma_z, -5*sigma_x, 5*sigma_x)
    eval_Wx(id) = interp_will(z_b[id], x_b[id], Wx_grid_cpu, -5*sigma_z, 5*sigma_z, -5*sigma_x, 5*sigma_x)
    
    ddelta_ds = kick_factor * map(eval_Ws, collect(1:Np))
    dxp_ds = kick_factor * map(eval_Wx, collect(1:Np))
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
        "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, 
        "lambda_grid"=> lambda_grid )
    
    return dd
    
end



function csr2d_kick_calc_case_A(z_b, x_b, weight,
    gamma, rho, phi,
    nz, nx,
    sigma_z, sigma_x)
    
    zmin = -5*sigma_z
    zmax = 5*sigma_z
    
    xmin = -5*sigma_x
    xmax = 5*sigma_x
    
    dz = (zmax - zmin) / (nz - 1)
    dx = (xmax - xmin) / (nx - 1)
    Δ = (dz, dx)
    
    #zv = LinRange(-5*sigma_z, 5*sigma_z, nz);
    #xv = LinRange(-5*sigma_x, 5*sigma_x, nx);

    # Charge deposition
    println(" Applying charge deposition...")
    charge_grid = histogram_cic_2d(z_b, x_b, weight, nz, zmin, zmax, nx, xmin, xmax)
    
    norm = sum(charge_grid) *dz*dx
    lambda_grid = charge_grid ./ norm
    
    
    ##### Applying GPU ####
    println(" Computing Case A wake grid(s) via GPU...")
    
    lambda_grid_gpu = CuArray(lambda_grid);
    
    Ws_grid = CuArray{Float64}(undef, nz, nx);
    Ncu = length(Ws_grid)
    kernel = @cuda launch=false compute_Ws_case_A_GPU!(Ws_grid, Δ, lambda_grid_gpu, gamma, rho, phi, 201, 50E-6, 50E-6)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)

    CUDA.@time CUDA.@sync kernel(Ws_grid, Δ, lambda_grid_gpu, gamma, rho, phi, 201, 50E-6, 50E-6; threads, blocks);
    Ws_grid_cpu = Array(Ws_grid)
    

    Wx_grid = CuArray{Float64}(undef, nz, nx);
    Ncu = length(Wx_grid)
    kernel = @cuda launch=false compute_Wx_case_A_GPU!(Ws_grid, Δ, lambda_grid_gpu, gamma, rho, phi, 201, 50E-6, 50E-6)
    config = launch_configuration(kernel.fun)
    threads = min(Ncu, config.threads)
    blocks = cld(Ncu, threads)
    
    CUDA.@time CUDA.@sync kernel(Wx_grid, Δ, lambda_grid_gpu, gamma, rho, phi, 201, 50E-6, 50E-6; threads, blocks);
    Wx_grid_cpu = Array(Wx_grid)
    
    #########################
    
    println(" Interpolating wake value at the particle positions...")
    
    Np = length(z_b)
    
    # Overall factor
    Nb = sum(weight) / e_charge
    kick_factor = r_e * Nb / gamma  # in m
    
    eval_Ws(id) = interp_will(z_b[id], x_b[id], Ws_grid_cpu, -5*sigma_z, 5*sigma_z, -5*sigma_x, 5*sigma_x)
    eval_Wx(id) = interp_will(z_b[id], x_b[id], Wx_grid_cpu, -5*sigma_z, 5*sigma_z, -5*sigma_x, 5*sigma_x)
    
    ddelta_ds = kick_factor * map(eval_Ws, collect(1:Np))
    dxp_ds = kick_factor * map(eval_Wx, collect(1:Np))
    
    dd = Dict("ddelta_ds" => ddelta_ds, "dxp_ds" => dxp_ds, 
        "Ws_grid" => Ws_grid_cpu, "Wx_grid" => Wx_grid_cpu, 
        "lambda_grid"=> lambda_grid )
    
    return dd
    
end