include("kick2.jl")
include("simple_track.jl")
include("emit_calc.jl")


function track_bend_with_2d_csr!(beam, charges, p0c, gamma; L=0, rho=1.0, 
                            N_step=20, s0=0.0, 
                            nz=201, nx=101, nz_cg=200, nx_cg=100, M=7, nxp=301,  
                            reverse_bend=false,
                            CSR_on=true, energy_kick_on=true, xp_kick_on=true, CSR_1D_only=false, 
                            bend_name="the bend", keep_Pin=false, debug=true, save_all_P=false)
    
    #rho = 1/g  
    #beam, charges = particle_group_to_bmad(Pin, p0c = p0c)
    if keep_Pin
        beam_list = [beam]
        s_list = [s0]
        emit_list = [emit(beam)]
        emit_disp_free_list = [emit_dispersion_free(beam)]
    else
        beam_list = []
        s_list = Array{Float64}([])
        emit_list = Array{Float64}([])
        emit_disp_free_list = Array{Float64}([])   
    end
    
    lambda_grid_list = Matrix{Float64}[]
    zmin_list = Array{Float64}([])
    zmax_list = Array{Float64}([])
    xmin_list = Array{Float64}([])
    xmax_list = Array{Float64}([])
    Ws_grid_list = Matrix{Float64}[]
    Wx_grid_list = Matrix{Float64}[]
    s_at_kick_list = Array{Float64}([])
    emit_at_kick_list = Array{Float64}([])
    emit_disp_free_at_kick_list = Array{Float64}([])
    
    
    s = s0
    ds_step = L/N_step
    phi = 0
    
    theta = ds_step/2/rho
    if reverse_bend
        theta = -theta
    end

    for i in 1:N_step
        println("Tracking through ", bend_name, " in the ", i, "th step starting at s= " , s,'m' ) 

        ## track through a bend of length ds/2
        #beam = track(beam, p0c = p0c, L=ds_step/2, theta = ds_step/2/rho, g_err=g_err)
        
        temp = [track_a_bend(beam[:,i], p0c, L=ds_step/2, theta=theta) for i in 1:Np]
        beam = reduce(hcat,temp)
        phi += ds_step/2/rho
        
        ## Calculate CSR kicks to xp and delta
        ####===================================
    
        if (CSR_on)
            
            if (CSR_1D_only)
                println("Applying 1D s-s kick...( To be implemented )")
                #csr_data = csr1d_steady_state_kick_calc(beam[4,:], charges, nz=nz, rho=rho, normalized_units=False)
                #delta_kick = csr_data['denergy_ds']/(gamma*mec2)
                #beam[6] = beam[6] + delta_kick * ds_step
                
            else
                
                csr_data = csr2d_kick_calc_entrance(beam[5,:], beam[1,:], charges,
                            gamma=gamma, rho=rho, phi=phi,
                            nz=nz, nx=nx, nz_cg=nz_cg, nx_cg=nx_cg, M=M, nxp=nxp, 
                            reverse_bend=reverse_bend)
                
                if (energy_kick_on)
                    println("Applying energy kick...")
                    delta_kick = csr_data["ddelta_ds"]
                    beam[6,:] = beam[6,:] .+ delta_kick * ds_step
                end
                
                if (xp_kick_on)
                    println("Applying xp_kick...")
                    xp_kick = csr_data["dxp_ds"]
                    beam[2,:] = beam[2,:] .+ xp_kick * ds_step
                end
                
                append!(zmin_list, minimum(beam[5,:]))
                append!(zmax_list, maximum(beam[5,:]))
                append!(xmin_list, minimum(beam[1,:]))
                append!(xmax_list, maximum(beam[1,:]))
                
                append!(s_at_kick_list, s+ds_step/2)
                append!(emit_at_kick_list, emit(beam))
                append!(emit_disp_free_at_kick_list, emit_dispersion_free(beam))
                
            end
        end
    
        ####====================================

        ## track through a bend of length ds/2
        temp = [track_a_bend(beam[:,i], p0c, L=ds_step/2, theta=theta) for i in 1:Np]
        beam = reduce(hcat,temp)
        phi += ds_step/2/rho
    
        s += ds_step
        append!(s_list, s)
        append!(emit_list, emit(beam))
        append!(emit_disp_free_list, emit_dispersion_free(beam))
        
        # This might take memory
        if save_all_P
            push!(beam_list, beam)
        end
        
        if CSR_on && !CSR_1D_only
            push!(Ws_grid_list, csr_data["Ws_grid"])
            push!(Wx_grid_list, csr_data["Wx_grid"])
            push!(lambda_grid_list, csr_data["lambda_grid"])
            
        end
    end
    
    
    dd = Dict("beam_out" => beam, "s_list" => s_list, "s_at_kick_list" => s_at_kick_list, 
        "emit_list" => emit_list, "emit_disp_free_list" => emit_disp_free_list,
        "emit_at_kick_list" => emit_at_kick_list, "emit_disp_free_at_kick_list" => emit_disp_free_at_kick_list,
        "beam_list" => beam_list, "Ws_grid_list"=>Ws_grid_list, "Wx_grid_list"=>Wx_grid_list,
        "zmin_list"=>zmin_list, "zmax_list"=>zmax_list, "xmin_list"=>xmin_list, "xmax_list"=>xmax_list,
        "lambda_grid_list"=>lambda_grid_list)
    
    return dd
end