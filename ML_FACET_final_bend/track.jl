using Plots
using LaTeXStrings

using DelimitedFiles

using Distributions, Random

import HDF5

#using JLD2, FileIO

const e_charge = 1.60217663E-19 
const r_e = 2.8179402895E-15
const mec2 = 510998.94999999995

# FACET-II  parameters

L_bend = 20.0
L_drift_side = 32.5
L_drift_middle = 32.5

rho = 1538.0
phi_m = L_bend/rho
gamma = 30e9/mec2
beta = (1-1/gamma^2)^(1/2)
p0c = gamma * beta * mec2;

sig_z = 40e-6

L0 = (24*sig_z*rho^2)^(1/3)

#beam_dat = readdlm("../FACET_chicane_tracking/FACET_beam_in.dat");
beam_dat = readdlm("FACET_beam_before_final_bend.dat");

#charges_dat = readdlm("FACET_chicane_tracking/FACET_beam_in_charges.dat");

b0 = beam_dat';
#charges = charges_dat;

Np = 1000000
Q = 2E-9

charges = ones(Np)*Q/Np;

# Check bunch length
sig_z = std(b0[5,:])

Nb = Q/e_charge

# Check characteristic CSR strength ( in 1/m^2 )
W0 = Nb* r_e * mec2 *(sig_z/abs(rho))^(2/3) / sig_z^2

#zmin = minimum(b0[5,:])
#zmax = maximum(b0[5,:])
#xmin = minimum(b0[1,:])
#xmax = maximum(b0[1,:])

include("../core/track_with_csr.jl")

function save_data(result, x_factor::Real,  z_factor::Real)

    #file_name = "result_x_factor_"*string(x_factor)*".h5"
    
    #file_name = "results_M_9_nxp_500/result_x_factor_"*string(x_factor)*".h5"
    
    file_name = "results_M_9_nxp_500/result_x_factor_"*string(x_factor)*"_z_factor_"*string(z_factor)*".h5"
    
    #file_name = "testing.h5"
    
    Ws_to_save = reshape(reduce(hcat, reshape(result["Ws_grid_list"], :, N_step)), (nz,nx,:) );
    Wx_to_save = reshape(reduce(hcat, reshape(result["Wx_grid_list"], :, N_step)), (nz,nx,:) );
    lambda_to_save = reshape(reduce(hcat, reshape(result["lambda_grid_list"], :, N_step)), (nz_cg,nx_cg,:) );

    println("Saving result for x_factor = "*string(x_factor)*"...")
    
    HDF5.h5open(file_name, "w") do file
        g = HDF5.create_group(file, "csr_result") # create a group
        g["zmin_list"] = result["zmin_list"]
        g["zmax_list"] = result["zmax_list"] 
        g["xmin_list"] = result["xmin_list"] 
        g["xmax_list"] = result["xmax_list"]
        g["s_at_kick_list"] = result["s_at_kick_list"]
        g["emit_at_kick_list"] = result["emit_at_kick_list"]
        g["emit_disp_free_at_kick_list"] = result["emit_disp_free_at_kick_list"]

        g["Ws_grid_list"] = Ws_to_save;
        g["Wx_grid_list"] = Wx_to_save;
        g["lambda_grid_list"] = lambda_to_save
        HDF5.attributes(file)["x_factor"] = x_factor

    end
end

# Numerical parameters
nz=201 
nx=101 

nz_cg=200 
nx_cg=100

M=9
nxp=500

N_step = 20

function track_and_save(x_factor::Real, z_factor::Real)

    # Scale the x of the beam based on x_factor
    b1 = copy(b0)
    b1[1,:] = x_factor * b0[1,:]
    b1[5,:] = z_factor * b0[5,:]
    
    # Tracking takes time
    result = track_bend_with_2d_csr!(b1, charges, p0c, gamma, L=L_bend, rho=rho, 
                            N_step=N_step, s0=0.0, 
                            nz=nz, nx=nx, M=M, nxp=nxp, nz_cg=nz_cg, nx_cg=nx_cg,
                            reverse_bend=false,
                            CSR_on=true, energy_kick_on=true, xp_kick_on=true, CSR_1D_only=false, 
                            bend_name="the bend", keep_Pin=true, debug=true, save_all_P=false)
    
    save_data(result, x_factor, z_factor)
    
end

#track_and_save(1.0,1.0)

#for i in 0.1:0.1:5.0
#    track_and_save(i, 0.2)
#end

my_device = parse(Int64, ARGS[1])
device!(my_device)
println("Using GPU: ", device())


x_factor_range = eval(Meta.parse(ARGS[2]))
z_factor = parse(Float64, ARGS[3])


for x_factor in x_factor_range
    println("Tracking for x_factor = ", x_factor,  ", z_factor = ", z_factor)
    track_and_save(x_factor, z_factor)
end

