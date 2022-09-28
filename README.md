# JuliaCSR2D
Computes the 2D transient CSR wakes using Julia.
The theory (https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.24.064402) assumes a relativistic $\gamma$ and a constant bending radius $\rho$.  
Description of codes and results are found in https://napac2022.vrws.de/papers/wepa04.pdf.

Transient wake = entrance wake ( case A+B ) and exit wake ( case C+D+E ).

===================================================

To run Julia on Jupyter notebooks, follow https://datatofish.com/add-julia-to-jupyter/

Required julia packages are listed in packages.jl 

===================================================
## Codes ##
-- All main codes are inside the "core/" directory.

-- Functions for wake computation are in core/integrator.jl. One can use CPU to compute the wake value at a single observation point with these functions. 

-- To compute wake values on a large grid of observation points, parallel computation on GPU is recommended. These GPU-required functions, along with the functions which calculate the resultant CSR $kicks$ at the particle positions, are available in core/kick2.jl.   

-- Tracking code for hard-edge dipoles and drifts and included in core/simple_track.jl.

=================================================
## Examples ##

(1) examples/convergence_test.ipynb: Check convergence of the computed wake at one observation point as numerical parameters $nxp$ and $M$ increase.
                                 Users can load in customized lattice parameters and beam file to check convergence. 
                                 
(2) examples/wake_grid_computation.ipynb: (GPU required) Compute the entrance/exit $W_s$ and $W_x$ wake grids using a set of lattice parameters and beam.               

(3) track_FACET.ipynb: includes the code for chicane tracking with 2D CSR. These tracking codes will soon be migrated into the core directory for general use. When running this notebook, choose the Gaussian beam, and skip the ASTRA beam and FACET beam definition since these beam files are not open to public.  

All other example notebooks are to be updated.


