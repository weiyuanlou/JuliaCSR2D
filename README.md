# JuliaCSR2D
Computes the 2D transient CSR wakes using Julia.
The theory (https://journals.aps.org/prab/pdf/10.1103/PhysRevAccelBeams.24.064402) assumes a relativistic $\gamma$ and a constant bending radius $\rho$.  

Transient wake = entrance wake ( case A+B ) and exit wake ( case C+D+E ).

===================================================

To run Julia on Jupyter notebooks, follow https://datatofish.com/add-julia-to-jupyter/

===================================================
## Codes ##
-- All main codes are inside the "core/" directory.

-- Functions for wake computation are in core/integrator.jl. One can use CPU to compute the wake value at a single observation point with these functions. 

-- To compute the wake value on an enitre grid of observation points, parallel computation on GPU is recommended to save time. These GPU-required functions, along with the functions calculating the resultant CSR kicks at the particle positions, are in core/kick2.jl, which is the top-level code file.   

-- The simple_track.jl includes the tracking code for a hard-edge dipole and drift.

=================================================
## Examples ##

The most updated example is track_FACET.ipynb, which includes the code for chicane tracking with 2D CSR. These tracking codes will soon be migrated into the core directory for general use. When running this notebook, choose the Gaussian beam, and skip the ASTRA beam and FACET beam definition since the beam files are not open to public.  

All other example notebooks are to be updated.

More examples are to be added. 

