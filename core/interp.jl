## This function is GPU compatible
function interp_will(z, x, grid, zmin, zmax, xmin, xmax)
###
    # Given a 2D grid g(zp,xp) with coordinates (zp, xp) defined by zmin, zmax, xmin, xmax, 
    # returns the value of f(z,x) via bilinear interpolation.
    # The grid is assumed to have a costant dz and dx.
###
    
    nz = size(grid)[1]
    nx = size(grid)[2]
    
    dz = (zmax - zmin) / (nz-1)
    dx = (xmax - xmin) / (nx-1)
    
    if z>=zmax || z<=zmin || x<=xmin || x>=xmax
        #println(" Point outside interpolation region!! ")
        return 0.0
    else
        
        z_ix = convert(Int, ceil( (z-zmin)/dz )) 
        x_ix = convert(Int, ceil( (x-xmin)/dx )) 
        
        if z_ix+1 > nz || x_ix+1 > nx
            return 0.0
        end
        
        z1 = zmin + (z_ix-1)*dz
        z2 = zmin + (z_ix)*dz
        x1 = xmin + (x_ix-1)*dx
        x2 = xmin + (x_ix)*dx
        
        f11 = grid[z_ix, x_ix]
        f12 = grid[z_ix, x_ix+1]
        f21 = grid[z_ix+1, x_ix]
        f22 = grid[z_ix+1, x_ix+1]
        
        temp = f11*(z2-z)*(x2-x) + f21*(z-z1)*(x2-x) + f12*(z2-z)*(x-x1) + f22*(z-z1)*(x-x1) 
        
        return temp/(z2-z1)/(x2-x1)
        
    end
end





function find_nearest_index(array, x)
    for i =1:length(array)-1
        if array[i] <= x && x < array[i+1]
             return i
        end
    end
    println("No index found...")
end


function interp_will2(z::Float64, x::Float64, grid, zvec, xvec)
    
    zmin = zvec[1]
    zmax = zvec[length(zvec)]
    xmin = xvec[1]
    xmax = xvec[length(xvec)]
    
    #dz = (zmax - zmin) / (nz-1)
    #dx = (xmax - xmin) / (nx-1)
    
    if z>=zmax || z<=zmin || x<=xmin || x>=xmax
        #println(" Point outside interpolation region!! ")
        return 0.0
    else
        z_ix = find_nearest_index(zvec, z)
        x_ix = find_nearest_index(xvec, x)
        
        z1 = zvec[z_ix]
        z2 = zvec[z_ix+1]
        x1 = xvec[x_ix]
        x2 = xvec[x_ix+1]
        
        f11 = grid[z_ix, x_ix]
        f12 = grid[z_ix, x_ix+1]
        f21 = grid[z_ix+1, x_ix]
        f22 = grid[z_ix+1, x_ix+1]
        
        temp = f11*(z2-z)*(x2-x) + f21*(z-z1)*(x2-x) + f12*(z2-z)*(x-x1) + f22*(z-z1)*(x-x1) 
        
        return temp/(z2-z1)/(x2-x1)
        
    end
end