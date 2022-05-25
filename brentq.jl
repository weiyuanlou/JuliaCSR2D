# Root finding

const _ECONVERGED = Symbol("converged")
const _ECONVERR   = Symbol("err")

const _iter = 100
const _xtol = 2e-12       #2e-12
const _rtol = 4eps()

"""Conditional checks for intervals in methods involving bisection"""
function _bisect_interval(a::Real, b::Real, fa::Real, fb::Real)

    if fa*fb > 0 
        error("f(a) and f(b) must have different signs")
    end
    root = 0.0
    status = _ECONVERR

    # Root found at either end of [a,b]
    if fa == 0
        root = a
        status = _ECONVERGED
    elseif fb == 0
        root = b
        status = _ECONVERGED
    end

    root, status
end

#typeof(_ECONVERGED)

function brentq(f::Function, a::Float64, b::Float64)
    xtol=_xtol
    rtol=_rtol
    maxiter=_iter
    disp=true
    """
    Modified from: 
    https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/optimize/root_finding.py#L6
    with fixed arguments. 
    
    Find a root of a function in a bracketing interval using Brent's method
    adapted from Scipy's brentq.
    Uses the classic Brent's method to find a zero of the function `f` on
    the sign changing interval [a , b].
    `f` must be jitted via numba.
    Parameters
    ----------
    f : jitted and callable
        Python function returning a number.  `f` must be continuous.
    a : number
        One end of the bracketing interval [a,b].
    b : number
        The other end of the bracketing interval [a,b].
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    xtol : number, optional(default=2e-12)
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be nonnegative.
    rtol : number, optional(default=4*np.finfo(float).eps)
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
    maxiter : number, optional(default=100)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge.
    Returns
    -------
    results : namedtuple
    """
    ## if xtol <= 0
    ##     error("xtol is too small (<= 0)")
    ## elseif maxiter < 1
    ##     error("maxiter must be greater than 0")
    ## end
    
    # Convert to float
    xpre = float(a)
    xcur = float(b)

    fpre = f(xpre)
    fcur = f(xcur)
    funcalls = 2
    
    # CM added
    xblk = xpre
    fblk = fpre
    spre = scur = xcur - xpre

    root, status = _bisect_interval(xpre, xcur, fpre, fcur)
    
    # Check for sign error and early termination
    if status == _ECONVERGED
        itr = 0
    else
        # Perform Brent's method
        for itr in 0:maxiter
            
            if fpre * fcur < 0
                xblk = xpre
                fblk = fpre
                spre = scur = xcur - xpre
            end
            
            if abs(fblk) < abs(fcur)
                xpre = xcur
                xcur = xblk
                xblk = xpre

                fpre = fcur
                fcur = fblk
                fblk = fpre
            end

            delta = (xtol + rtol * abs(xcur)) / 2
            sbis = (xblk - xcur) / 2

            # Root found
            if fcur == 0 || (abs(sbis) < delta)
                status = _ECONVERGED
                root = xcur
                itr += 1
                break
            end

            if (abs(spre) > delta) && (abs(fcur) < abs(fpre))
                if xpre == xblk
                    # interpolate
                    stry = -fcur * (xcur - xpre) / (fcur - fpre)
                else
                    # extrapolate
                    dpre = (fpre - fcur) / (xpre - xcur)
                    dblk = (fblk - fcur) / (xblk - xcur)
                    stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre)) 
                end

                if (2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta))
                    # good short step
                    spre = scur
                    scur = stry
                else
                    # bisect
                    spre = sbis
                    scur = sbis
                end
            else
                # bisect
                spre = sbis
                scur = sbis
            end

            xpre = xcur
            fpre = fcur
            if abs(scur) > delta
                xcur += scur
            else
                
                xcur += sbis > 0 ? delta : -delta
            end
            
            fcur = f(xcur)
            funcalls += 1
        end
    end

    ## if disp && status == _ECONVERR
    ##     error("Failed to converge")
    ## end

    #return _results((root, funcalls, itr, status))
    return root
end


function find_root_Will(f::Function, x0::Float64, x1::Float64, n::Int)
    """
    Find an root of f(x) with x0 < x < x1 .
    The domain (x0, x1) is first sliced into n sub-domains.
    The function tries to find the first sub-domain with two endpoints xl and xr for which f(xl)*f(xr)<0.
    If found, the Brent's method is applied to find the root.
    If not found, the function returns -1.0.
    """
    
    root = -1.0
    
    dx = (x1-x0)/n
    
    sign0 = sign(f(x0))
    xtemp = x0
    for i in 1:n
        xtemp += dx
        sign1 = sign(f(xtemp))
        
        if sign0 != sign1
            
            root = brentq(f, xtemp-dx, xtemp)
            
            # Return the first root found
            return root   
        end
        sign0 = sign1
    end
    
    return root
end