mec2 = 510998.94999999995

function sinc2(x)
    return sinc(x / pi)
end

function cosc2(x)
    if x == 0
        return -0.5
    else
        return (cos(x) - 1) / x^2
    end
end

function track_a_bend(b, p0c; L=0, theta=0, g_err=0)
    """
    Tracks a 6-D beam through a bending magnet.
    See chapter 23.6 of the Bmad manual.
    
    Input:
        b: initial 6D bmad beam coord
        p0c: reference momentum in eV/c
        L: length to track in m
        theta: bending angle 
        g_err: error in g = theta/L
    Ouptut:
        d: final 6D bmad beam coord
    """
    x = b[1]
    px = b[2]
    y = b[3]
    py = b[4]
    z = b[5]
    pz = b[6]

    px_norm = sqrt((1 + pz)^2 - py^2)  # For simplicity

    phi1 = asin(px / px_norm)

    g = theta / L
    g_tot = g + g_err
    gp = g_tot / px_norm

    alpha = (2 * (1 + g * x) * sin(theta + phi1) * L * sinc2(theta)
        - gp * ((1 + g * x) * L * sinc2(theta))^2)

    x2_t1 = x * cos(theta) + L^2 * g * cosc2(theta)
    x2_t2 = sqrt((cos(theta + phi1)^2) + gp * alpha)
    x2_t3 = cos(theta + phi1)
    
    if abs(theta + phi1) < (pi/2)
        x2 = x2_t1 + alpha / (x2_t2 + x2_t3)
    else
        x2 = x2_t1 + (x2_t2 - x2_t3) / gp
    end
    
    #println("x2:", x2)
    
    Lcu = x2 - L^2 * g * cosc2(theta) - x * cos(theta)
    Lcv = -L * sinc2(theta) - x * sin(theta)

    theta_p = 2 * (theta + phi1 - pi / 2 - atan(Lcv, Lcu))

    Lc = sqrt(Lcu^2 + Lcv^2)
    Lp = Lc / sinc2(theta_p / 2)

    P = p0c * (1 + pz)  # in eV
    E = sqrt(P^2 + mec2^2)  # in eV
    E0 = sqrt(p0c^2 + mec2^2)  # in eV
    beta = P / E
    beta0 = p0c / E0

    xf = x2
    pxf = px_norm * sin(theta + phi1 - theta_p)
    yf = y + py * Lp / px_norm
    pyf = py
    zf = z + (beta * L / beta0) - ((1 + pz) * Lp / px_norm)
    pzf = pz

    return Array([xf, pxf, yf, pyf, zf, pzf])
    
end




function track_a_drift(b, p0c; L=0)
    """
    Tracks a 6-D beam through a drift.
    See chapter 23.7 of the Bmad manual.
    
    Input:
        b: initial 6D bmad beam coord
        p0c: reference momentum in eV/c
        L: length to track in m
    Ouptut:
        d: final 6D bmad beam coord
    """
    x = b[1]
    px = b[2]
    y = b[3]
    py = b[4]
    z = b[5]
    pz = b[6]

    pl = sqrt(1 - (px^2 + py^2) / (1 + pz^2))  # unitless

    P = p0c * (1 + pz)  # in eV
    E = sqrt(P^2 + mec2^2)  # in eV
    E0 = sqrt(p0c^2 + mec2^2)  # in eV
    beta = P / E
    beta0 = p0c / E0

    xf = x + L * px / (1 + pz) / pl
    pxf = px
    yf = y + L * py / (1 + pz) / pl
    pyf = py
    zf = z + (beta / beta0 - 1 / pl) * L
    pzf = pz

    return Array([xf, pxf, yf, pyf, zf, pzf])
    
end