function emit(bb)
    return sqrt(cov(bb[1,:], bb[1,:])*cov(bb[2,:], bb[2,:]) - cov(bb[1,:], bb[2,:])^2)
end

function emit_dispersion_free(bb)
    delta2 = cov(bb[6,:], bb[6,:])
    xd = cov(bb[1,:], bb[6,:])
    pd = cov(bb[2,:], bb[6,:])
    
    eb = cov(bb[1,:], bb[1,:]) - xd^2 / delta2
    eg = cov(bb[2,:], bb[2,:]) - pd^2 / delta2
    ea = -cov(bb[1,:], bb[2,:]) + xd * pd / delta2
    
    emit = sqrt(eb*eg - ea^2)
    return emit 
end