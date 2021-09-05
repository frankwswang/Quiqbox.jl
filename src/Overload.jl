# Julia internal methods overload.
import Base: ==
==(pb1::ParamBox, pb2::ParamBox) = (pb1[] == pb2[] && 
                                    pb1.canDiff[] == pb2.canDiff[] && 
                                    pb1.index == pb2.index && 
                                    pb1.map[] == pb2.map[])


import Base: show
const nSigShown = 10
function show(io::IO, pb::ParamBox)
    v = typeof(pb).parameters[1]
    if pb.index === nothing 
        i = v
    else
        i = string(v)*numToSubs(pb.index)
    end
    c = pb.canDiff[] ? :green : :light_black
    if typeof(pb.map[]) == typeof(itself)
        j = ""
    else
        j = " -> $(Symbol(pb.map[]))($i)"
    end
    print(io, typeof(pb))
    print(io, "(", round(pb.data[], sigdigits=nSigShown), ")[", i, j, "]")
    printstyled(io, "[∂]", color=c)
end

function show(io::IO, gf::GaussFunc)
    print(io, typeof(gf), "(xpn=")
    show(io, gf.xpn)
    print(io, ", con=")
    show(io, gf.con)
    print(io, ")")
end

function show(io::IO, bf::BasisFunc)
    xyz1 = bf.ijk[1]
    xyz2 = ""
    print(io, typeof(bf))
    print(io, "(gauss, subshell, center)[")
    printstyled(io, xyz1, color=:cyan)
    print(io, xyz2)
    print(io, "][", round(bf.center[1](), sigdigits=nSigShown), ", ",
                    round(bf.center[2](), sigdigits=nSigShown), ", ",
                    round(bf.center[3](), sigdigits=nSigShown), "]")
end

function show(io::IO, bf::BasisFuncs)
    OON = typeof(bf).parameters[3]
    SON = SubshellDimList[bf.subshell]
    if OON == 1
        xyz1 = bf.ijk[1]
        xyz2 = ""
    else
        xyz1 = "$(bf.ijk |> length)"
        xyz2 = "/$(SON)"
    end
    print(io, typeof(bf))
    print(io, "(gauss, subshell, center)[")
    printstyled(io, xyz1, color=:cyan)
    print(io, xyz2)
    print(io, "][", round(bf.center[1](), sigdigits=nSigShown), ", ",
                    round(bf.center[2](), sigdigits=nSigShown), ", ",
                    round(bf.center[3](), sigdigits=nSigShown), "]")
end

function show(io::IO, bfm::BasisFuncMix)
    print(io, typeof(bfm))
    print(io, "(BasisFunc, param)")
end

function show(io::IO, gtb::GTBasis)
    print(io, typeof(gtb))
    print(io, "(basis, S, Te, eeI, getVne, getHcore)")
end

function show(io::IO, box::GridBox)
    print(io, typeof(box))
    print(io, "(num, len, coord)")
end

function show(io::IO, vars::HFtempVars)
    print(io, typeof(vars))
    print(io, "(shared.Etot=[", round(vars.shared.Etots[1], sigdigits=nSigShown),", … , ", 
                                round(vars.shared.Etots[end], sigdigits=nSigShown), "], "*
                            "shared.Dtots, Cs, Es, Ds, Fs)")
end

function show(io::IO, vars::HFfinalVars)
    print(io, typeof(vars))
    print(io, "(E0HF=", round(vars.E0HF, sigdigits=nSigShown), ", C, F, D, Emo, occu, temp"*
              ", isConverged)")
end


import Base: getindex, setindex!
getindex(m::ParamBox) = m.data[]


setindex!(a::ParamBox, b) = begin a.data[] = b end


# Quiqbox methods overload.
## Method overload of `hasBoolRelation` from Tools.jl.
## If `ignoreContainer = true`, then `ignoreFunction` is automatically set to be `true` 
## as the `map` function for `ParamBox` is considered as a type of container for the actual
## stored data.
function hasBoolRelation(boolFunc::F, pb1::ParamBox, pb2::ParamBox; 
                         ignoreFunction::Bool=false, 
                         ignoreContainer::Bool=false,
                         decomposeNumberCollection::Bool=false) where {F<:Function}
    if ignoreContainer
        boolFunc(pb1.data, pb2.data)
    elseif ignoreFunction
        boolFunc(pb1.data, pb2.data) && 
        boolFunc(pb1.canDiff[],pb2.canDiff[]) 
    else
        boolFunc(pb1.map[], pb2.map[]) && 
        boolFunc(pb1.data, pb2.data) && 
        boolFunc(pb1.canDiff[],pb2.canDiff[])
    end
end