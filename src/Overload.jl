using Symbolics

# Julia internal methods overload.
import Base: ==
==(pb1::ParamBox, pb2::ParamBox) = (pb1[] == pb2[] && 
                                    pb1.canDiff[] == pb2.canDiff[] && 
                                    pb1.index == pb2.index && 
                                    pb1.map[] == pb2.map[])


import Base: show
function show(io::IO, pb::ParamBox)
    v = typeof(pb).parameters[1]
    if pb.index === nothing 
        i = Symbolics.variable(v)
    else
        i = Symbolics.variable(v, pb.index)
    end
    c = pb.canDiff[] ? :green : :light_black
    if typeof(pb.map[]) == typeof(itself)
        j = ""
    else
        j = " -> $(Symbol(pb.map[]))($i)"
    end
    print(io, typeof(pb))
    print(io, "($(pb.data[]))[$i$j]")
    printstyled(io, "[∂]", color=c)
end

function show(io::IO, gf::GaussFunc)
    print(io, "GaussFunc(xpn=")
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
    print(io, "][$(bf.center[1]()), $(bf.center[2]()), $(bf.center[3]())]")
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
    print(io, "][$(bf.center[1]()), $(bf.center[2]()), $(bf.center[3]())]")
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
    print(io, "(shared.Etot=[$(vars.shared.Etots[1]), … , $(vars.shared.Etots[end])], shared.Dtots, ")
    print(io, "Cs, Es, Ds, Fs)")
end

function show(io::IO, vars::HFfinalVars)
    print(io, typeof(vars))
    print(io, "(E0HF=$(vars.E0HF), C, F, D, Emo, occu, temp, isConverged)")
end


import Base: getindex, setindex!
getindex(m::ParamBox) = m.data[]


setindex!(a::ParamBox, b) = begin a.data[] = b end


# Quiqboc methods overload.
## Method overload of `hasBoolRelation` from Tools.jl.
function hasBoolRelation(boolFunc::Function, pb1::ParamBox, pb2::ParamBox; 
                         ignoreFunction=false, ignoreContainerType=false)
    if ignoreContainerType
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