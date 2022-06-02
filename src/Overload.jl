# Julia internal methods overload.

import Base: ==
==(pb1::ParamBox, pb2::ParamBox) = (pb1[] == pb2[] && 
                                    pb1.dataName == pb2.dataName && 
                                    pb1.canDiff[] == pb2.canDiff[] && 
                                    pb1.index[] == pb2.index[] && 
                                    typeof(pb1.map) === typeof(pb2.map))

==(cp1::SpatialPoint, cp2::SpatialPoint) = (cp1.param == cp2.param)
==(t1::XYZTuple, t2::XYZTuple) = (t1.tuple == t2.tuple)


diffColorSym(pb::ParamBox) = isDiffParam(pb) ? :green : :light_black

import Base: show
const nSigShown = 10
function show(io::IO, pb::ParamBox)
    print(io, typeof(pb))
    print(io, "(", round(pb.data[], sigdigits=nSigShown), ")")
    print(io, "[")
    printstyled(io, "∂", color=diffColorSym(pb))
    print(io, "][")
    printstyled(io, "$(pb|>getVar)", color=:cyan)
    print(io, "]")
end

function show(io::IO, sp::SpatialPoint)
    pbs = sp.param
    print(io, typeof(sp)|>supertype, "(param)")
    print(io, [i() for i in pbs])
    for pb in pbs
        print(io, "[")
        printstyled(io, "∂", color=diffColorSym(pb))
        print(io, "]")
    end
end

function show(io::IO, gf::GaussFunc)
    print(io, typeof(gf), "(xpn=")
    show(io, gf.xpn)
    print(io, ", con=")
    show(io, gf.con)
    print(io, ")")
end

function show(io::IO, bf::BasisFunc)
    print(io, typeof(bf))
    print(io, "(center, gauss)[")
    printstyled(io, bf.ijk[1]|>ijkToStr, color=:cyan)
    cen = round.([i() for i in bf.center], sigdigits=nSigShown)
    print(io, "]", cen)
end

function show(io::IO, bf::BasisFuncs{𝑙, <:Any, ON}) where {𝑙, ON}
    SON = SubshellXYZsizes[𝑙+1]
    if ON == 1
        xyz1 = bf.ijk[1] |> ijkToStr
        xyz2 = ""
    else
        xyz1 = "$(bf.ijk |> length)"
        xyz2 = "/$(SON)"
    end
    print(io, typeof(bf))
    print(io, "(center, gauss)[")
    printstyled(io, xyz1, color=:cyan)
    print(io, xyz2)
    cen = round.([i() for i in bf.center], sigdigits=nSigShown)
    print(io, "]", cen)
end

function show(io::IO, bfm::BasisFuncMix)
    print(io, typeof(bfm))
    print(io, "(BasisFunc, param)")
end

function show(io::IO, ::EmptyBasisFunc)
    print(io, EmptyBasisFunc)
end

function show(io::IO, gtb::GTBasis)
    print(io, typeof(gtb))
    print(io, "(basis, S, Te, eeI, getVne, getHcore)")
end

function show(io::IO, box::GridBox)
    print(io, typeof(box))
    print(io, "(num, len, coord)")
end

function show(io::IO, config::SCFconfig)
    print(io, typeof(config))
    print(io, "(interval=", config.interval, ",", 
              " oscillateThreshold=", config.oscillateThreshold, ",", 
              " method, methodConfig)", config.method|>collect)
end

function show(io::IO, vars::HFtempVars)
    print(io, typeof(vars))
    print(io, "(shared.Etots=[", round(vars.shared.Etots[1], sigdigits=nSigShown),", … , ", 
                                 round(vars.shared.Etots[end], sigdigits=nSigShown), "], "*
              "shared.Dtots, N, Cs, Fs, Ds, Es)")
end

function show(io::IO, vars::HFfinalVars)
    print(io, typeof(vars))
    print(io, "(Ehf=", round(vars.Ehf, sigdigits=nSigShown), ", Enn, N, nuc, nucCoords, " * 
              "C, F, D, Emo, occu, temp, isConverged)")
end


import Base: +
+(bfm1::CompositeGTBasisFuncs{<:Any, 1}, bfm2::CompositeGTBasisFuncs{<:Any, 1}) = 
add(bfm1, bfm2)


import Base: *
*(bfm1::GaussFunc, bfm2::GaussFunc) = mul(bfm1, bfm2)

*(bfm::GaussFunc, coeff::Real) = mul(bfm, coeff)

*(coeff::Real, bfm::GaussFunc) = mul(coeff, bfm)

*(bfm1::CompositeGTBasisFuncs{<:Any, 1}, bfm2::CompositeGTBasisFuncs{<:Any, 1}) = 
mul(bfm1, bfm2)

*(bfm::CompositeGTBasisFuncs{<:Any, 1}, coeff::Real) = 
mul(bfm, coeff)

*(coeff::Real, bfm::CompositeGTBasisFuncs{<:Any, 1}) = 
mul(coeff, bfm)


# Iteration Interface
import Base: iterate, size, length, ndims
iterate(pb::ParamBox) = (pb.data[], nothing)
iterate(::ParamBox, _) = nothing
size(::ParamBox) = ()
length(::ParamBox) = 1
ndims(::ParamBox) = 0
size(::ParamBox, d::Integer) = d == 1 ? 1 : throw(BoundsError())

iterate(gf::GaussFunc) = (gf, nothing)
iterate(::GaussFunc, _) = nothing
size(::GaussFunc) = ()
length(::GaussFunc) = 1

iterate(sp::SpatialPoint) = iterate(sp.param)
iterate(sp::SpatialPoint, state) = iterate(sp.param, state)
size(::SpatialPoint{D}) where {D} = (D,)
length(::SpatialPoint{D}) where {D} = D

iterate(bf::BasisFunc) = (bf, nothing)
iterate(::BasisFunc, _) = nothing
size(::BasisFunc) = ()
length(::BasisFunc) = 1

iterate(bfm::BasisFuncMix) = (bfm, nothing)
iterate(::BasisFuncMix, _) = nothing
size(::BasisFuncMix) = ()
length(::BasisFuncMix) = 1

iterate(bfZero::EmptyBasisFunc) = (bfZero, nothing)

function iterate(bfs::CompositeGTBasisFuncs{<:Any, N}) where {N}
    item, state = iterate(bfs.ijk)
    (BasisFunc(bfs.center, bfs.gauss, item, bfs.normalizeGTO), state)
end
function iterate(bfs::CompositeGTBasisFuncs{<:Any, N}, state) where {N}
    iter = iterate(bfs.ijk, state)
    iter !== nothing ? (BasisFunc(bfs.center, bfs.gauss, iter[1], bfs.normalizeGTO), 
                        iter[2]) : nothing
end
size(::CompositeGTBasisFuncs{<:Any, N}) where {N} = (N,)
length(::CompositeGTBasisFuncs{<:Any, N}) where {N} = N

size(x::SpatialOrbital, d::Integer) = d == 1 ? length(x) : throw(BoundsError())


# Indexing Interface
import Base: getindex, setindex!, firstindex, lastindex, eachindex, axes
getindex(pb::ParamBox) = pb.data[]
getindex(pb::ParamBox, ::Val{:first}) = getindex(pb)
getindex(pb::ParamBox, ::Val{:last}) = getindex(pb)
setindex!(pb::ParamBox, d) = begin pb.data[] = d end
firstindex(::ParamBox) = Val(:first)
lastindex(::ParamBox) = Val(:last)
axes(::ParamBox) = ()

getindex(sp::SpatialPoint, args...) = getindex(sp.param, args...)
getindex(sp::SpatialPoint) = sp.param
firstindex(sp::SpatialPoint) = firstindex(sp.param)
lastindex(sp::SpatialPoint) = lastindex(sp.param)
eachindex(sp::SpatialPoint) = eachindex(sp.param)
axes(sp::SpatialPoint) = axes(sp.param)
ndims(sp::SpatialPoint) = ndims(sp.param)

getindex(gf::GaussFunc) = gf.param |> collect
getindex(gf::GaussFunc, ::Val{:first}) = getindex(gf)
getindex(gf::GaussFunc, ::Val{:last}) = getindex(gf)
firstindex(::GaussFunc) = Val(:first)
lastindex(::GaussFunc) = Val(:last)

getindex(bf::BasisFunc) = bf.gauss |> collect
getindex(bf::BasisFunc, ::Val{:first}) = getindex(bf)
getindex(bf::BasisFunc, ::Val{:last}) = getindex(bf)
firstindex(::BasisFunc) = Val(:first)
lastindex(::BasisFunc) = Val(:last)

getindex(bfm::BasisFuncMix) = (collect ∘ flatten)( getfield.(bfm.BasisFunc, :gauss) )
getindex(bfm::BasisFuncMix, ::Val{:first}) = getindex(bfm)
getindex(bfm::BasisFuncMix, ::Val{:last}) = getindex(bfm)
firstindex(::BasisFuncMix) = Val(:first)
lastindex(::BasisFuncMix) = Val(:last)

getindex(bfs::BasisFuncs, i) = 
BasisFunc(bfs.center, bfs.gauss, bfs.ijk[i], bfs.normalizeGTO)
getindex(bfs::BasisFuncs{<:Any, <:Any, N}, ::Colon) where {N} = [getindex(bfs, i) for i=1:N]
firstindex(bfs::BasisFuncs) = 1
lastindex(::BasisFuncs{<:Any, <:Any, N}) where {N} = N
eachindex(bfs::BasisFuncs{<:Any, <:Any, N}) where {N} = Base.OneTo(lastindex(bfs))
getindex(bfs::BasisFuncs) = getfield.(bfs[:], :gauss) |> flatten

getindex(xyz::XYZTuple, args...) = getindex(xyz.tuple, args...)
getindex(xyz::XYZTuple) = xyz.tuple
firstindex(xyz::XYZTuple) = firstindex(xyz.tuple)
lastindex(xyz::XYZTuple) = lastindex(xyz.tuple)
eachindex(xyz::XYZTuple) = eachindex(xyz.tuple)
axes(xyz::XYZTuple) = axes(xyz.tuple)


# Broadcasting Interface
import Base: broadcastable
broadcastable(pb::ParamBox) = Ref(pb)
broadcastable(gf::GaussFunc) = Ref(gf)
broadcastable(bf::CompositeGTBasisFuncs{<:Any, 1}) = Ref(bf)
broadcastable(bfs::BasisFuncs) = getindex(bfs, :)
Base.broadcastable(sp::SpatialPoint) = Base.broadcastable(sp.param)


# Quiqbox methods overload.
## Method overload of `hasBoolRelation` from Tools.jl.
function hasBoolRelation(boolFunc::F, 
                         pb1::ParamBox{<:Any, <:Any, F1}, pb2::ParamBox{<:Any, <:Any, F2}; 
                         ignoreFunction::Bool=false, ignoreContainer::Bool=false,
                         kws...) where {F<:Function, F1, F2}
    if ignoreContainer
        boolFunc(pb1(), pb2())
    elseif ignoreFunction || F1 == F2 == FLevel(itself)
        boolFunc(pb1.data, pb2.data)
    else
        boolFunc(pb1.canDiff[], pb2.canDiff[]) && boolFunc(pb1.map, pb2.map) && 
        boolFunc(pb1.data, pb2.data)
    end
end