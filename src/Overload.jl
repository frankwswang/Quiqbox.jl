# Julia internal methods overload.

import Base: ==
==(pb1::ParamBox, pb2::ParamBox) = (pb1[] == pb2[] && 
                                    pb1.dataName == pb2.dataName && 
                                    pb1.canDiff[] == pb2.canDiff[] && 
                                    pb1.index[] == pb2.index[] && 
                                    typeof(pb1.map) === typeof(pb2.map))

==(cp1::SpatialPoint, cp2::SpatialPoint) = (cp1.param == cp2.param)
==(t1::LTuple, t2::LTuple) = (t1.tuple == t2.tuple)
==(t1::LTuple{D, 0}, t2::LTuple{D, 0}) where {D} = true
==(t1::LTuple{1, L}, t2::LTuple{1, L}) where {L} = true


diffColorSym(pb::ParamBox) = ifelse(isDiffParam(pb), :green, :light_black)

import Base: show
const nSigShown = 10
function show(io::IO, pb::ParamBox)
    v = pb.data[]
    print(io, typeof(pb))
    print(io, "(", v isa Integer ? v : round(v, sigdigits=nSigShown), ")")
    print(io, "[")
    printstyled(io, "∂", color=diffColorSym(pb))
    print(io, "][")
    printstyled(io, "$(getVar(pb, true))", color=:cyan)
    print(io, "]")
end

function show(io::IO, sp::SpatialPoint)
    pbs = sp.param
    print(io, typeof(sp), "(point.param)")
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
    printstyled(io, bf.l[1]|>LtoStr, color=:cyan)
    cen = round.([i() for i in bf.center], sigdigits=nSigShown)
    print(io, "]", cen)
end

function show(io::IO, bf::BasisFuncs{<:Any, <:Any, 𝑙, <:Any, <:Any, ON}) where {𝑙, ON}
    SON = SubshellXYZsizes[𝑙+1]
    if ON == 1
        xyz1 = bf.l[1] |> LtoStr
        xyz2 = ""
    else
        xyz1 = "$(bf.l |> length)"
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
    print(io, "(basis, S, Te, eeI)")
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
              "C, F, D, Eo, occu, temp, isConverged, basis)")
end


import Base: +
+(bfm1::CGTBasisFuncs1O{T, D}, bfm2::CGTBasisFuncs1O{T, D}) where {T, D} = add(bfm1, bfm2)


import Base: *
*(bfm1::GaussFunc, bfm2::GaussFunc) = mul(bfm1, bfm2)

*(bfm::GaussFunc, coeff::Real) = mul(bfm, coeff)

*(coeff::Real, bfm::GaussFunc) = mul(coeff, bfm)

*(bfm1::CGTBasisFuncs1O{T, D}, bfm2::CGTBasisFuncs1O{T, D}) where {T, D} = mul(bfm1, bfm2)

*(bfm::CGTBasisFuncs1O, coeff::Real) = mul(bfm, coeff)

*(coeff::Real, bfm::CGTBasisFuncs1O) = mul(coeff, bfm)


# Iteration Interface
import Base: iterate, size, length, ndims
iterate(pb::ParamBox) = (pb.data[], nothing)
iterate(::ParamBox, _) = nothing
size(::ParamBox) = ()
length(::ParamBox) = 1
ndims(::ParamBox) = 0
size(::ParamBox, d::Integer) = (d == 1) ? 1 : throw(BoundsError())

iterate(gf::GaussFunc) = (gf, nothing)
iterate(::GaussFunc, _) = nothing
size(::GaussFunc) = ()
length(::GaussFunc) = 1
size(::GaussFunc, d::Integer) = (d == 1) ? 1 : throw(BoundsError())

iterate(sp::SpatialPoint) = iterate(sp.param)
iterate(sp::SpatialPoint, state) = iterate(sp.param, state)
size(::SpatialPoint{<:Any, D}) where {D} = (D,)
length(::SpatialPoint{<:Any, D}) where {D} = D

iterate(bf::BasisFunc) = (bf, nothing)
iterate(::BasisFunc, _) = nothing
size(::BasisFunc) = ()
length(::BasisFunc) = 1

iterate(bfm::BasisFuncMix) = (bfm, nothing)
iterate(::BasisFuncMix, _) = nothing
size(::BasisFuncMix) = ()
length(::BasisFuncMix) = 1

iterate(bfZero::EmptyBasisFunc) = (bfZero, nothing)

function iterate(bfs::CompositeGTBasisFuncs)
    item, state = iterate(bfs.l)
    (BasisFunc(bfs.center, bfs.gauss, item, bfs.normalizeGTO), state)
end
function iterate(bfs::CompositeGTBasisFuncs, state)
    iter = iterate(bfs.l, state)
    iter !== nothing ? (BasisFunc(bfs.center, bfs.gauss, iter[1], bfs.normalizeGTO), 
                        iter[2]) : nothing
end
size(::CGTBasisFuncsON{ON}) where {ON} = (ON,)
length(::CGTBasisFuncsON{ON}) where {ON} = ON

size(x::SpatialOrbital, d::Integer) = (d == 1) ? length(x) : throw(BoundsError())


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

getindex(bfm::BasisFuncMix) = (collect ∘ flatten)( getproperty.(bfm.BasisFunc, :gauss) )
getindex(bfm::BasisFuncMix, ::Val{:first}) = getindex(bfm)
getindex(bfm::BasisFuncMix, ::Val{:last}) = getindex(bfm)
firstindex(::BasisFuncMix) = Val(:first)
lastindex(::BasisFuncMix) = Val(:last)

getindex(bfs::BasisFuncs, is::AbstractVector{Int}) = 
BasisFuncs(bfs.center, bfs.gauss, bfs.l[is], bfs.normalizeGTO)
getindex(bfs::BasisFuncs, i::Int) = 
BasisFunc(bfs.center, bfs.gauss, bfs.l[i], bfs.normalizeGTO)
getindex(bfs::BFuncsON{ON}, ::Colon) where {ON} = [getindex(bfs, i) for i=1:ON]
firstindex(bfs::BasisFuncs) = 1
lastindex(::BFuncsON{ON}) where {ON} = ON
eachindex(bfs::BFuncsON) = Base.OneTo(lastindex(bfs))
getindex(bfs::BasisFuncs) = getproperty.(bfs[:], :gauss) |> flatten

getindex(xyz::LTuple, args...) = getindex(xyz.tuple, args...)
getindex(xyz::LTuple) = xyz.tuple
firstindex(xyz::LTuple) = firstindex(xyz.tuple)
lastindex(xyz::LTuple) = lastindex(xyz.tuple)
eachindex(xyz::LTuple) = eachindex(xyz.tuple)
axes(xyz::LTuple) = axes(xyz.tuple)


# Broadcasting Interface
import Base: broadcastable
broadcastable(pb::ParamBox) = Ref(pb)
broadcastable(gf::GaussFunc) = Ref(gf)
broadcastable(bf::CGTBasisFuncs1O) = Ref(bf)
broadcastable(bfs::BasisFuncs) = getindex(bfs, :)
Base.broadcastable(sp::SpatialPoint) = Base.broadcastable(sp.param)


# Quiqbox methods overload.
## Method overload of `hasBoolRelation` from Tools.jl.
function hasBoolRelation(boolFunc::F, 
                         pb1::ParamBox{<:Any, V1, F1}, pb2::ParamBox{<:Any, V2, F2}; 
                         ignoreFunction::Bool=false, ignoreContainer::Bool=false, 
                         kws...) where {F<:Function, V1, V2, F1, F2}
    # ifelse(ignoreContainer, 
    #     boolFunc(pb1(), pb2()), 

    #     ifelse(V1 == V2, 
    #         ifelse((ignoreFunction || F1 == F2 == FI), 
    #             boolFunc(pb1.data, pb2.data), 

    #             ( boolFunc(pb1.canDiff[], pb2.canDiff[]) && 
    #               boolFunc(pb1.map, pb2.map) && 
    #               boolFunc(pb1.data, pb2.data) )
    #         ), 

    #         false
    #     )
    # )

    if ignoreContainer
        boolFunc(pb1(), pb2())
    else
        ifelse(V1 == V2, 
            ifelse((ignoreFunction || F1 == F2 == FI), 
                boolFunc(pb1.data, pb2.data), 

                ( boolFunc(pb1.canDiff[], pb2.canDiff[]) && 
                  boolFunc(pb1.map, pb2.map) && 
                  boolFunc(pb1.data, pb2.data) )
            ), 

            false
        )
    end

    # if ignoreContainer
    #     boolFunc(pb1(), pb2())
    # elseif V1 == V2
    #     ifelse( (ignoreFunction || F1 == F2 == FI), 
    #         boolFunc(pb1.data, pb2.data), 

    #         ( boolFunc(pb1.canDiff[], pb2.canDiff[]) && 
    #           boolFunc(pb1.map, pb2.map) && 
    #           boolFunc(pb1.data, pb2.data) )
    #     )
    # else
    #     false
    # end
end


"""

    flatten(b::AbstractVector{<:GTBasisFuncs{T, D}}) where {T, D} -> 
    AbstractVector{<:CompositeGTBasisFuncs{T, D, <:Any, 1}}

    flatten(b::Tuple{Vararg{GTBasisFuncs{T, D}}}) where {T, D} -> 
    Tuple{Vararg{CompositeGTBasisFuncs{T, D, <:Any, 1}}}

Flatten a collection of `CompositeGTBasisFuncs` by decomposing every 
`CompositeGTBasisFuncs{T, D, <:Any, ON}` where `ON > 1` into multiple 
`CompositeGTBasisFuncs{T, D, <:Any, 1}`.
"""
flatten(bs::AbstractVector{<:GTBasisFuncs{T, D}}) where {T, D} = 
reshape(hcat(decomposeCore.(Val(false), bs)...), :)

flatten(bs::Tuple{Vararg{GTBasisFuncs{T, D}}}) where {T, D} = 
hcat(decomposeCore.(Val(false), bs)...) |> Tuple

flatten(bs::Union{AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                  Tuple{Vararg{FGTBasisFuncs1O{T, D}}}}) where {T, D} = 
itself(bs)