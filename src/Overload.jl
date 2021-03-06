# Julia internal methods overload.

import Base: ==
==(pb1::ParamBox, pb2::ParamBox) = (pb1[] == pb2[] && 
                                    pb1.dataName == pb2.dataName && 
                                    pb1.canDiff[] == pb2.canDiff[] && 
                                    pb1.index[] == pb2.index[] && 
                                    typeof(pb1.map) === typeof(pb2.map))

==(cp1::SpatialPoint, cp2::SpatialPoint) = (cp1.param == cp2.param)
==(t1::LTuple, t2::LTuple) = (t1.tuple == t2.tuple)
==(::LTuple{D, 0}, ::LTuple{D, 0}) where {D} = true
==(::LTuple{1, L1}, ::LTuple{1, L2}) where {L1, L2} = (L1 == L2)
==(::LTuple{1, 0}, ::LTuple{1, 0}) = true


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

getSPNDstring(t::Type{P1D{T, Lx}}) where {T, Lx} = 
(string(t), "P1D{$T, $(Lx)}")

getSPNDstring(t::Type{P2D{T, Lx, Ly}}) where {T, Lx, Ly} = 
(string(t), "P2D{$T, $(Lx), $(Ly)}")

getSPNDstring(t::Type{P3D{T, Lx, Ly, Lz}}) where {T, Lx, Ly, Lz} = 
(string(t), "P3D{$T, $(Lx), $(Ly), $(Lz)}")

function typeStrOf(sp::Type{SpatialPoint{T, D, PDT}}) where {T, D, PDT}
    spTstrO = sp |> string
    pbsTstrO, pbsTstrN = PDT |> getSPNDstring
    replace(spTstrO, pbsTstrO=>pbsTstrN, count=1)
end

typeStrOf(::T) where {T<:SpatialPoint} = typeStrOf(T)

function typeStrOf(bT::Type{<:FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, PDT}}) where 
                  {PDT}
    bTstrO = bT |> string
    pbsTstrO, pbsTstrN = PDT |> getSPNDstring
    replace(bTstrO, pbsTstrO=>pbsTstrN, count=1)
end

typeStrOf(::T) where {T<:FloatingGTBasisFuncs} = typeStrOf(T)

function typeStrOf(gbT::Type{<:GridBox{<:Any, <:Any, <:Any, SPT}}) where {SPT}
    gbTstrO = gbT |> string
    pbsTstrN = SPT |> typeStrOf
    replace(gbTstrO, string(SPT)=>pbsTstrN, count=1)
end

typeStrOf(::T) where {T<:GridBox} = typeStrOf(T)

typeStrOf(bfmT::Type{<:BasisFuncMix{<:Any, <:Any, <:Any, BFT}}) where {BFT} = 
replace(string(bfmT), string(BFT)=>typeStrOf(BFT), count=1)

typeStrOf(::T) where {T<:BasisFuncMix} = typeStrOf(T)

function getFieldNameStr(::T) where {T} 
    fields = fieldnames(T)
    str = fields |> string
    length(fields) == 1 && (str = str[1:end-2]*")")
    replace(str, ':'=>"")
end

function show(io::IO, sp::SpatialPoint)
    pbs = sp.param
    print(io, typeStrOf(sp), getFieldNameStr(sp))
    print(io, [i() for i in pbs])
    for pb in pbs
        print(io, "[")
        printstyled(io, ifelse(isDiffParam(pb), "𝛛", "∂"), color=diffColorSym(pb))
        print(io, "]")
    end
end

function show(io::IO, gf::GaussFunc)
    str = getFieldNameStr(gf)
    str = replace(str, "xpn"=>"xpn()=$(round(gf.xpn(), sigdigits=nSigShown))")
    str = replace(str, "con"=>"con()=$(round(gf.con(), sigdigits=nSigShown))")
    print(io, typeof(gf), str)
end

function show(io::IO, bf::BasisFunc)
    print(io, typeStrOf(bf))
    print(io, getFieldNameStr(bf), "[")
    printstyled(io, bf.l[1]|>LtoStr, color=:cyan)
    cen = round.([i() for i in bf.center], sigdigits=nSigShown)
    print(io, "]", cen)
end

function show(io::IO, bfs::BasisFuncs{<:Any, <:Any, 𝑙, <:Any, <:Any, ON}) where {𝑙, ON}
    SON = SubshellXYZsizes[𝑙+1]
    if ON == 1
        xyz1 = bfs.l[1] |> LtoStr
        xyz2 = ""
    else
        xyz1 = "$(bfs.l |> length)"
        xyz2 = "/$(SON)"
    end
    print(io, typeStrOf(bfs))
    print(io, getFieldNameStr(bfs), "[")
    printstyled(io, xyz1, color=:cyan)
    print(io, xyz2)
    cen = round.([i() for i in bfs.center], sigdigits=nSigShown)
    print(io, "]", cen)
end

show(io::IO, bfm::BasisFuncMix) = print(io, typeStrOf(bfm), getFieldNameStr(bfm))

show(io::IO, ::T) where {T<:EmptyBasisFunc} = print(io, T)

show(io::IO, gtb::GTBasis) = print(io, typeof(gtb), getFieldNameStr(gtb))

show(io::IO, box::GridBox) = print(io, typeStrOf(box), getFieldNameStr(box))

function show(io::IO, config::SCFconfig)
    print(io, typeof(config))
    str = getFieldNameStr(config)
    str = replace(str, "method,"=>"method=$(config.method),")
    str = replace(str, "interval"=>"interval=$(config.interval)")
    print(io, str)
end

function show(io::IO, vars::HFtempVars)
    print(io, typeof(vars))
    str = getFieldNameStr(vars)
    Etot0 = round(vars.shared.Etots[1], sigdigits=nSigShown)
    EtotL = round(vars.shared.Etots[end], sigdigits=nSigShown)
    str = replace(str, "shared"=>"shared.Etots=[$(Etot0), … , $(EtotL)]")
    print(io, str)
end

function show(io::IO, vars::HFfinalVars)
    print(io, typeof(vars))
    str = getFieldNameStr(vars)
    Ehf = round(vars.Ehf, sigdigits=nSigShown)
    str = replace(str, "Ehf"=>"Ehf=$(Ehf)")
    print(io, str)
end

show(io::IO, matter::MatterByHF) = print(io, typeof(matter), getFieldNameStr(matter))


import Base: +
+(bfm1::CGTBasisFuncs1O{T, D}, bfm2::CGTBasisFuncs1O{T, D}) where {T, D} = add(bfm1, bfm2)


import Base: *
*(bfm1::GaussFunc, bfm2::GaussFunc) = mul(bfm1, bfm2)

*(bfm::GaussFunc, coeff::Real) = mul(bfm, coeff)

*(coeff::Real, bfm::GaussFunc) = mul(coeff, bfm)

*(bfm1::CGTBasisFuncs1O{T, D}, bfm2::CGTBasisFuncs1O{T, D}) where {T, D} = mul(bfm1, bfm2)

*(bfm::CGTBasisFuncs1O, coeff::Real) = mul(bfm, coeff)

*(coeff::Real, bfm::CGTBasisFuncs1O) = mul(coeff, bfm)


## Iteration Interface
import Base: iterate, size, length, eltype
iterate(pb::ParamBox) = (pb.data[], nothing)
iterate(::ParamBox, _) = nothing
size(::ParamBox) = ()
length(::ParamBox) = 1
eltype(::ParamBox{T}) where {T} = T
size(pb::ParamBox, d::Integer) = size(pb.data[], d)

iterate(sp::SpatialPoint) = iterate(sp.param)
iterate(sp::SpatialPoint, state) = iterate(sp.param, state)
size(::SpatialPoint{<:Any, D}) where {D} = (D,)
length(::SpatialPoint{<:Any, D}) where {D} = D
eltype(::SpatialPoint{T}) where {T} = ParamBox{T}
function size(::SpatialPoint{<:Any, D}, d::Integer) where {D}
    if d > 0
        ifelse(d==1, D, 1)
    else
        throw(BoundsError())
    end
end

iterate(gf::GaussFunc) = (gf, nothing)
iterate(::GaussFunc, _) = nothing
size(::GaussFunc) = ()
length(::GaussFunc) = 1
size(::GaussFunc, d::Integer) = (d > 0) ? 1 : throw(BoundsError())

iterate(bf::CGTBasisFuncs1O) = (bf, nothing)
iterate(::CGTBasisFuncs1O, _) = nothing
size(::CGTBasisFuncs1O) = ()
length(::CGTBasisFuncs1O) = 1

function iterate(bfs::BasisFuncs)
    item, state = iterate(bfs.l)
    (BasisFunc(bfs.center, bfs.gauss, item, bfs.normalizeGTO), state)
end
function iterate(bfs::BasisFuncs, state)
    iter = iterate(bfs.l, state)
    iter !== nothing ? (BasisFunc(bfs.center, bfs.gauss, iter[1], bfs.normalizeGTO), 
                        iter[2]) : nothing
end

size(::CGTBasisFuncsON{ON}) where {ON} = (ON,)
length(::CGTBasisFuncsON{ON}) where {ON} = ON
eltype(::BasisFuncs{T, D, 𝑙, GN, PT}) where {T, D, 𝑙, GN, PT} = BasisFunc{T, D, 𝑙, GN, PT}

function size(x::SpatialBasis, d::Integer)
    if d > 0
        ifelse(d==1, length(x), 1)
    else
        throw(BoundsError())
    end
end

## Indexing Interface
import Base: getindex, setindex!, firstindex, lastindex, eachindex, axes
getindex(pb::ParamBox) = pb.data[]
getindex(pb::ParamBox, ::Val{:first}) = getindex(pb)
getindex(pb::ParamBox, ::Val{:last}) = getindex(pb)
setindex!(pb::ParamBox, d) = begin pb.data[] = d end
firstindex(::ParamBox) = Val(:first)
lastindex(::ParamBox) = Val(:last)
axes(::ParamBox) = ()

getindex(container::ParameterizedContainer) = container.param

getindex(sp::SpatialPoint, args...) = getindex(sp.param, args...)
firstindex(sp::SpatialPoint) = firstindex(sp.param)
lastindex(sp::SpatialPoint) = lastindex(sp.param)
eachindex(sp::SpatialPoint) = eachindex(sp.param)
axes(sp::SpatialPoint) = axes(sp.param)

@inline function getindex(bf::CGTBasisFuncs1O, i::Int)
    @boundscheck ( i==1 || throw(BoundsError(bf, i)) )
    bf
end

getindex(bfs::BasisFuncs, is::AbstractVector{Int}) = 
BasisFuncs(bfs.center, bfs.gauss, bfs.l[is], bfs.normalizeGTO)
getindex(bfs::BasisFuncs, i::Int) = 
BasisFunc(bfs.center, bfs.gauss, bfs.l[i], bfs.normalizeGTO)
getindex(bfs::BasisFuncs{T, D, 𝑙, GN, PT, ON}, ::Colon) where {T, D, 𝑙, GN, PT, ON} = 
BasisFunc{T, D, 𝑙, GN, PT}[getindex(bfs, i) for i=1:ON]
firstindex(bfs::BasisFuncs) = 1
lastindex(::BFuncsON{ON}) where {ON} = ON
eachindex(bfs::BFuncsON) = Base.OneTo(lastindex(bfs))

getindex(xyz::LTuple) = xyz.tuple
getindex(xyz::LTuple, args...) = getindex(xyz.tuple, args...)
firstindex(xyz::LTuple) = firstindex(xyz.tuple)
lastindex(xyz::LTuple) = lastindex(xyz.tuple)
eachindex(xyz::LTuple) = eachindex(xyz.tuple)
axes(xyz::LTuple) = axes(xyz.tuple)


## Broadcasting Interface
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
end


"""

    flatten(bs::AbstractVector{<:GTBasisFuncs{T, D}}) where {T, D} -> 
    AbstractVector{<:GTBasisFuncs{T, D, 1}}

    flatten(bs::Tuple{Vararg{GTBasisFuncs{T, D}}}) where {T, D} -> 
    Tuple{Vararg{GTBasisFuncs{T, D, 1}}}

Flatten a collection of `GTBasisFuncs` by decomposing every `GTBasisFuncs{T, D, ON}` 
where `ON > 1` into multiple `GTBasisFuncs{T, D, 1}`.
"""
flatten(bs::AbstractVector{<:GTBasisFuncs{T, D}}) where {T, D} = 
reshape(hcat(decomposeCore.(Val(false), bs)...), :)

flatten(bs::Tuple{Vararg{GTBasisFuncs{T, D}}}) where {T, D} = 
hcat(decomposeCore.(Val(false), bs)...) |> Tuple

flatten(bs::Union{AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                  Tuple{Vararg{FGTBasisFuncs1O{T, D}}}}) where {T, D} = 
itself(bs)


# The overload of following functions (or methods for specific types) are defined in 
# separate files to ensure precompilation capability.

## Methods for type __ : 
### LTuple

## Function __ : 
### getTypeParams