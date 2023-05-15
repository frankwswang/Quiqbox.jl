# Julia internal methods overload.

import Base: ==
==(pb1::ParamBox, pb2::ParamBox) = (pb1.data == pb2.data && 
                                    isDiffParam(pb1) == isDiffParam(pb2) && 
                                    pb1.index[] == pb2.index[] && 
                                    pb1.map == pb2.map)

==(cp1::SpatialPoint, cp2::SpatialPoint) = (cp1.param == cp2.param)
==(t1::LTuple, t2::LTuple) = (t1.tuple == t2.tuple)
==(::LTuple{D, 0}, ::LTuple{D, 0}) where {D} = true
==(::LTuple{1, L1}, ::LTuple{1, L2}) where {L1, L2} = (L1 == L2)
==(::LTuple{1, 0}, ::LTuple{1, 0}) = true


function omitLastTypeParPrint(io::IO, str::String, nRepeat::Int=1)
    for _ in 1:nRepeat
        str = str[begin:prevind(str, findlast(',', str))]
    end
    print(io, str)
    nRepeat > 0 && print(io, ", …", "}")
end


function printFLcore(io::IO, ::Type{F}) where {F<:Function}
    FL = getFLevel(F)
    if F <: DI
        printstyled(io, FL, color=:light_black)
    else
        print(io, FL)
    end
end

printFLcore(io, ::Type{<:ParamBox{<:Any, <:Any, F}}) where {F<:Function} = 
printFLcore(io, F)


function printFL(io::IO, ::Type{T}) where {T}
    print(io, "{")
    printFLcore(io, T)
    print(io, "}")
end

function printFL(io::IO, Fs::Tuple{Vararg{Type}})
    print(io, "{")
    for (i, F) in enumerate(Fs)
        printFLcore(io, F)
        i < length(Fs) && print(io, ", ")
    end
    print(io, "}")
end

printFL(io::IO, ::Type{T}) where {T<:Tuple{Vararg{ParamBox}}} = printFL(io, fieldtypes(T))

printFL(io::IO, ::F) where {F<:Function} = printFL(io, F)

printFL(io, ::Type{<:FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, F}}) where {F} = 
printFL(io, F)


diffColorSym(@nospecialize(pb::ParamBox)) = ifelse(isDiffParam(pb), :green, :light_black)

printDiffSymCore(io::IO, bl::Bool, pb::ParamBox) = 
printstyled(io, ifelse(bl, "𝛛", "∂"), color=diffColorSym(pb))

function printDiffSym(io::IO, @nospecialize(pb::ParamBox))
    bl = isDiffParam(pb)
    print(io, "[")
    printDiffSymCore(io, bl, pb)
    print(io, "]")
    ifelse(bl, "⟦→⟧", "⟦=⟧")
end

function printDiffSym(io::IO, @nospecialize(pbs::Tuple{Vararg{ParamBox}}))
    print(io, "[")
    bl1 = true
    for pb in pbs
        bl2 = isDiffParam(pb)
        bl1 *= !bl2
        printDiffSymCore(io, bl2, pb)
    end
    print(io, "]")
    ifelse(bl1, "=", "→")
end


function printIndVarCore(io::IO, pb::ParamBox)
    printstyled(io, "$(indVarOf(pb)[begin])", color=:cyan)
end

function printIndVar(io::IO, @nospecialize(pb::ParamBox))
    print(io, "[")
    printIndVarCore(io, pb)
    print(io, "]")
end

function printIndVar(io::IO, @nospecialize(pbs::Tuple{Vararg{ParamBox}}))
    print(io, "[")
    for (i, pb) in pbs
        printIndVarCore(io, pb)
        i < length(pbs) && print(io, ", ")
    end
    print(io, "]")
end


function printValCore(io::IO, n::T) where {T<:Number}
    print(io, n isa Integer ? n : round(n, sigdigits=min(DefaultDigits, getAtolDigits(T))))
end

function printVal(io::IO, n::Number)
    print(io, "[")
    printValCore(io, n)
    print(io, "]")
end

function printVal(io::IO, v::Union{Vector{<:T}, Tuple{Vararg{T}}}, 
                  isVec::Bool=false) where {T<:Number}
    print(io, "[")
    print(io, ifelse(isVec, "(", "{"))
    for (i, n) in enumerate(v)
        print(io, 
              n isa Integer ? n : round(n, sigdigits=min(DefaultDigits, getAtolDigits(T))))
        i < length(v) && print(io, ", ")
    end
    print(io, ifelse(isVec, ")", "}"))
    print(io, "]")
end


function shortenTuplePB(str::String)
    TPBmarker = "#TPB#"
    str = replace(str, "Tuple{ParamBox{"=>TPBmarker*"{ParamBox{")
    str = ShortenStrClip(str, TPBmarker)
    replace(str, TPBmarker*"{…}"=>"Tuple{Vararg{ParamBox{…}}}")
end

function shortenFandTPB(str::String)
    str = replace(str, "$(iT)"=>"Quiqbox.iT")
    str = shortenTuplePB(str)
    for SF in AllStructFunctions
        str = ShortenStrClip(str, "$(SF)")
    end
    str
end

shortenFandTPB(::Type{T}) where {T} = (shortenF∘string)(T)

shortenFandTPB(obj) = (shortenF∘typeof)(obj)


function printAMconfig(io, (l,)::Tuple{LTuple})
    print(io, "[")
    printstyled(io, LtoStr(l), color=:cyan)
    print(io, "]")
end

function printAMconfig(io, ::NTuple{N, LTuple{D, L}}) where {N, D, L}
    SON = SubshellXYZsizes[L+1]
    xyzN = "$(N)"
    xyzNmax = "/$(SON)"
    print(io, "[")
    printstyled(io, xyzN, color=:cyan)
    print(io, xyzNmax)
    print(io, "]")
end


function getFieldNameStr(::Type{T}) where {T}
    fields = fieldnames(T)
    str = fields |> string
    length(fields) == 1 && (str = str[1:end-2]*")")
    replace(str, ':'=>"")
end


import Base: show
function show(io::IO, ::MIME"text/plain", @nospecialize(pb::ParamBox))
    Tstr = (string∘typeof)(pb)
    Tstr = shortenFandTPB(Tstr)
    omitLastTypeParPrint(io, Tstr)
    printFL(io, pb.map)
    symInfo = printDiffSym(io, pb)
    printIndVar(io, pb)
    print(io, symInfo)
    printVal(io, pb())
end

function show(io::IO, ::MIME"text/plain", @nospecialize(sp::SpatialPoint))
    Tstr = (string∘typeof)(sp)
    Tstr = shortenFandTPB(Tstr)
    omitLastTypeParPrint(io, Tstr, 2)
    printFL(io, getTypeParams(sp)[end])
    pbs = sp.param
    printDiffSym(io, pbs)
    printVal(io, coordOf(sp), true)
end

function show(io::IO, ::MIME"text/plain", @nospecialize(gf::GaussFunc))
    Tstr = (string∘typeof)(gf)
    Tstr = shortenFandTPB(Tstr)
    omitLastTypeParPrint(io, Tstr, 2)
    printFL(io, getTypeParams(gf)[end-1:end])
    pbs = gf.param
    printDiffSym(io, pbs)
    printVal(io, (gf.xpn(), gf.con()))
end

function show(io::IO, ::MIME"text/plain", @nospecialize(bf::FloatingGTBasisFuncs))
    Tstr = (string∘typeof)(bf)
    Tstr = shortenFandTPB(Tstr)
    omitLastTypeParPrint(io, Tstr, ifelse(bf isa BasisFunc, 1, 2))
    printFL(io, getTypeParams(bf)[end-1])
    printVal(io, centerCoordOf(bf), true)
    printAMconfig(io, bf.l)
end

function show(io::IO, ::MIME"text/plain", gb::GridBox)
    Tstr = (string∘typeof)(gb)
    Tstr = shortenFandTPB(Tstr)
    Tstr = ShortenStrClip(Tstr, "$(SpatialPoint)")
    omitLastTypeParPrint(io, Tstr)
    print(io, getFieldNameStr(gb|>typeof))
end

function show(io::IO, ::MIME"text/plain", obj::QuiqboxContainer, printFieldNames::Bool=true)
    Tstr = (string∘typeof)(obj)
    Tstr = shortenFandTPB(Tstr)
    print(io, Tstr)
    printFieldNames && print(io, getFieldNameStr(obj|>typeof))
end

show(io::IO, mime::MIME"text/plain", bfm::BasisFuncMix) = show(io, mime, bfm, false)

show(io::IO, ::MIME"text/plain", ::T) where {T<:EmptyBasisFunc} = print(io, T)


function show(io::IO, ::MIME"text/plain", @nospecialize(config::SCFconfig))
    print(io, typeof(config))
    str = getFieldNameStr(SCFconfig)
    str = replace(str, "interval"=>"interval=$(config.interval)")
    print(io, str)
end

function show(io::IO, ::MIME"text/plain", @nospecialize(config::HFconfig))
    print(io, typeof(config))
    str = getFieldNameStr(HFconfig)
    print(io, str)
end

function show(io::IO, ::MIME"text/plain", vars::HFtempVars{T}) where {T}
    print(io, typeof(vars))
    str = getFieldNameStr(HFtempVars)
    Etot0 = round(vars.shared.Etots[1], sigdigits=min(DefaultDigits, getAtolDigits(T)))
    EtotL = round(vars.shared.Etots[end], sigdigits=min(DefaultDigits, getAtolDigits(T)))
    str = replace(str, "shared"=>"shared.Etots=[$(Etot0), … , $(EtotL)]")
    print(io, str)
end

function show(io::IO, ::MIME"text/plain", vars::HFfinalVars{T}) where {T}
    print(io, typeof(vars))
    str = getFieldNameStr(HFfinalVars)
    Ehf = round(vars.Ehf, sigdigits=min(DefaultDigits, getAtolDigits(T)))
    str = replace(str, "Ehf"=>"Ehf=$(Ehf)")
    print(io, str)
end

function show(io::IO, ::MIME"text/plain", @nospecialize(config::POconfig))
    print(io, typeof(config))
    str = getFieldNameStr(POconfig)
    str = replace(str, "method,"=>"method=$(config.method),")
    print(io, str)
end

show(io::IO, ::MIME"text/plain", @nospecialize(matter::MatterByHF)) = 
print(io, typeof(matter), getFieldNameStr(MatterByHF))


show(io::IO, @nospecialize(obj::AbstractQuiqboxContainer)) = 
show(io, MIME"text/plain"(), obj)


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
iterate(pb::ParamBox) = (pb.data[][begin][], nothing)
iterate(@nospecialize(_::ParamBox), _) = nothing
size(::ParamBox) = ()
length(::ParamBox) = 1
eltype(::ParamBox{T}) where {T} = T
size(pb::ParamBox, d::Integer) = size(pb.data[][begin][], d)

iterate(sp::SpatialPoint) = iterate(sp.param)
iterate(sp::SpatialPoint, state) = iterate(sp.param, state)
size(::SpatialPoint{<:Any, D}) where {D} = (D,)
length(::SpatialPoint{<:Any, D}) where {D} = D
eltype(sp::SpatialPoint) = eltype(sp.param)
function size(::SpatialPoint{<:Any, D}, d::Integer) where {D}
    @boundscheck ( d > 0 || throw(BoundsError()) )
    ifelse(d==1, D, 1)
end

iterate(gf::GaussFunc) = (gf, nothing)
iterate(@nospecialize(_::GaussFunc), _) = nothing
size(::GaussFunc) = ()
length(::GaussFunc) = 1
size(::GaussFunc, d::Integer) = (d > 0) ? 1 : throw(BoundsError())

iterate(bf::CGTBasisFuncs1O) = (bf, nothing)
iterate(@nospecialize(_::CGTBasisFuncs1O), _) = nothing
size(::CGTBasisFuncs1O) = ()
length(::CGTBasisFuncs1O) = 1
eltype(::T) where {T<:CGTBasisFuncs1O} = T

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
eltype(::BasisFuncs{T, D, 𝑙, GN, F}) where {T, D, 𝑙, GN, F} = BasisFunc{T, D, 𝑙, GN, F}

function size(x::SpatialBasis, d::Integer)
    @boundscheck ( d > 0 || throw(BoundsError(x, d)) )
    ifelse(d==1, length(x), 1)
end

## Indexing Interface
import Base: getindex, setindex!, firstindex, lastindex, eachindex, axes
getindex(pb::ParamBox) = pb.data[][begin][]
getindex(pb::ParamBox, ::Val{:first}) = getindex(pb)
getindex(pb::ParamBox, ::Val{:last}) = getindex(pb)
setindex!(pb::ParamBox, d) = begin pb.data[][begin][] = d end
firstindex(@nospecialize(_::ParamBox)) = Val(:first)
lastindex(@nospecialize(_::ParamBox)) = Val(:last)
axes(@nospecialize(_::ParamBox)) = ()

getindex(container::ParameterizedContainer) = container.param

getindex(sp::SpatialPoint, is) = getindex(sp.param, is)
firstindex(sp::SpatialPoint) = firstindex(sp.param)
lastindex(sp::SpatialPoint) = lastindex(sp.param)
eachindex(sp::SpatialPoint) = eachindex(sp.param)
axes(sp::SpatialPoint) = axes(sp.param)

getindex(b::CGTBasisFuncs1O, ::Val{:first}) = itself(b)
getindex(b::CGTBasisFuncs1O, ::Val{:last}) = itself(b)
firstindex(@nospecialize(_::CGTBasisFuncs1O)) = Val(:first)
lastindex(@nospecialize(_::CGTBasisFuncs1O)) = Val(:last)

@inline function getindex(bf::CGTBasisFuncs1O, i::Int)
    @boundscheck ( i==1 || throw(BoundsError(bf, i)) )
    bf[begin]
end

getindex(bfs::BasisFuncs, is::AbstractVector{Int}) = 
BasisFuncs(bfs.center, bfs.gauss, bfs.l[is], bfs.normalizeGTO)
getindex(bfs::BasisFuncs, i::Int) = 
BasisFunc(bfs.center, bfs.gauss, bfs.l[i], bfs.normalizeGTO)
getindex(bfs::BasisFuncs, ::Colon) = itself(bfs)
firstindex(@nospecialize(_::BasisFuncs)) = 1
lastindex(::BFuncsON{ON}) where {ON} = ON
eachindex(bfs::BFuncsON) = Base.OneTo(lastindex(bfs))

getindex(xyz::LTuple) = xyz.tuple
getindex(xyz::LTuple, args) = getindex(xyz.tuple, args)
firstindex(xyz::LTuple) = firstindex(xyz.tuple)
lastindex(xyz::LTuple) = lastindex(xyz.tuple)
eachindex(xyz::LTuple) = eachindex(xyz.tuple)
axes(xyz::LTuple) = axes(xyz.tuple)


## Broadcasting Interface
import Base: broadcastable
broadcastable(pb::ParamBox) = Ref(pb)
broadcastable(gf::GaussFunc) = Ref(gf)
broadcastable(bf::CGTBasisFuncs1O) = Ref(bf)
broadcastable(bfs::BasisFuncs) = [i for i in bfs]
Base.broadcastable(sp::SpatialPoint) = Base.broadcastable(sp.param)


# Quiqbox methods overload.
## Method overload of `hasBoolRelation` from Tools.jl.
function hasBoolRelation(boolFunc::F, 
                         pb1::ParamBox{<:Any, V1, F1}, pb2::ParamBox{<:Any, V2, F2}; 
                         ignoreFunction::Bool=false, ignoreContainer::Bool=false, 
                         kws...) where {F<:Function, V1, V2, F1, F2}
    ifelse(ignoreContainer || V1 == V2, 
        ifelse((ignoreFunction || FLevel(F1) == FLevel(F2) == IL), 
            boolFunc(pb1.data[][begin], pb2.data[][begin]), 

            ( boolFunc(isDiffParam(pb1), isDiffParam(pb2)) && 
              hasBoolRelation(boolFunc, pb1.map, pb2.map) && 
              boolFunc(pb1.data[][begin], pb2.data[][begin]) )
        ), 

        false
    )
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
reshape(mapreduce(b->decomposeCore(Val(false), b), hcat, bs), :)

flatten(@nospecialize(bs::Tuple{Vararg{GTBasisFuncs{T, D}}} where {T, D})) = 
mapreduce(b->decomposeCore(Val(false), b), hcat, bs) |> Tuple

flatten(@nospecialize(bs::Union{AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                          Tuple{Vararg{FGTBasisFuncs1O{T, D}}}} where {T, D})) = 
itself(bs)


# The overload of following functions (or methods for specific types) are defined in 
# separate files to ensure precompilation capability.

## Methods for type __ : 
### LTuple

## Function __ : 
### getTypeParams
### getFLevel