export FLevel

"""

    DI{F<:Function} <: StructFunction{F}

A "dressed-up" [`itself`](@ref) that carries the information of a function (of type `F`). 
For an instance `di=$(DI)(someFunction)` where `someFunction isa Function`, 
`di(anyArgument) === anyArgument` and `di() === someFunction`.
"""
struct DI{F<:Function} <: StructFunction{F}
    f::F
    DI(f::F) where {F} = new{F}(f)
    DI(di::DI{F}) where {F} = new{F}(di.f)
end

(::DI)(x) = itself(x)
(di::DI)() = di.f

dressOf(::Type{DI{F}}) where {F} = F
dressOf(::Type{F}) where {F<:Function} = F


const IF = Union{iT, typeof(Base.identity)}

getFLevel(::Type{<:IF}) = 0
getFLevel(::Type{<:Function}) = 1
getFLevel(::Type{<:DI{F}}) where {F} = getFLevel(F)
getFLevel(::Type{<:CompositeFunction{F1, F2}}) where {F1, F2} = getFLevel(F1)+getFLevel(F2)
getFLevel(::Type{Function}) = Inf
getFLevel(::F) where {F<:Function} = getFLevel(F)

struct FLevel{L} <: MetaParam{FLevel} end

FLevel(::Type{F}) where {F<:Function} = FLevel{getFLevel(F)}
FLevel(::F) where {F<:Function} = FLevel(F)
FLevel(L::Int) = FLevel{L}

const IL = FLevel(itself)


struct Layered{F1<:Function, F2} <: CompositeFunction{F1, F2}
    inner::F1
    outer::F2
    Layered(inner::F1, outer::F2) where {F1, F2} = new{F1, F2}(inner, outer)
end

(lf::Layered)(x) = (lf.outer∘lf.inner)(x)

Absolute(f) = Layered(f, abs)

const Labs{F} = Layered{F, typeof(abs)}

Absolute(f::Labs) = itself(f)


const PFfusingOperators = Dict([*, +, ^] .=> [*, +, *])

struct PF{F, O, T} <: ChainedFunction{F, O, PF}
    f::F
    o::O
    c::T
end
PF(pf::PF{F, O, T1}, o::O, c::T2) where {F, O, T1, T2} = 
PF{F, O, promote_type(T1, T2)}(pf.f, o, PFfusingOperators[o](pf.c, c))
(pf::PF)(x) = pf.o(pf.f(x), pf.c)

struct ChainedPF{FI, N, OS<:NTuple{N, Function}, 
                        TS<:NTuple{N, Any}} <: ChainedFunction{FI, OS, ChainedPF}
    fi::FI
    os::OS
    cs::TS

    ChainedPF(fi::FI, os::OS, cs::TS) where 
             {FI<:Function, N, OS<:NTuple{N, Function}, TS<:NTuple{N, Any}} = 
    new{FI, N, OS, TS}(fi, os, cs)
end

PF(pf::PF, o::O, c) where {O} = ChainedPF(pf.f, (pf.o, o), (pf.c, c))

PF(cf::ChainedPF, o::O, c) where {O} = ChainedPF(cf.fi, (cf.os..., o), (cf.cs..., c))

(cf::ChainedPF{FI, N})(x) where {FI, N} = chainPF(cf.fi(x), cf.os, cf.cs, Val(N))

chainPF(x, os, cs, ::Val{N}) where {N} = os[N](chainPF(x, os, cs, Val(N-1)), cs[N])
chainPF(x, os, cs, ::Val{1}) = os[begin](x, cs[begin])

# Return Constant
struct RC{T} <: StructuredFunction
    c::T
end
(rc::RC{T})(_) where {T} = rc.c


function combinePF(::typeof(+), pf1::PF{F, typeof(*)}, pf2::PF{F, typeof(*)}) where {F}
    c = (pf1.c + pf2.c)
    iszero(c) ? RC(typeof(c)(0)) : (isone(c) ? pf1.f : PF(pf1.f, *, c))
end

function combinePF(::typeof(+), pf1::PF{F, typeof(+)}, pf2::PF{F, typeof(+)}) where {F}
    c = pf1.c + pf2.c
    res = PF(pf1.f, *, 2)
    iszero(c) ? res : PF(res, +, c)
end

function combinePF(::typeof(+), pf1::PF{F, typeof(*)}, pf2::PF{F, typeof(+)}) where {F}
    c = pf1.c + 1
    iszero(c) ? RC(pf2.c) : (isone(c) ? PF(pf1.f, +, pf2.c) : PF(PF(pf1.f, *, c), +, pf2.c))
end

combinePF(::typeof(+), pf1::PF{F, typeof(+)}, pf2::PF{F, typeof(*)}) where {F} = 
combinePF(+, pf2, pf1)

combinePF(op::Function, pf1::Function, pf2::Function) = x->op(pf1(x), pf2(x))

function combinePF(::typeof(*), pf1::PF{F, typeof(*)}, pf2::PF{F, typeof(*)}) where {F}
    c = pf1.c * pf2.c
    iszero(c) ? RC(typeof(c)(0)) : (isone(c) ? PF(pf1.f, ^, 2) : PF(PF(pf1.f, ^, 2), *, c))
end

function combinePF(::typeof(*), pf1::PF{F, typeof(^)}, pf2::PF{F, typeof(^)}) where {F}
    c = pf1.c + pf2.c
    iszero(c) ? RC(typeof(c)(1)) : (isone(c) ? pf1.f : PF(pf1.f, ^, c))
end

function combinePF(::typeof(*), pf1::PF{F, typeof(^)}, pf2::PF{F, typeof(*)}) where {F}
    c = pf1.c + 1
    if iszero(c)
        RC(pf2.c)
    else
        f = isone(c) ? pf1.f : PF(pf1.f, ^, c)
        PF(f, *, pf2.c)
    end
end

combinePF(::typeof(*), pf1::PF{F, typeof(*)}, pf2::PF{F, typeof(^)}) where {F} = 
combinePF(*, pf2, pf1)


const AllStructFunctions = [DI, Layered, PF, ChainedPF, RC]