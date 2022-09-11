export FLevel

struct DressedItself{L, F<:Function} <: StructFunction{F}
    f::F
    DressedItself(f::F) where {F} = new{getFLevel(F), F}(f)
    DressedItself(di::DressedItself{L, F}) where {L, F} = new{L, F}(di.f)
end

(::DressedItself)(x) = itself(x)
(di::DressedItself)() = di.f


struct FLevel{L} <: MetaParam{FLevel} end

FLevel(::Type{itselfT}) = FLevel{0}
FLevel(::Type{typeof(Base.identity)}) = FLevel{0}
FLevel(::Type{<:Function}) = FLevel{1}
FLevel(::Type{<:DressedItself{L}}) where {L} = FLevel{L}
FLevel(::Type{<:CompositeFunction{F2, F1}}) where {F1, F2} = 
FLevel{getFLevel(F2)+getFLevel(F1)}
FLevel(::F) where {F<:Function} = FLevel(F)

const FI = FLevel(itself)

getFLevel(::Type{FLevel{L}}) where {L} = L
getFLevel(::Type{F}) where {F<:Function} = (getFLevel∘FLevel)(F)
getFLevel(::F) where {F<:Function} = getFLevel(F)
getFLevel(sym::Symbol) = (getFLevel∘getFunc)(sym)


# struct Inverse{F<:Function} <: StructFunction{F} end

struct Layered{F2, F1<:Function} <: CompositeFunction{F2, F1}
    outer::F2
    inner::F1
    Layered(outer::F2, inner::F1) where {F1, F2} = new{F2, F1}(outer, inner)
    # Layered(::Inverse{F}, ::F) where {F} = itself
    # Layered(::F, ::Inverse{F}) where {F} = itself
end

(lf::Layered)(x) = (lf.outer∘lf.inner)(x)

Absolute(f) = Layered(abs, f)


# Product Function
struct Pf{F<:Function, T} <: ParameterizedFunction{Pf, F, T} # Pf{T} ?
    f::F
    c::T
end
Pf(f::Pf{F, T}, c) where {F<:Function, T} = Pf{F, T}(f.f, f.c*T(c))
(f::Pf)(x::T) where {T} = f.f(x)*f.c


# Sum Function
struct Sf{F<:Function, T} <: ParameterizedFunction{Sf, F, T}
    f::F
    c::T
end
Sf(f::Sf{F, T}, c) where {F<:Function, T} = Sf{F, T}(f.f, f.c+T(c))
(f::Sf)(x::T) where {T} = f.f(x)+f.c

# Exponent Function
struct Xf{F<:Function, T} <: ParameterizedFunction{Xf, F, T}
    f::F
    c::T
end
Xf(f::Xf{F, T}, c) where {F<:Function, T} = Xf{F, T}(f.f, f.c*T(c))
(f::Xf)(x::T) where {T} = f.f(x)^f.c


function combineParFunc(::typeof(+), pf1::Pf{F}, pf2::Pf{F}) where {F}
    c = (pf1.c + pf2.c)
    isone(c) ? pf1.f : Pf(pf1.f, c)
end

function combineParFunc(::typeof(+), pf1::Sf{F}, pf2::Sf{F}) where {F}
    c = pf1.c + pf2.c
    iszero(c) ? Pf(pf1.f, 2) : Sf(f(pf1.f, 2), c)
end

function combineParFunc(::typeof(+), pf1::Pf{F}, pf2::Sf{F}) where {F}
    c = pf1.c + 1
    isone(c) ? Sf(pf1.f, pf2.c) : Sf(Pf(pf1.f, c), pf2.c)
end

combineParFunc(::typeof(+), pf1::Sf{F}, pf2::Pf{F}) where {F} = 
combineParFunc(+, pf2, pf1)

combineParFunc(op::Function, pf1::Function, pf2::Function) = x->op(pf1(x), pf2(x))

function combineParFunc(::typeof(*), pf1::Pf{F}, pf2::Pf{F}) where {F}
    c = pf1.c * pf2.c
    isone(c) ? Xf(pf1.f, 2) : Pf(Xf(pf1.f, 2), c)
end

function combineParFunc(::typeof(*), pf1::Xf{F}, pf2::Xf{F}) where {F}
    c = pf1.c + pf2.c
    iszero(c) ? (_->1) : (isone(c) ? pf1.f : Xf(pf1.f, c))
end

function combineParFunc(::typeof(*), pf1::Xf{F}, pf2::Pf{F}) where {F}
    c = pf1.c + 1
    f = iszero(c) ? (_->1) : (isone(c) ? pf1.f : Xf(pf1.f, c))
    Pf(f, pf2.c)
end

combineParFunc(::typeof(*), pf1::Sf{F}, pf2::Pf{F}) where {F} = 
combineParFunc(+, pf2, pf1)