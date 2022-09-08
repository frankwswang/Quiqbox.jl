export FLevel

struct FLevel{L} <: MetaParam{FLevel} end

FLevel(::Type{itselfT}) = FLevel{0}
FLevel(::Type{typeof(Base.identity)}) = FLevel{0}
FLevel(::Type{<:Function}) = FLevel{1}
FLevel(::Type{<:DressedFunction{L}}) where {L} = FLevel{L}
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


struct DressedItself{L, F<:Function} <: DressedFunction{L, F}
    f::F
    DressedItself(f::F) where {F} = new{getFLevel(F), F}(f)
    DressedItself(di::DressedItself{L, F}) where {L, F} = new{L, F}(di.f)
end

(::DressedItself)(x) = itself(x)
(di::DressedItself)() = di.f


# Product Function
struct Pf{T, F<:Function} <: ParameterizedFunction{Pf, F}
    c::T
    f::F
end
Pf(c, f::Pf{T, F}) where {T, F<:Function} = Pf{T, F}(f.c*T(c), f.f)
(f::Pf)(x::T) where {T} = f.c * f.f(x)


# Sum Function
struct Sf{T, F<:Function} <: ParameterizedFunction{Sf, F}
    c::T
    f::F
end
Sf(c, f::Sf{T, F}) where {T, F<:Function} = Sf{T, F}(f.c+T(c), f.f)
(f::Sf)(x::T) where {T} = f.c + f.f(x)

# Exponent Function
struct Xf{P, F<:Function} <: ParameterizedFunction{Xf, F}
    f::F
    Xf(P::Int, f::F) where {F<:Function} = new{P, F}(f)
end
Xf(PN::Int, f::Xf{P, F}) where {P, F<:Function} = Xf(P+PN, f.f)
(f::Xf{P})(x) where {P} = f.f(x)^P


function combineParFunc(::typeof(+), pf1::F1, pf2::F2) where 
                       {F, F1<:ParameterizedFunction{Pf, F}, 
                           F2<:ParameterizedFunction{Pf, F}}
    c = (pf1.c + pf2.c)
    isone(c) ? pf1.f : Pf(c, pf1.f)
end

function combineParFunc(::typeof(+), pf1::F1, pf2::F2) where 
              {F, F1<:ParameterizedFunction{Sf, F}, F2<:ParameterizedFunction{Sf, F}}
    c = pf1.c + pf2.c
    iszero(c) ? Pf(2, pf1.f) : Sf(c, Pf(2, pf1.f))
end

function combineParFunc(::typeof(+), pf1::F1, pf2::F2) where 
              {F, F1<:ParameterizedFunction{Pf, F}, F2<:ParameterizedFunction{Sf, F}}
    c = pf1.c + 1
    isone(c) ? Sf(pf2.c, pf1.f) : Sf(pf2.c, Pf(c, pf1.f))
end

combineParFunc(::typeof(+), pf1::F1, pf2::F2) where 
              {F, F1<:ParameterizedFunction{Sf, F}, F2<:ParameterizedFunction{Pf, F}} = 
combineParFunc(+, pf2, pf1)

combineParFunc(op::Function, pf1::Function, pf2::Function) = x->op(pf1(x), pf2(x))

function combineParFunc(::typeof(*), pf1::F1, pf2::F2) where 
              {F, F1<:ParameterizedFunction{Pf, F}, F2<:ParameterizedFunction{Pf, F}}
    c = pf1.c * pf2.c
    isone(c) ? Xf(2, pf1.f) : Pf(c, Xf(2, pf1.f))
end

function combineParFunc(::typeof(*), pf1::Xf{P1, F}, pf2::Xf{P2, F}) where {P1, P2, F}
    P = P1 + P2
    iszero(P) ? (_->1) : (isone(P) ? pf1.f : Xf(P, pf1.f))
end

function combineParFunc(::typeof(*), pf1::Xf{P, F}, pf2::Pf{<:Any, F}) where {P, F}
    P2 = P + 1
    f = iszero(P2) ? (_->1) : (isone(P2) ? pf1.f : Xf(P2, pf1.f))
    Pf(pf2.c, f)
end

combineParFunc(::typeof(*), pf1::F1, pf2::F2) where 
              {F, F1<:ParameterizedFunction{Sf, F}, F2<:ParameterizedFunction{Pf, F}} = 
combineParFunc(+, pf2, pf1)