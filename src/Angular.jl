abstract type SphericalHarmonics{D} <: CompositeFunction end

abstract type RealSolidHarmonics{D} <: SphericalHarmonics{D} end


struct WeakComp{N} # Weak composition of an integer
    tuple::NTuple{N, Int}
    total::Int

    function WeakComp(t::NonEmptyTuple{Int, M}) where {M}
        if any(i < 0 for i in t)
            throw(DomainError(t, "The element(s) of `t` should all be non-negative."))
        end
        new{M+1}(t, sum(t))
    end
end


struct CartSHarmonics{D} <: RealSolidHarmonics{D}
    m::WeakComp{D}

    function CartSHarmonics(t::NTuple{D, Int}) where {D}
        new{D}(WeakComp(t))
    end
end

function (csh::CartSHarmonics{D})(dr::NTuple{D, Real}) where {D}
    prod(dr .^ csh.m.tuple)
end

struct PureSHarmonics{D} <: RealSolidHarmonics{D}
    l::Int
    m::Int

    function PureSHarmonics(::Val{D}, l::Int, m::Int) where {D}
        checkPositivity(D::Int)
        if l < abs(m)
            throw(AssertionError("`l` should be the upper bound of `m`'s absolute value."))
        end
        new{D}(l, m)
    end
end

function (psh::PureSHarmonics{D})(r::NTuple{D, Real}) where {D}
    evalPureSHarmonicsCore(Val(D), psh.l, psh.m, r)
end


function get3DimCtoPSHarmonicsCoeffN(l::Int, m::Int, ::Type{T}=Float64) where {T<:Real}
    l = abs(l)
    T(1) / (2^abs(m) * factorial(l)) * 
    sqrt( T(2) * factorial(l + abs(m)) * factorial(l - abs(m)) / (1 + m==0) )
end

function get3DimCtoPSHarmonicsCoeffC(l::Int, (m, t, u, v)::NTuple{4, Int}, 
                                     vm::Real, ::Type{T}=Float64) where {T<:Real}
    l = abs(l)
    ifelse(isodd(t + v - vm), -1, 1) * (T(1) / 4)^t * 
    binomial(l, t) * binomial(l-t, abs(m)+t) * binomial(t, u) * binomial(abs(m), 2v)
end


function evalPureSHarmonicsCore(::Val{3}, l::Int, m::Int, r::NTuple{3, Real})
    l = abs(l)
    res = T(0)
    vm = ifelse(m<0, T(1)/2, T(0))
    for t in 0:( (l::Int - abs(m))/2 ), u in 1:t, v in vm:( (abs(m)/2 - vm) + vm )
        m = WeakComp( (2t+abs(m)-2(u+v), 2(u+v), l-2t-abs(m)) )
        res += get3DimCtoPSHarmonicsCoeffC(Val(3), (m, t, u, v), vm, T) * 
               evalCartSHarmonicsCore(Val(3), m, r)
    end
    res * get3DimCtoPSHarmonicsCoeffN(l, m, T)
end

#########################################################

# Clebsch–Gordan coefficients & angular momentum addition

function check1j1m(j, m)
    if j < 0 || !isinteger(2j) || abs(m) > j || !isinteger(2m)
        (throw∘DomainError)((j, m), 
                            "The information of the angular momentum (j, m) is illegal.")
    end
end


function check3j(js)
    j1, j2, j3 = js
    if !(isinteger∘sum)(js) || (j3 > j1+j2) || (j3 < abs(j1-j2))
        (throw∘DomainError)(js, "The combination of input angular momentums is illegal.")
    end
end


struct CGcoeff{T<:Real, DJ1, DJ2, DJ3}
    m1::T
    m2::T
    coeff::T
    function CGcoeff(j1::Real, m1::Real, j2::Real, m2::Real, j3::Real)
        js = (j1, j2, j3)
        ms = (m1, m2, m1+m2)
        check1j1m.(js, ms)
        check3j(js)
        c = getCGcoeffCore(j1, m1, j2, m2, j3)
        new{typeof(c), Int(2j1), Int(2j2), Int(2j3)}(m1, m2, c)
    end

    CGcoeff(dbjs::NTuple{3, Int}, m1::Real, m2::Real, c::T) where {T} = 
    new{T, dbjs[1], dbjs[2], dbjs[3]}(m1, m2, c)
end


function genCGcoeffCoreCore(op::F, m1::Real, m2::Real, 
                            cgcA::CGcoeff{T1, DJ1, DJ2, DJ3}, 
                            cgcB::CGcoeff{T2, DJ1, DJ2, DJ3}) where 
                           {F<:Function, T1, T2, DJ1, DJ2, DJ3}
    m3 = m1 + m2
    j1 = 0.5DJ1
    j2 = 0.5DJ2
    j3 = 0.5DJ3
    cA = cgcA.coeff
    cB = cgcB.coeff
    rc1 = (j1 + op(-m1) + 1) * (j1 + op(m1))
    rc2 = (j2 + op(-m2) + 1) * (j2 + op(m2))
    rc3 = (j3 + op(-m3) + 1) * (j3 + op(m3))
    (cA, cB), (rc1, rc2, rc3)
end

function genCGcoeffCoreHypo(op::F, cgc1::CGcoeff{T1, DJ1, DJ2, DJ3}, 
                                   cgc2::CGcoeff{T2, DJ1, DJ2, DJ3}) where 
                           {F<:Union{typeof(+), typeof(-)}, T1, T2, DJ1, DJ2, DJ3}
    if cgc1.m1 > cgc2.m1
        cgc1, cgc2 = cgc2, cgc1
    end
    m1 = max(op(cgc1.m1), op(cgc2.m1)) |> op
    m2 = max(op(cgc1.m2), op(cgc2.m2)) |> op
    (c1, c2), (rc1, rc2, rc3) = genCGcoeffCoreCore(op, m1, m2, cgc1, cgc2)
    c = ( √(rc1 / rc3)*c1 + √(rc2 / rc3)*c2 )
    CGcoeff((DJ1, DJ2, DJ3), m1, m2, c)
end

function genCGcoeffCoreCath(op::F, cgc1::CGcoeff{T1, DJ1, DJ2, DJ3}, 
                                   cgc3::CGcoeff{T2, DJ1, DJ2, DJ3}) where 
                           {F<:Union{typeof(+), typeof(-)}, T1, T2, DJ1, DJ2, DJ3}
    if op(cgc1.m1+cgc1.m2) < op(cgc3.m1+cgc3.m2)
        cgc1, cgc3 = cgc3, cgc1
    end
    m1 = cgc3.m1
    m2 = cgc3.m2
    (c1, c3), (rc1, rc2, rc3) = genCGcoeffCoreCore(∘(op, -), m1, m2, cgc1, cgc3)
    # c = ( -2√(c1sq*c3sq*rc1*rc3) + rc1*c1sq + rc3*c3sq ) / rc2
    c = √(rc3 / rc2)*c3 - √(rc1 / rc2)*c1
    CGcoeff((DJ1, DJ2, DJ3), m1+(op∘fastIsApprox)(cgc1.m1, m1), 
                             m2+(op∘fastIsApprox)(cgc1.m2, m2), c)
end

function genCGcoeff(op::F, cgc1::CGcoeff{T1, DJ1, DJ2, DJ3}, 
                           cgc2::CGcoeff{T2, DJ1, DJ2, DJ3}) where 
                   {F<:Union{typeof(+), typeof(-)}, T1, T2, DJ1, DJ2, DJ3}
    diff1 = cgc1.m1 - cgc2.m1
    diff2 = cgc2.m2 - cgc1.m2
    diff1Abs = abs(diff1)
    diff2Abs = abs(diff2)
    if fastIsApprox(diff1, diff2) && fastIsApprox(diff1Abs, 1)
        genCGcoeffCoreHypo(op, cgc1, cgc2)
    elseif (fastIsApprox(diff2Abs) && fastIsApprox(diff1Abs, 1)) || 
           (fastIsApprox(diff1Abs) && fastIsApprox(diff2Abs, 1))
        genCGcoeffCoreCath(op, cgc1, cgc2)
    else
        (throw∘DomainError)((diff1, diff2), 
                            "The differences between each two ms are not consistent.")
    end
end


function getCGcoeffCore(j₁::Real, m₁::Real, j₂::Real, m₂::Real, j::Real)
    m = m₁+m₂
    ss = (j₁+j₂-j, j₁-m₁, j₂+m₂, j-j₂+m₁, j-j₁-m₂)
    p1 = fct(j+m)*fct(j-m) * fct(j₁+m₁)*fct(j₁-m₁) * fct(j₂+m₂)*fct(j₂-m₂) * 
            (2j+1) * ( fct(j+j₁-j₂)*fct(j-j₁+j₂)*fct(j₁+j₂-j) / fct(j₁+j₂+j+1) )
    kUpper = min(ss[1], ss[2], ss[3]) |> Int
    kLower = max(0, -min(ss[4], ss[5])) |> Int
    p2 = mapreduce(+, kLower:kUpper) do k
        (-1)^k / 
        ( fct(k) * fct(ss[1] - k) * fct(ss[2] - k) * fct(ss[3] - k) * 
                   fct(ss[4] + k) * fct(ss[5] + k) )
    end
    sqrt(p1) * p2
end


function getCGcoeff(j₁::Real, m₁::Real, j₂::Real, m₂::Real, j₃::Real)
    js = (j₁, j₂, j₃)
    ms = (m₁, m₂, m₁+m₂)
    check1j1m.(js, ms)
    check3j(js)
    getCGcoeffCore(j₁, m₁, j₂, m₂, j₃)
end

getCGcoeff(cgc::CGcoeff) = cgc.coeff

function getCGcoeff(op::Function, cgc1::CGcoeff, cgc2::CGcoeff)
    cgc3 = genCGcoeff(op, cgc1, cgc2)
    getCGcoeff(cgc3)
end