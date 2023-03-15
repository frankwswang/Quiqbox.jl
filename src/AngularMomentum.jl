function check1j(j)
    if j < 0 || !isinteger(2j)
        (throw∘DomainError)(j, "The value of angular momentum j is illegal.")
    end
end

function check1m(j, m)
    if abs(m) > j || !isinteger(m)
        (throw∘DomainError)(m, "The value of magnetic quantum number m is illegal.")
    end
end

function check3j(js)
    check1j.(js)
    j1, j2, j3 = js
    if !(isinteger∘sum)(js) || (j3 > j1+j2) || (j3 < abs(j1-j2))
        (throw∘DomainError)(js, "The combination of input angular momentums is illegal.")
    end
end


struct CGcoeff{T1, T2, DJ1, DJ2, DJ3}
    m1::T1
    m2::T1
    coeffSquare::T2
    function CGcoeff(j1::T, m1::T, j2::T, m2::T, j3::T) where {T}
        js = (j1, j2, j3)
        ms = (m1, m2, m1+m2)
        check3j(js)
        check1m.(js, ms)
        cSquare = getCGcoeffCore(j1, m1, j2, m2, j3)
        new{T, typeof(cSquare), Int(2j1), Int(2j2), Int(2j3)}(m1, m2, cSquare)
    end

    CGcoeff(dbjs::NTuple{3, Int}, m1::T1, m2::T1, cSquare::T2) where {T1, T2} = 
    new{T1, T2, dbjs[1], dbjs[2], dbjs[3]}(m1, m2, cSquare)
end


function genCGcoeff(op::F, cgc1::CGcoeff{T1, T2, DJ1, DJ2, DJ3}, 
                    cgc2::CGcoeff{T1, T2, DJ1, DJ2, DJ3}) where 
                   {F<:Union{typeof(+), typeof(-)}, T1, T2, DJ1, DJ2, DJ3}
    if (diff = (cgc1.m1 - cgc2.m1)) == (cgc2.m2 - cgc1.m2) && abs(abs(diff)-1) < 2numEps(T1)
        if cgc1.m1 > cgc2.m1
            cgc1, cgc2 = cgc2, cgc1
        end
        m1 = max(op(cgc1.m1), op(cgc2.m1)) |> op
        m2 = max(op(cgc1.m2), op(cgc2.m2)) |> op
        m3 = m1 + m2
        j1 = DJ1 / 2
        j2 = DJ2 / 2
        j3 = DJ3 / 2
        c1sq = cgc1.coeffSquare
        c2sq = cgc2.coeffSquare
        rc1 = (op(j1, -m1) + 1) * op(j1, m1)
        rc2 = (op(j2, -m2) + 1) * op(j2, m2)
        rc3 = (op(j3, -m3) + 1) * op(j3, m3)
        @show c1sq c2sq rc1 rc2
        cSquare = ( 2√(c1sq*c2sq*rc1*rc2) + rc1*c1sq + rc2*c2sq ) / rc3
        CGcoeff((Int(2j1), Int(2j2), DJ3), m1, m2, cSquare)
    else
        (throw∘DomainError)((diff, cgc2.m2-cgc1.m2), 
                            "The differences between each two ms are not consistent.")
    end
end


function getCGcoeffCore(j₁::Real, m₁::Real, j₂::Real, m₂::Real, j::Real)
    m = m₁+m₂
    ss = (j₁+j₂-j, j₁-m₁, j₂+m₂, j-j₂+m₁, j-j₁-m₂)
    p1 = fct(j+m)*fct(j-m) * fct(j₁+m₁)*fct(j₁-m₁) * fct(j₂+m₂)*fct(j₂-m₂) * 
            (2j+1) * fct(j+j₁-j₂)*fct(j-j₁+j₂)*fct(j₁+j₂-j) / fct(j₁+j₂+j+1)
    kUpper = min(ss[1], ss[2], ss[3]) |> Int
    kLower = max(0, -min(ss[4], ss[5])) |> Int
    p2 = mapreduce(+, kLower:kUpper) do k
        (-1)^k / 
        ( fct(k) * 
            fct(ss[1] - k)*fct(ss[2] - k)*fct(ss[3] - k) * 
            fct(ss[4] + k)*fct(ss[5] + k) )
    end
    p1 * p2^2
end


function getCGcoeff(j₁::Real, m₁::Real, j₂::Real, m₂::Real, j₃::Real)
    js = (j₁, j₂, j₃)
    ms = (m₁, m₂, m₁+m₂)
    check3j(js)
    check1m.(js, ms)
    √getCGcoeffCore(j₁, m₁, j₂, m₂, j₃)
end

getCGcoeff(cgc::CGcoeff) = √cgc.coeffSquare

function getCGcoeff(op::Function, cgc1::CGcoeff, cgc2::CGcoeff)
    cgc3 = genCGcoeff(op, cgc1, cgc2)
    getCGcoeff(cgc3)
end