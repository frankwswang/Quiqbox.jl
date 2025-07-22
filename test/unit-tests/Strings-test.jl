using Test
using Quiqbox
using Quiqbox: advancedParse, alignNum, alignNumSign, numToSups, superscriptNum, numToSubs, 
               subscriptNum

@testset "Strings.jl tests" begin

# function advancedParse (with adaptiveParse)
@test advancedParse(Float64, "1") === 1.0
@test advancedParse(Float64, "1.0") === 1.0
@test advancedParse(Float64, "1.0 + im") === "1.0 + im"
@test advancedParse(Float64, "1 + 1im") === 1.0 + 1.0im
@test advancedParse(Float64, "1.0 + 1.1im") === 1.0 + 1.1im
@test advancedParse(BigFloat, "1.0") isa BigFloat
@test advancedParse(BigFloat, "1.0") == BigFloat(1.0)


# function alignNum
@test alignNum(1.234f0, 3) ==    "  1.234        "
@test alignNum(1.234f2, 3) ==    "123.4          "
@test alignNum(1.234f3, 3) ==   "1234.0          "
@test alignNum(1.234f4, 3) ==  "12340.0          "
@test alignNum(1.234f5, 3) == "123400.0          "
@test alignNum(1.234f6, 3) ==    "  1.234e6      "
@test alignNum(1.234f10,3) ==    "  1.234e10     "

@test alignNum(1.234e0, 3) ==    "  1.234                 "
@test alignNum(1.234e2, 3) ==    "123.4                   "
@test alignNum(1.234e3, 3) ==   "1234.0                   "
@test alignNum(1.234e4, 3) ==  "12340.0                   "
@test alignNum(1.234e5, 3) == "123400.0                   "
@test alignNum(1.234e6, 3) ==    "  1.234e6               "
@test alignNum(1.234e10,3) ==    "  1.234e10              "
@test alignNum(1.234e100,3) ==   "  1.234e100             "

@test alignNum(1.234f0, 3, roundDigits=6) ==     "  1.234000"
@test alignNum(1.234f2, 3, roundDigits=6) ==     "123.4000  "
@test alignNum(1.234f3, 3, roundDigits=6) ==    "1234.000   "
@test alignNum(1.234f4, 3, roundDigits=6) ==   "12340.00    "
@test alignNum(1.234f5, 3, roundDigits=6) ==  "123400.0     "
@test alignNum(1.234f6, 3, roundDigits=6) ==  "1234000      "

@test alignNum(1.234e0, 3, roundDigits=6) ==             "  1.234000"
@test alignNum(1.234e2, 3, roundDigits=6) ==             "123.400000"
@test alignNum(1.234e3, 3, roundDigits=6) ==            "1234.000000"
@test alignNum(1.234e4, 3, roundDigits=6) ==           "12340.000000"
@test alignNum(1.234e14,3, roundDigits=6) == "123400000000000.0     "
@test alignNum(1.234e15,3, roundDigits=6) == "1234000000000000      "
@test alignNum(1.234e100,3,roundDigits=6) == "123400000000000000000000000000000000000000"*
                                             "000000000000000000000000000000000000000000"*
                                             "00000000000000000      "

@test alignNum(1.23f0, 3, roundDigits=6) == "  1.230000"
@test alignNum(1.23f1, 3, roundDigits=6) == " 12.30000 "
@test alignNum(1.23f2, 3, roundDigits=6) == "123.0000  "
@test alignNum(1.23e2, 3, roundDigits=6) == "123.000000"
@test alignNum(1.23f3, 3, roundDigits=6) == "  1.23e+03" == 
      alignNum(1.23e3, 3, roundDigits=6)
@test alignNum(1.23f6, 3, roundDigits=6) == "  1.23e+06" ==
      alignNum(1.23e6, 3, roundDigits=6)
@test alignNum(1.23f9, 3, roundDigits=6) == "  1.23e+09" ==
      alignNum(1.23e9, 3, roundDigits=6)

@test alignNum(-1.23f0, 3, roundDigits=6) == " -1.230000"
@test alignNum(-1.23f1, 3, roundDigits=6) == "-12.30000 "
@test alignNum(-1.23f2, 3, roundDigits=6) == " -1.23e+02" == 
      alignNum(-1.23e2, 3, roundDigits=6)

###

@test alignNum(1.234e-1, 3) == "  0.1234                " == 
      alignNum(1.234f-1, 3) * repeat(" ", 9)
@test alignNum(1.234e-4, 3) == "  0.0001234             " == 
      alignNum(1.234f-4, 3) * repeat(" ", 9)
@test alignNum(1.234e-5, 3) == "  1.234e-5              " == 
      alignNum(1.234f-5, 3) * repeat(" ", 9)
@test alignNum(1.234e-10,3) == "  1.234e-10             " == 
      alignNum(1.234f-10, 3) * repeat(" ", 9)
@test alignNum(1.234e-100,3) =="  1.234e-100            "

@test alignNum(1.234f-1, 3, roundDigits=6) == "  0.123400" == 
      alignNum(1.234e-1, 3, roundDigits=6)
@test alignNum(1.234f-3, 3, roundDigits=6) == "  0.001234" == 
      alignNum(1.234e-3, 3, roundDigits=6)
@test alignNum(1.234f-4, 3, roundDigits=6) == "  0.000123" == 
      alignNum(1.234e-4, 3, roundDigits=6)
@test alignNum(1.234f-5, 3, roundDigits=6) == "  1.23e-05" == 
      alignNum(1.234e-5, 3, roundDigits=6)
@test alignNum(1.234f-10,3, roundDigits=6) == "  1.23e-10" == 
      alignNum(1.234e-10,3, roundDigits=6)
@test alignNum(1.234e-100,3,roundDigits=6) == "  1.2e-100"

@test alignNum(1.23f-3, 3, roundDigits=6) == "  0.001230" ==
      alignNum(1.23e-3, 3, roundDigits=6)
@test alignNum(1.23f-4, 3, roundDigits=6) == "  1.23e-04" ==
      alignNum(1.23e-4, 3, roundDigits=6)
@test alignNum(1.23f-5, 3, roundDigits=6) == "  1.23e-05" ==
      alignNum(1.23e-5, 3, roundDigits=6)
@test alignNum(1.23f-10,3, roundDigits=6) == "  1.23e-10" ==
      alignNum(1.23e-10,3, roundDigits=6)
@test alignNum(1.23e-100,3,roundDigits=6) == "  1.2e-100"

@test alignNum(-1.23f-1, 3, roundDigits=6) == " -0.123000"
@test alignNum(-1.23f-2, 3, roundDigits=6) == " -0.012300" == 
      alignNum(-1.23e-2, 3, roundDigits=6)

@test alignNum(NaN) == "     NaN  " == alignNum(NaN32)
@test alignNum(NaN, 2) == "NaN  " == alignNum(NaN, 3)
@test alignNum(NaN32, 2) ==  "NaN  " == alignNum(NaN32, 3)
@test alignNum(NaN, roundDigits=6) == "     NaN      " == alignNum(NaN32, roundDigits=6)

@test alignNum( 0.0, 2, roundDigits=5) == " 0.00000"
@test alignNum(-0.0, 2, roundDigits=5) == "-0.00000"
@test alignNum( 0.0, 1, roundDigits=5) ==  "0.00000"
@test alignNum(-0.0, 1, roundDigits=5) == "-0.00000"
@test alignNum( 0.0, 0, roundDigits=5) ==  "0.00000"
@test alignNum(-0.0, 0, roundDigits=5) == "-0.00000"
@test alignNum( 0.0,-1, roundDigits=5) ==  "0.00000"
@test alignNum(-0.0,-1, roundDigits=5) == "-0.00000"
@test alignNum( 0.0,-1) ==  "0.0                   "
@test alignNum(-0.0,-1) == "-0.0                   "


# function alignNumSign
@test alignNumSign(-1) == "-1"
@test alignNumSign( 1) == " 1"
@test alignNumSign(-1.2) == "-1.2"
@test alignNumSign(+1.2) == " 1.2"
@test alignNumSign( 1, 3) == " 1 "
@test alignNum( 1.2, 10, 0) == repeat(" ", 8) * alignNumSign( 1.2)
@test alignNum(-1.2, 10, 0) == repeat(" ", 8) * alignNumSign(-1.2)


# function numToSups & numToSubs
ds = [superscriptNum, subscriptNum]
fs = [numToSups, numToSubs]
for (d,f) in zip(ds, fs)
    bl = true
    for i = 0:9
        bl *= (f(i) == d['0' + i]|>string)
    end
    @test bl
    num = rand(100000000:999999999)
    str = ""
    for i in num |> string
        str *= d[i]
    end
    @test str == f(num)
end
@test numToSups(nothing) == ""
@test numToSubs(nothing) == ""

end