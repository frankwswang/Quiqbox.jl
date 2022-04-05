using libcint_jll


function intPtrFunc!(::Val{:cint1e_ovlp_cart}, buf::Ptr{Cdouble}, shls::Ptr{Cint}, 
                     atm::Ptr{Cint}, natm::Cint, bas::Ptr{Cint}, nbas::Cint, 
                     env::Ptr{Cdouble}, opt::Ptr=C_NULL)
    ccall((:cint1e_ovlp_cart, libcint), Cvoid, 
          (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}),
          buf, shls, atm, natm, bas, nbas, env)
end

function intPtrFunc!(::Val{:cint1e_nuc_cart}, buf::Ptr{Cdouble}, shls::Ptr{Cint}, 
                     atm::Ptr{Cint}, natm::Cint, bas::Ptr{Cint}, nbas::Cint, 
                     env::Ptr{Cdouble}, opt::Ptr=C_NULL)
    ccall((:cint1e_nuc_cart, libcint), Cvoid, 
          (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}),
          buf, shls, atm, natm, bas, nbas, env)
end

function intPtrFunc!(::Val{:cint1e_kin_cart}, buf::Ptr{Cdouble}, shls::Ptr{Cint}, 
                     atm::Ptr{Cint}, natm::Cint, bas::Ptr{Cint}, nbas::Cint, 
                     env::Ptr{Cdouble}, opt::Ptr=C_NULL)
    ccall((:cint1e_kin_cart, libcint), Cvoid, 
          (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}),
          buf, shls, atm, natm, bas, nbas, env)
end

function intPtrFunc!(::Val{:cint2e_cart}, buf::Ptr{Cdouble}, shls::Ptr{Cint}, 
                     atm::Ptr{Cint}, natm::Cint, bas::Ptr{Cint}, nbas::Cint, 
                     env::Ptr{Cdouble}, opt::Ptr=C_NULL)
    ccall((:cint2e_cart, libcint), Cvoid, 
          (Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}, 
           Ptr{Cvoid}),
          buf, shls, atm, natm, bas, nbas, env, opt)
end