function cintFunc!(libcintFunc::Val, 
                   buf::Array{Float64, N}, shls::Vector{<:Signed}, atm::Vector{<:Signed}, 
                   natm::Signed, bas::Vector{<:Signed}, nbas::Signed, env::Vector{Float64}, 
                   opt::Ptr=C_NULL) where {N}
    shls = shls .|> Cint
    atm = atm .|> Cint
    natm = natm |> Cint
    bas = bas .|> Cint
    nbas = nbas |> Cint
    bufAP = ArrayPointer(buf, showReminder=false)
    shlsAP = ArrayPointer(shls, showReminder=false)
    atmAP = ArrayPointer(atm, showReminder=false)
    basAP = ArrayPointer(bas, showReminder=false)
    envAP = ArrayPointer(env, showReminder=false)
    intPtrFunc!(libcintFunc, bufAP.ptr, shlsAP.ptr, atmAP.ptr, natm, basAP.ptr, nbas, envAP.ptr, opt)
    copyto!(buf, bufAP.arr)
    Libc.free(bufAP.ptr)
    Libc.free(shlsAP.ptr)
    Libc.free(atmAP.ptr)
    Libc.free(basAP.ptr)
    Libc.free(envAP.ptr)
    buf
end