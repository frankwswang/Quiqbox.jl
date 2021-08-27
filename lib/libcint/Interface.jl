function cintFunc!(libcintFunc::Val, 
                   buf::Array{Float64, N}, shls::Array{<:Signed, 1}, atm::Array{<:Signed, 1}, natm::Signed, 
                   bas::Array{<:Signed, 1}, nbas::Signed, env::Array{Float64, 1}, opt::Ptr=C_NULL) where {N}
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