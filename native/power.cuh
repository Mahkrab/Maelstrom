// Positive finite powf adapted from Arm optimized-routines, revision
// f67e7daa5457646fad9f75580263dba654c988aa (math/powf.c,
// math/powf_log2_data.c and math/exp2f_data.c). Binary64 intermediates reproduce
// the host library's approximation, including its binary32 rounding boundaries.
// Copyright (c) 2017-2026, Arm Limited. SPDX-License-Identifier: MIT
// See power-LICENSE and docs/cuda.md for provenance and the supported domain.
#pragma once

__device__ __constant__ double power_log_table[16][2] = {
 {0x1.661ec79f8f3bep+0,-0x1.efec65b963019p-2},
 {0x1.571ed4aaf883dp+0,-0x1.b0b6832d4fca4p-2},
 {0x1.49539f0f010bp+0,-0x1.7418b0a1fb77bp-2},
 {0x1.3c995b0b80385p+0,-0x1.39de91a6dcf7bp-2},
 {0x1.30d190c8864a5p+0,-0x1.01d9bf3f2b631p-2},
 {0x1.25e227b0b8eap+0,-0x1.97c1d1b3b7afp-3},
 {0x1.1bb4a4a1a343fp+0,-0x1.2f9e393af3c9fp-3},
 {0x1.12358f08ae5bap+0,-0x1.960cbbf788d5cp-4},
 {0x1.0953f419900a7p+0,-0x1.a6f9db6475fcep-5},
 {0x1p+0,0x0p+0},
 {0x1.e608cfd9a47acp-1,0x1.338ca9f24f53dp-4},
 {0x1.ca4b31f026aap-1,0x1.476a9543891bap-3},
 {0x1.b2036576afce6p-1,0x1.e840b4ac4e4d2p-3},
 {0x1.9c2d163a1aa2dp-1,0x1.40645f0c6651cp-2},
 {0x1.886e6037841edp-1,0x1.88e9c2c1b9ff8p-2},
 {0x1.767dcf5534862p-1,0x1.ce0a44eb17bccp-2}
};
__device__ __constant__ unsigned long long power_exp_table[32] = {
 0x3ff0000000000000,0x3fefd9b0d3158574,0x3fefb5586cf9890f,0x3fef9301d0125b51,
 0x3fef72b83c7d517b,0x3fef54873168b9aa,0x3fef387a6e756238,0x3fef1e9df51fdee1,
 0x3fef06fe0a31b715,0x3feef1a7373aa9cb,0x3feedea64c123422,0x3feece086061892d,
 0x3feebfdad5362a27,0x3feeb42b569d4f82,0x3feeab07dd485429,0x3feea47eb03a5585,
 0x3feea09e667f3bcd,0x3fee9f75e8ec5f74,0x3feea11473eb0187,0x3feea589994cce13,
 0x3feeace5422aa0db,0x3feeb737b0cdc5e5,0x3feec49182a3f090,0x3feed503b23e255d,
 0x3feee89f995ad3ad,0x3feeff76f2fb5e47,0x3fef199bdd85529c,0x3fef3720dcef9069,
 0x3fef5818dcfba487,0x3fef7c97337b9b5f,0x3fefa4afa2a490da,0x3fefd0765b6e4540
};

__device__ float ordered_power_reference(float x,float exponent) {
    if (x==0 || !isfinite(x)) return x;
    auto ix=__float_as_uint(x);
    if (ix<0x00800000) { ix=__float_as_uint(x*0x1p23f); ix-=23u<<23; }
    unsigned int tmp=ix-0x3f330000, index=(tmp>>19)%16, top=tmp&0xff800000;
    int k=static_cast<int>(top)>>23;
    double z=static_cast<double>(__uint_as_float(ix-top));
    double r=z*power_log_table[index][0]-1;
    double y0=power_log_table[index][1]+static_cast<double>(k);
    double r2=r*r, r4=r2*r2;
    double y=0x1.27616c9496e0bp-2*r-0x1.71969a075c67ap-2;
    double p=0x1.ec70a6ca7baddp-2*r-0x1.7154748bef6c8p-1;
    double q=0x1.71547652ab82bp0*r+y0;
    q=p*r2+q;
    y=y*r4+q;
    double xd=static_cast<double>(exponent)*y;
    if (xd<=-150) return 0;
    if (xd>0x1.fffffffd1d571p+6) return __int_as_float(0x7f800000);
    double kd=xd+0x1.8p+47;
    auto ki=static_cast<unsigned long long>(__double_as_longlong(kd));
    kd-=0x1.8p+47;
    r=xd-kd;
    auto t=power_exp_table[ki%32]+(ki<<47);
    double scale=__longlong_as_double(static_cast<long long>(t));
    z=0x1.c6af84b912394p-5*r+0x1.ebfce50fac4f3p-3;
    r2=r*r;
    y=0x1.62e42ff0c52d6p-1*r+1;
    y=z*r2+y;
    return static_cast<float>(y*scale);
}

// Only accept the quartic away from binary32 rounding boundaries. The retained
// approximation handles the remaining inputs and all other exponents. The
// complete non-negative binary32 domain is checked by power_exhaustive.cu.
__device__ bool try_power_four(float x, float &result) {
    if (!(x>0) || !isfinite(x)) return false;
    double square=static_cast<double>(x)*static_cast<double>(x);
    double fourth=square*square;
    result=static_cast<float>(fourth);
    const auto bits=__float_as_uint(result);
    if (bits<=0x00800000U || bits>=0x7f7fffffU) return false;
    // Binary64 has 29 more fraction bits. Bit 28 is the binary32
    // rounding midpoint; retain the reference near either side of it.
    const auto tail=static_cast<unsigned long long>(__double_as_longlong(fourth))&0x1fffffffULL;
    const auto distance=tail>0x10000000ULL ? tail-0x10000000ULL : 0x10000000ULL-tail;
    return distance>0x200000ULL;
}

__device__ float ordered_power(float x,float exponent) {
    float result;
    if (exponent==4.0f && try_power_four(x,result)) return result;
    return ordered_power_reference(x,exponent);
}
