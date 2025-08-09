#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _18
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
};

struct _64
{
    float2 _m0[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(32u, 1u, 1u);

kernel void main0(constant _18& _20 [[buffer(0)]], device _64& _66 [[buffer(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= (1u << _20._m3))
        {
            break;
        }
        if ((gl_GlobalInvocationID.x & (1u << _20._m1)) != 0u)
        {
            uint _53 = gl_GlobalInvocationID.x ^ (1u << _20._m2);
            if (gl_GlobalInvocationID.x < _53)
            {
                float2 _71 = _66._m0[gl_GlobalInvocationID.x];
                _66._m0[gl_GlobalInvocationID.x] = _66._m0[_53];
                _66._m0[_53] = _71;
            }
        }
        break;
    } while(false);
}

