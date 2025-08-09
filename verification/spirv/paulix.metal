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

struct _51
{
    float2 _m0[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(32u, 1u, 1u);

kernel void main0(constant _18& _20 [[buffer(0)]], device _51& _53 [[buffer(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= (1u << _20._m2))
        {
            break;
        }
        uint _40 = gl_GlobalInvocationID.x ^ (1u << _20._m1);
        if (gl_GlobalInvocationID.x < _40)
        {
            float2 _58 = _53._m0[gl_GlobalInvocationID.x];
            _53._m0[gl_GlobalInvocationID.x] = _53._m0[_40];
            _53._m0[_40] = _58;
        }
        break;
    } while(false);
}

