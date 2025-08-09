#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _18
{
    uint _m0;
    uint _m1;
    uint _m2;
};

struct _86
{
    float2 _m0[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(32u, 1u, 1u);

kernel void main0(constant _18& _20 [[buffer(0)]], device _86& _88 [[buffer(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= (1u << _20._m2))
        {
            break;
        }
        uint _38 = 1u << _20._m0;
        uint _43 = 1u << _20._m1;
        bool _49 = (gl_GlobalInvocationID.x & _38) != 0u;
        bool _54 = (gl_GlobalInvocationID.x & _43) != 0u;
        if (_49 != _54)
        {
            uint _75 = (((gl_GlobalInvocationID.x & (~_38)) & (~_43)) | (_54 ? _38 : 0u)) | (_49 ? _43 : 0u);
            if (gl_GlobalInvocationID.x < _75)
            {
                float2 _92 = _88._m0[gl_GlobalInvocationID.x];
                _88._m0[gl_GlobalInvocationID.x] = _88._m0[_75];
                _88._m0[_75] = _92;
            }
        }
        break;
    } while(false);
}

