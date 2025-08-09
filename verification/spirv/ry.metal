#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _19
{
    uint _m0;
    uint _m1;
    uint _m2;
    float _m3;
};

struct _51
{
    float2 _m0[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(32u, 1u, 1u);

kernel void main0(constant _19& _21 [[buffer(0)]], device _51& _53 [[buffer(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= (1u << _21._m2))
        {
            break;
        }
        uint _41 = gl_GlobalInvocationID.x ^ (1u << _21._m1);
        if (gl_GlobalInvocationID.x < _41)
        {
            float2 _58 = _53._m0[gl_GlobalInvocationID.x];
            float2 _62 = _53._m0[_41];
            float _70 = _21._m3 * 0.5;
            float _71 = cos(_70);
            float _76 = sin(_70);
            _53._m0[gl_GlobalInvocationID.x] = float2((_71 * _58.x) - (_76 * _62.x), (_71 * _58.y) - (_76 * _62.y));
            _53._m0[_41] = float2((_71 * _62.x) + (_76 * _58.x), (_71 * _62.y) + (_76 * _58.y));
        }
        break;
    } while(false);
}

