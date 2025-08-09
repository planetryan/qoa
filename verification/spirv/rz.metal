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

struct _66
{
    float2 _m0[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(32u, 1u, 1u);

kernel void main0(constant _19& _21 [[buffer(0)]], device _66& _68 [[buffer(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= (1u << _21._m2))
        {
            break;
        }
        if ((gl_GlobalInvocationID.x & (1u << _21._m1)) != 0u)
        {
            float _54 = _21._m3 * (-0.5);
            float _55 = cos(_54);
            float _61 = sin(_54);
            _68._m0[gl_GlobalInvocationID.x] = float2((_68._m0[gl_GlobalInvocationID.x].x * _55) - (_68._m0[gl_GlobalInvocationID.x].y * _61), (_68._m0[gl_GlobalInvocationID.x].x * _61) + (_68._m0[gl_GlobalInvocationID.x].y * _55));
        }
        break;
    } while(false);
}

