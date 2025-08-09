#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _17
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
};

struct _77
{
    float4 _m0[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(256u, 1u, 1u);

kernel void main0(constant _17& _19 [[buffer(0)]], device _77& _79 [[buffer(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= _19._m1)
        {
            break;
        }
        uint _35 = 1u << _19._m0;
        uint _40 = gl_GlobalInvocationID.x & (~_35);
        uint _44 = gl_GlobalInvocationID.x | _35;
        if ((gl_GlobalInvocationID.x & _35) != 0u)
        {
            break;
        }
        uint _55 = _40 / 2u;
        uint _58 = _44 / 2u;
        bool _63 = (_40 & 1u) != 0u;
        bool _67 = (_44 & 1u) != 0u;
        float2 _188;
        if (_63)
        {
            _188 = float2(((device float*)&_79._m0[_55])[2u], ((device float*)&_79._m0[_55])[3u]);
        }
        else
        {
            _188 = float2(((device float*)&_79._m0[_55])[0u], ((device float*)&_79._m0[_55])[1u]);
        }
        float2 _189;
        if (_67)
        {
            _189 = float2(((device float*)&_79._m0[_58])[2u], ((device float*)&_79._m0[_58])[3u]);
        }
        else
        {
            _189 = float2(((device float*)&_79._m0[_58])[0u], ((device float*)&_79._m0[_58])[1u]);
        }
        float2 _123 = _188 + _189;
        float2 _124 = _123 * 0.707106769084930419921875;
        float2 _129 = _188 - _189;
        float2 _130 = _129 * 0.707106769084930419921875;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture);
        if (_63)
        {
            ((device float*)&_79._m0[_55])[2u] = _124.x;
            ((device float*)&_79._m0[_55])[3u] = _124.y;
        }
        else
        {
            ((device float*)&_79._m0[_55])[0u] = _124.x;
            ((device float*)&_79._m0[_55])[1u] = _124.y;
        }
        if (_67)
        {
            ((device float*)&_79._m0[_58])[2u] = _130.x;
            ((device float*)&_79._m0[_58])[3u] = _130.y;
        }
        else
        {
            ((device float*)&_79._m0[_58])[0u] = _130.x;
            ((device float*)&_79._m0[_58])[1u] = _130.y;
        }
        break;
    } while(false);
}

