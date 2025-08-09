#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct _75
{
    uint _m0;
    uint _m1;
    uint _m2;
};

struct _119
{
    float _m0;
    float _m1;
};

struct _121
{
    _119 _m0[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(32u, 1u, 1u);

kernel void main0(constant _75& _77 [[buffer(0)]], device _121& _123 [[buffer(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    do
    {
        if (gl_GlobalInvocationID.x >= uint(1 << int(_77._m2)))
        {
            break;
        }
        uint _95 = uint(1 << int(_77._m1));
        uint _102 = (gl_GlobalInvocationID.x >> _77._m1) & 1u;
        switch (_77._m0)
        {
            case 0u:
            {
                if (_102 == 0u)
                {
                    uint _117 = gl_GlobalInvocationID.x | _95;
                    _119 _127 = _123._m0[gl_GlobalInvocationID.x];
                    _119 _135 = _123._m0[_117];
                    _123._m0[gl_GlobalInvocationID.x]._m0 = (_127._m0 + _135._m0) * 0.707106769084930419921875;
                    _123._m0[gl_GlobalInvocationID.x]._m1 = (_127._m1 + _135._m1) * 0.707106769084930419921875;
                    _123._m0[_117]._m0 = (_127._m0 - _135._m0) * 0.707106769084930419921875;
                    _123._m0[_117]._m1 = (_127._m1 - _135._m1) * 0.707106769084930419921875;
                }
                break;
            }
            case 1u:
            {
                if (_102 == 0u)
                {
                    uint _185 = gl_GlobalInvocationID.x | _95;
                    _119 _189 = _123._m0[gl_GlobalInvocationID.x];
                    _119 _197 = _123._m0[_185];
                    _123._m0[gl_GlobalInvocationID.x]._m0 = _197._m0;
                    _123._m0[gl_GlobalInvocationID.x]._m1 = _197._m1;
                    _123._m0[_185]._m0 = _189._m0;
                    _123._m0[_185]._m1 = _189._m1;
                }
                break;
            }
            case 2u:
            {
                if (_102 == 1u)
                {
                    _123._m0[gl_GlobalInvocationID.x]._m0 = -_123._m0[gl_GlobalInvocationID.x]._m0;
                    _123._m0[gl_GlobalInvocationID.x]._m1 = -_123._m0[gl_GlobalInvocationID.x]._m1;
                }
                break;
            }
            default:
            {
                break;
            }
        }
        break;
    } while(false);
}

