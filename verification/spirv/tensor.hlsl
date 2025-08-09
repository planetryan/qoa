struct _119
{
    float _m0;
    float _m1;
};

static const uint3 gl_WorkGroupSize = uint3(32u, 1u, 1u);

cbuffer _75_77 : register(b1)
{
    uint _77_m0 : packoffset(c0);
    uint _77_m1 : packoffset(c0.y);
    uint _77_m2 : packoffset(c0.z);
};

RWByteAddressBuffer _123 : register(u0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    do
    {
        if (gl_GlobalInvocationID.x >= uint(1 << int(_77_m2)))
        {
            break;
        }
        uint _95 = uint(1 << int(_77_m1));
        uint _102 = (gl_GlobalInvocationID.x >> _77_m1) & 1u;
        switch (_77_m0)
        {
            case 0u:
            {
                if (_102 == 0u)
                {
                    uint _117 = gl_GlobalInvocationID.x | _95;
                    _119 _127;
                    _127._m0 = asfloat(_123.Load(gl_GlobalInvocationID.x * 8 + 0));
                    _127._m1 = asfloat(_123.Load(gl_GlobalInvocationID.x * 8 + 4));
                    _119 _135;
                    _135._m0 = asfloat(_123.Load(_117 * 8 + 0));
                    _135._m1 = asfloat(_123.Load(_117 * 8 + 4));
                    _123.Store(gl_GlobalInvocationID.x * 8 + 0, asuint((_127._m0 + _135._m0) * 0.707106769084930419921875f));
                    _123.Store(gl_GlobalInvocationID.x * 8 + 4, asuint((_127._m1 + _135._m1) * 0.707106769084930419921875f));
                    _123.Store(_117 * 8 + 0, asuint((_127._m0 - _135._m0) * 0.707106769084930419921875f));
                    _123.Store(_117 * 8 + 4, asuint((_127._m1 - _135._m1) * 0.707106769084930419921875f));
                }
                break;
            }
            case 1u:
            {
                if (_102 == 0u)
                {
                    uint _185 = gl_GlobalInvocationID.x | _95;
                    _119 _189;
                    _189._m0 = asfloat(_123.Load(gl_GlobalInvocationID.x * 8 + 0));
                    _189._m1 = asfloat(_123.Load(gl_GlobalInvocationID.x * 8 + 4));
                    _119 _197;
                    _197._m0 = asfloat(_123.Load(_185 * 8 + 0));
                    _197._m1 = asfloat(_123.Load(_185 * 8 + 4));
                    _123.Store(gl_GlobalInvocationID.x * 8 + 0, asuint(_197._m0));
                    _123.Store(gl_GlobalInvocationID.x * 8 + 4, asuint(_197._m1));
                    _123.Store(_185 * 8 + 0, asuint(_189._m0));
                    _123.Store(_185 * 8 + 4, asuint(_189._m1));
                }
                break;
            }
            case 2u:
            {
                if (_102 == 1u)
                {
                    _123.Store(gl_GlobalInvocationID.x * 8 + 0, asuint(-asfloat(_123.Load(gl_GlobalInvocationID.x * 8 + 0))));
                    _123.Store(gl_GlobalInvocationID.x * 8 + 4, asuint(-asfloat(_123.Load(gl_GlobalInvocationID.x * 8 + 4))));
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

[numthreads(32, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
