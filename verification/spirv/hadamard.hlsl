static const uint3 gl_WorkGroupSize = uint3(256u, 1u, 1u);

RWByteAddressBuffer _79 : register(u0);
cbuffer _17_19
{
    uint _19_m0 : packoffset(c0);
    uint _19_m1 : packoffset(c0.y);
    uint _19_m2 : packoffset(c0.z);
    uint _19_m3 : packoffset(c0.w);
};


static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    do
    {
        if (gl_GlobalInvocationID.x >= _19_m1)
        {
            break;
        }
        uint _35 = 1u << _19_m0;
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
            _188 = float2(asfloat(_79.Load(_55 * 16 + 8)), asfloat(_79.Load(_55 * 16 + 12)));
        }
        else
        {
            _188 = float2(asfloat(_79.Load(_55 * 16 + 0)), asfloat(_79.Load(_55 * 16 + 4)));
        }
        float2 _189;
        if (_67)
        {
            _189 = float2(asfloat(_79.Load(_58 * 16 + 8)), asfloat(_79.Load(_58 * 16 + 12)));
        }
        else
        {
            _189 = float2(asfloat(_79.Load(_58 * 16 + 0)), asfloat(_79.Load(_58 * 16 + 4)));
        }
        float2 _123 = _188 + _189;
        float2 _124 = _123 * 0.707106769084930419921875f;
        float2 _129 = _188 - _189;
        float2 _130 = _129 * 0.707106769084930419921875f;
        AllMemoryBarrier();
        if (_63)
        {
            _79.Store(_55 * 16 + 8, asuint(_124.x));
            _79.Store(_55 * 16 + 12, asuint(_124.y));
        }
        else
        {
            _79.Store(_55 * 16 + 0, asuint(_124.x));
            _79.Store(_55 * 16 + 4, asuint(_124.y));
        }
        if (_67)
        {
            _79.Store(_58 * 16 + 8, asuint(_130.x));
            _79.Store(_58 * 16 + 12, asuint(_130.y));
        }
        else
        {
            _79.Store(_58 * 16 + 0, asuint(_130.x));
            _79.Store(_58 * 16 + 4, asuint(_130.y));
        }
        break;
    } while(false);
}

[numthreads(256, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_GlobalInvocationID = stage_input.gl_GlobalInvocationID;
    comp_main();
}
