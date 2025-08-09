static const uint3 gl_WorkGroupSize = uint3(32u, 1u, 1u);

cbuffer _18_20 : register(b1)
{
    uint _20_m0 : packoffset(c0);
    uint _20_m1 : packoffset(c0.y);
    uint _20_m2 : packoffset(c0.z);
};

RWByteAddressBuffer _88 : register(u0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    do
    {
        if (gl_GlobalInvocationID.x >= (1u << _20_m2))
        {
            break;
        }
        uint _38 = 1u << _20_m0;
        uint _43 = 1u << _20_m1;
        bool _49 = (gl_GlobalInvocationID.x & _38) != 0u;
        bool _54 = (gl_GlobalInvocationID.x & _43) != 0u;
        if (_49 != _54)
        {
            uint _75 = (((gl_GlobalInvocationID.x & (~_38)) & (~_43)) | (_54 ? _38 : 0u)) | (_49 ? _43 : 0u);
            if (gl_GlobalInvocationID.x < _75)
            {
                float2 _92 = asfloat(_88.Load2(gl_GlobalInvocationID.x * 8 + 0));
                _88.Store2(gl_GlobalInvocationID.x * 8 + 0, asuint(asfloat(_88.Load2(_75 * 8 + 0))));
                _88.Store2(_75 * 8 + 0, asuint(_92));
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
