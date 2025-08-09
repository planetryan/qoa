static const uint3 gl_WorkGroupSize = uint3(32u, 1u, 1u);

cbuffer _18_20 : register(b1)
{
    uint _20_m0 : packoffset(c0);
    uint _20_m1 : packoffset(c0.y);
    uint _20_m2 : packoffset(c0.z);
    uint _20_m3 : packoffset(c0.w);
};

RWByteAddressBuffer _53 : register(u0);

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
        uint _40 = gl_GlobalInvocationID.x ^ (1u << _20_m1);
        if (gl_GlobalInvocationID.x < _40)
        {
            float2 _58 = asfloat(_53.Load2(gl_GlobalInvocationID.x * 8 + 0));
            _53.Store2(gl_GlobalInvocationID.x * 8 + 0, asuint(asfloat(_53.Load2(_40 * 8 + 0))));
            _53.Store2(_40 * 8 + 0, asuint(_58));
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
