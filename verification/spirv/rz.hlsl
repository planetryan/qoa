static const uint3 gl_WorkGroupSize = uint3(32u, 1u, 1u);

cbuffer _19_21 : register(b1)
{
    uint _21_m0 : packoffset(c0);
    uint _21_m1 : packoffset(c0.y);
    uint _21_m2 : packoffset(c0.z);
    float _21_m3 : packoffset(c0.w);
};

RWByteAddressBuffer _68 : register(u0);

static uint3 gl_GlobalInvocationID;
struct SPIRV_Cross_Input
{
    uint3 gl_GlobalInvocationID : SV_DispatchThreadID;
};

void comp_main()
{
    do
    {
        if (gl_GlobalInvocationID.x >= (1u << _21_m2))
        {
            break;
        }
        if ((gl_GlobalInvocationID.x & (1u << _21_m1)) != 0u)
        {
            float _54 = _21_m3 * (-0.5f);
            float _55 = cos(_54);
            float _61 = sin(_54);
            _68.Store2(gl_GlobalInvocationID.x * 8 + 0, asuint(float2((asfloat(_68.Load2(gl_GlobalInvocationID.x * 8 + 0)).x * _55) - (asfloat(_68.Load2(gl_GlobalInvocationID.x * 8 + 0)).y * _61), (asfloat(_68.Load2(gl_GlobalInvocationID.x * 8 + 0)).x * _61) + (asfloat(_68.Load2(gl_GlobalInvocationID.x * 8 + 0)).y * _55))));
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
