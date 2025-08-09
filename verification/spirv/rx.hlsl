static const uint3 gl_WorkGroupSize = uint3(32u, 1u, 1u);

cbuffer _19_21 : register(b1)
{
    uint _21_m0 : packoffset(c0);
    uint _21_m1 : packoffset(c0.y);
    uint _21_m2 : packoffset(c0.z);
    float _21_m3 : packoffset(c0.w);
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
        if (gl_GlobalInvocationID.x >= (1u << _21_m2))
        {
            break;
        }
        uint _41 = gl_GlobalInvocationID.x ^ (1u << _21_m1);
        if (gl_GlobalInvocationID.x < _41)
        {
            float2 _58 = asfloat(_53.Load2(gl_GlobalInvocationID.x * 8 + 0));
            float2 _62 = asfloat(_53.Load2(_41 * 8 + 0));
            float _70 = _21_m3 * 0.5f;
            float _71 = cos(_70);
            float _76 = sin(_70);
            _53.Store2(gl_GlobalInvocationID.x * 8 + 0, asuint(float2((_71 * _58.x) - (_76 * _62.y), (_71 * _58.y) + (_76 * _62.x))));
            _53.Store2(_41 * 8 + 0, asuint(float2((_71 * _62.x) - (_76 * _58.y), (_71 * _62.y) + (_76 * _58.x))));
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
