; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 179
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID %params %_
               OpExecutionMode %main LocalSize 256 1 1
               OpSource GLSL 450
               OpName %main "main"
               OpName %idx "idx"
               OpName %gl_GlobalInvocationID "gl_GlobalInvocationID"
               OpName %PushConstants "PushConstants"
               OpMemberName %PushConstants 0 "targetQubit"
               OpMemberName %PushConstants 1 "stateSize"
               OpMemberName %PushConstants 2 "reserved1"
               OpMemberName %PushConstants 3 "reserved2"
               OpName %params "params"
               OpName %mask "mask"
               OpName %idx0 "idx0"
               OpName %idx1 "idx1"
               OpName %vec_idx0 "vec_idx0"
               OpName %vec_idx1 "vec_idx1"
               OpName %odd0 "odd0"
               OpName %odd1 "odd1"
               OpName %psi0 "psi0"
               OpName %QuantumState "QuantumState"
               OpMemberName %QuantumState 0 "amplitudes"
               OpName %_ ""
               OpName %psi1 "psi1"
               OpName %inv_sqrt2 "inv_sqrt2"
               OpName %new_psi0 "new_psi0"
               OpName %new_psi1 "new_psi1"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %PushConstants Block
               OpMemberDecorate %PushConstants 0 Offset 0
               OpMemberDecorate %PushConstants 1 Offset 4
               OpMemberDecorate %PushConstants 2 Offset 8
               OpMemberDecorate %PushConstants 3 Offset 12
               OpDecorate %_runtimearr_v4float ArrayStride 16
               OpDecorate %QuantumState Block
               OpMemberDecorate %QuantumState 0 Offset 0
               OpDecorate %_ Binding 0
               OpDecorate %_ DescriptorSet 0
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %uint_0 = OpConstant %uint 0
%_ptr_Input_uint = OpTypePointer Input %uint
%PushConstants = OpTypeStruct %uint %uint %uint %uint
%_ptr_PushConstant_PushConstants = OpTypePointer PushConstant %PushConstants
     %params = OpVariable %_ptr_PushConstant_PushConstants PushConstant
        %int = OpTypeInt 32 1
      %int_1 = OpConstant %int 1
%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
       %bool = OpTypeBool
     %uint_1 = OpConstant %uint 1
      %int_0 = OpConstant %int 0
     %uint_2 = OpConstant %uint 2
%_ptr_Function_bool = OpTypePointer Function %bool
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
%_ptr_Function_v2float = OpTypePointer Function %v2float
    %v4float = OpTypeVector %float 4
%_runtimearr_v4float = OpTypeRuntimeArray %v4float
%QuantumState = OpTypeStruct %_runtimearr_v4float
%_ptr_StorageBuffer_QuantumState = OpTypePointer StorageBuffer %QuantumState
          %_ = OpVariable %_ptr_StorageBuffer_QuantumState StorageBuffer
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
     %uint_3 = OpConstant %uint 3
%_ptr_Function_float = OpTypePointer Function %float
%float_0_707106769 = OpConstant %float 0.707106769
    %uint_72 = OpConstant %uint 72
   %uint_264 = OpConstant %uint 264
   %uint_256 = OpConstant %uint 256
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_256 %uint_1 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
        %idx = OpVariable %_ptr_Function_uint Function
       %mask = OpVariable %_ptr_Function_uint Function
       %idx0 = OpVariable %_ptr_Function_uint Function
       %idx1 = OpVariable %_ptr_Function_uint Function
   %vec_idx0 = OpVariable %_ptr_Function_uint Function
   %vec_idx1 = OpVariable %_ptr_Function_uint Function
       %odd0 = OpVariable %_ptr_Function_bool Function
       %odd1 = OpVariable %_ptr_Function_bool Function
       %psi0 = OpVariable %_ptr_Function_v2float Function
         %73 = OpVariable %_ptr_Function_v2float Function
       %psi1 = OpVariable %_ptr_Function_v2float Function
        %101 = OpVariable %_ptr_Function_v2float Function
  %inv_sqrt2 = OpVariable %_ptr_Function_float Function
   %new_psi0 = OpVariable %_ptr_Function_v2float Function
   %new_psi1 = OpVariable %_ptr_Function_v2float Function
         %14 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
         %15 = OpLoad %uint %14
               OpStore %idx %15
         %16 = OpLoad %uint %idx
         %23 = OpAccessChain %_ptr_PushConstant_uint %params %int_1
         %24 = OpLoad %uint %23
         %26 = OpUGreaterThanEqual %bool %16 %24
               OpSelectionMerge %28 None
               OpBranchConditional %26 %27 %28
         %27 = OpLabel
               OpReturn
         %28 = OpLabel
         %33 = OpAccessChain %_ptr_PushConstant_uint %params %int_0
         %34 = OpLoad %uint %33
         %35 = OpShiftLeftLogical %uint %uint_1 %34
               OpStore %mask %35
         %37 = OpLoad %uint %idx
         %38 = OpLoad %uint %mask
         %39 = OpNot %uint %38
         %40 = OpBitwiseAnd %uint %37 %39
               OpStore %idx0 %40
         %42 = OpLoad %uint %idx
         %43 = OpLoad %uint %mask
         %44 = OpBitwiseOr %uint %42 %43
               OpStore %idx1 %44
         %45 = OpLoad %uint %idx
         %46 = OpLoad %uint %mask
         %47 = OpBitwiseAnd %uint %45 %46
         %48 = OpINotEqual %bool %47 %uint_0
               OpSelectionMerge %50 None
               OpBranchConditional %48 %49 %50
         %49 = OpLabel
               OpReturn
         %50 = OpLabel
         %53 = OpLoad %uint %idx0
         %55 = OpUDiv %uint %53 %uint_2
               OpStore %vec_idx0 %55
         %57 = OpLoad %uint %idx1
         %58 = OpUDiv %uint %57 %uint_2
               OpStore %vec_idx1 %58
         %61 = OpLoad %uint %idx0
         %62 = OpBitwiseAnd %uint %61 %uint_1
         %63 = OpINotEqual %bool %62 %uint_0
               OpStore %odd0 %63
         %65 = OpLoad %uint %idx1
         %66 = OpBitwiseAnd %uint %65 %uint_1
         %67 = OpINotEqual %bool %66 %uint_0
               OpStore %odd1 %67
         %72 = OpLoad %bool %odd0
               OpSelectionMerge %75 None
               OpBranchConditional %72 %74 %90
         %74 = OpLabel
         %81 = OpLoad %uint %vec_idx0
         %83 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %81 %uint_2
         %84 = OpLoad %float %83
         %85 = OpLoad %uint %vec_idx0
         %87 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %85 %uint_3
         %88 = OpLoad %float %87
         %89 = OpCompositeConstruct %v2float %84 %88
               OpStore %73 %89
               OpBranch %75
         %90 = OpLabel
         %91 = OpLoad %uint %vec_idx0
         %92 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %91 %uint_0
         %93 = OpLoad %float %92
         %94 = OpLoad %uint %vec_idx0
         %95 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %94 %uint_1
         %96 = OpLoad %float %95
         %97 = OpCompositeConstruct %v2float %93 %96
               OpStore %73 %97
               OpBranch %75
         %75 = OpLabel
         %98 = OpLoad %v2float %73
               OpStore %psi0 %98
        %100 = OpLoad %bool %odd1
               OpSelectionMerge %103 None
               OpBranchConditional %100 %102 %111
        %102 = OpLabel
        %104 = OpLoad %uint %vec_idx1
        %105 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %104 %uint_2
        %106 = OpLoad %float %105
        %107 = OpLoad %uint %vec_idx1
        %108 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %107 %uint_3
        %109 = OpLoad %float %108
        %110 = OpCompositeConstruct %v2float %106 %109
               OpStore %101 %110
               OpBranch %103
        %111 = OpLabel
        %112 = OpLoad %uint %vec_idx1
        %113 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %112 %uint_0
        %114 = OpLoad %float %113
        %115 = OpLoad %uint %vec_idx1
        %116 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %115 %uint_1
        %117 = OpLoad %float %116
        %118 = OpCompositeConstruct %v2float %114 %117
               OpStore %101 %118
               OpBranch %103
        %103 = OpLabel
        %119 = OpLoad %v2float %101
               OpStore %psi1 %119
               OpStore %inv_sqrt2 %float_0_707106769
        %124 = OpLoad %float %inv_sqrt2
        %125 = OpLoad %v2float %psi0
        %126 = OpLoad %v2float %psi1
        %127 = OpFAdd %v2float %125 %126
        %128 = OpVectorTimesScalar %v2float %127 %124
               OpStore %new_psi0 %128
        %130 = OpLoad %float %inv_sqrt2
        %131 = OpLoad %v2float %psi0
        %132 = OpLoad %v2float %psi1
        %133 = OpFSub %v2float %131 %132
        %134 = OpVectorTimesScalar %v2float %133 %130
               OpStore %new_psi1 %134
               OpMemoryBarrier %uint_1 %uint_72
               OpControlBarrier %uint_2 %uint_2 %uint_264
        %137 = OpLoad %bool %odd0
               OpSelectionMerge %139 None
               OpBranchConditional %137 %138 %148
        %138 = OpLabel
        %140 = OpLoad %uint %vec_idx0
        %141 = OpAccessChain %_ptr_Function_float %new_psi0 %uint_0
        %142 = OpLoad %float %141
        %143 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %140 %uint_2
               OpStore %143 %142
        %144 = OpLoad %uint %vec_idx0
        %145 = OpAccessChain %_ptr_Function_float %new_psi0 %uint_1
        %146 = OpLoad %float %145
        %147 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %144 %uint_3
               OpStore %147 %146
               OpBranch %139
        %148 = OpLabel
        %149 = OpLoad %uint %vec_idx0
        %150 = OpAccessChain %_ptr_Function_float %new_psi0 %uint_0
        %151 = OpLoad %float %150
        %152 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %149 %uint_0
               OpStore %152 %151
        %153 = OpLoad %uint %vec_idx0
        %154 = OpAccessChain %_ptr_Function_float %new_psi0 %uint_1
        %155 = OpLoad %float %154
        %156 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %153 %uint_1
               OpStore %156 %155
               OpBranch %139
        %139 = OpLabel
        %157 = OpLoad %bool %odd1
               OpSelectionMerge %159 None
               OpBranchConditional %157 %158 %168
        %158 = OpLabel
        %160 = OpLoad %uint %vec_idx1
        %161 = OpAccessChain %_ptr_Function_float %new_psi1 %uint_0
        %162 = OpLoad %float %161
        %163 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %160 %uint_2
               OpStore %163 %162
        %164 = OpLoad %uint %vec_idx1
        %165 = OpAccessChain %_ptr_Function_float %new_psi1 %uint_1
        %166 = OpLoad %float %165
        %167 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %164 %uint_3
               OpStore %167 %166
               OpBranch %159
        %168 = OpLabel
        %169 = OpLoad %uint %vec_idx1
        %170 = OpAccessChain %_ptr_Function_float %new_psi1 %uint_0
        %171 = OpLoad %float %170
        %172 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %169 %uint_0
               OpStore %172 %171
        %173 = OpLoad %uint %vec_idx1
        %174 = OpAccessChain %_ptr_Function_float %new_psi1 %uint_1
        %175 = OpLoad %float %174
        %176 = OpAccessChain %_ptr_StorageBuffer_float %_ %int_0 %173 %uint_1
               OpStore %176 %175
               OpBranch %159
        %159 = OpLabel
               OpReturn
               OpFunctionEnd
