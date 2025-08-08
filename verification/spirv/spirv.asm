; SPIR-V
; Version: 1.0
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 238
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 32 1 1
               OpSource GLSL 450
               OpSourceExtension "GL_KHR_shader_subgroup_arithmetic"
               OpSourceExtension "GL_KHR_shader_subgroup_ballot"
               OpSourceExtension "GL_KHR_shader_subgroup_basic"
               OpSourceExtension "GL_KHR_shader_subgroup_clustered"
               OpSourceExtension "GL_KHR_shader_subgroup_quad"
               OpName %main "main"
               OpName %Complex "Complex"
               OpMemberName %Complex 0 "real"
               OpMemberName %Complex 1 "imag"
               OpName %complex_add_struct_Complex_f1_f11_struct_Complex_f1_f11_ "complex_add(struct-Complex-f1-f11;struct-Complex-f1-f11;"
               OpName %a "a"
               OpName %b "b"
               OpName %complex_sub_struct_Complex_f1_f11_struct_Complex_f1_f11_ "complex_sub(struct-Complex-f1-f11;struct-Complex-f1-f11;"
               OpName %a_0 "a"
               OpName %b_0 "b"
               OpName %complex_scale_struct_Complex_f1_f11_f1_ "complex_scale(struct-Complex-f1-f11;f1;"
               OpName %a_1 "a"
               OpName %s "s"
               OpName %i "i"
               OpName %gl_GlobalInvocationID "gl_GlobalInvocationID"
               OpName %total_states "total_states"
               OpName %GateParams "GateParams"
               OpMemberName %GateParams 0 "gate_type"
               OpMemberName %GateParams 1 "target_qubit"
               OpMemberName %GateParams 2 "num_qubits"
               OpName %gate_params "gate_params"
               OpName %stride "stride"
               OpName %target_bit "target_bit"
               OpName %j "j"
               OpName %a_2 "a"
               OpName %Complex_0 "Complex"
               OpMemberName %Complex_0 0 "real"
               OpMemberName %Complex_0 1 "imag"
               OpName %StateVector "StateVector"
               OpMemberName %StateVector 0 "state"
               OpName %state_vector "state_vector"
               OpName %b_1 "b"
               OpName %inv_sqrt2 "inv_sqrt2"
               OpName %new_a "new_a"
               OpName %param "param"
               OpName %param_0 "param"
               OpName %param_1 "param"
               OpName %param_2 "param"
               OpName %new_b "new_b"
               OpName %param_3 "param"
               OpName %param_4 "param"
               OpName %param_5 "param"
               OpName %param_6 "param"
               OpName %j_0 "j"
               OpName %temp_i "temp_i"
               OpName %temp_j "temp_j"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %GateParams Block
               OpMemberDecorate %GateParams 0 Offset 0
               OpMemberDecorate %GateParams 1 Offset 4
               OpMemberDecorate %GateParams 2 Offset 8
               OpDecorate %gate_params Binding 1
               OpDecorate %gate_params DescriptorSet 0
               OpMemberDecorate %Complex_0 0 Offset 0
               OpMemberDecorate %Complex_0 1 Offset 4
               OpDecorate %_runtimearr_Complex_0 ArrayStride 8
               OpDecorate %StateVector BufferBlock
               OpMemberDecorate %StateVector 0 Offset 0
               OpDecorate %state_vector Binding 0
               OpDecorate %state_vector DescriptorSet 0
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %Complex = OpTypeStruct %float %float
%_ptr_Function_Complex = OpTypePointer Function %Complex
          %9 = OpTypeFunction %Complex %_ptr_Function_Complex %_ptr_Function_Complex
%_ptr_Function_float = OpTypePointer Function %float
         %19 = OpTypeFunction %Complex %_ptr_Function_Complex %_ptr_Function_float
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
       %uint = OpTypeInt 32 0
%_ptr_Function_uint = OpTypePointer Function %uint
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %uint_0 = OpConstant %uint 0
%_ptr_Input_uint = OpTypePointer Input %uint
 %GateParams = OpTypeStruct %uint %uint %uint
%_ptr_Uniform_GateParams = OpTypePointer Uniform %GateParams
%gate_params = OpVariable %_ptr_Uniform_GateParams Uniform
      %int_2 = OpConstant %int 2
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
       %bool = OpTypeBool
     %uint_1 = OpConstant %uint 1
  %Complex_0 = OpTypeStruct %float %float
%_runtimearr_Complex_0 = OpTypeRuntimeArray %Complex_0
%StateVector = OpTypeStruct %_runtimearr_Complex_0
%_ptr_Uniform_StateVector = OpTypePointer Uniform %StateVector
%state_vector = OpVariable %_ptr_Uniform_StateVector Uniform
%_ptr_Uniform_Complex_0 = OpTypePointer Uniform %Complex_0
%float_0_707106769 = OpConstant %float 0.707106769
%_ptr_Uniform_float = OpTypePointer Uniform %float
    %uint_32 = OpConstant %uint 32
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_32 %uint_1 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
          %i = OpVariable %_ptr_Function_uint Function
%total_states = OpVariable %_ptr_Function_uint Function
     %stride = OpVariable %_ptr_Function_uint Function
 %target_bit = OpVariable %_ptr_Function_uint Function
          %j = OpVariable %_ptr_Function_uint Function
        %a_2 = OpVariable %_ptr_Function_Complex Function
        %b_1 = OpVariable %_ptr_Function_Complex Function
  %inv_sqrt2 = OpVariable %_ptr_Function_float Function
      %new_a = OpVariable %_ptr_Function_Complex Function
      %param = OpVariable %_ptr_Function_Complex Function
    %param_0 = OpVariable %_ptr_Function_Complex Function
    %param_1 = OpVariable %_ptr_Function_Complex Function
    %param_2 = OpVariable %_ptr_Function_float Function
      %new_b = OpVariable %_ptr_Function_Complex Function
    %param_3 = OpVariable %_ptr_Function_Complex Function
    %param_4 = OpVariable %_ptr_Function_Complex Function
    %param_5 = OpVariable %_ptr_Function_Complex Function
    %param_6 = OpVariable %_ptr_Function_float Function
        %j_0 = OpVariable %_ptr_Function_uint Function
     %temp_i = OpVariable %_ptr_Function_Complex Function
     %temp_j = OpVariable %_ptr_Function_Complex Function
         %72 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
         %73 = OpLoad %uint %72
               OpStore %i %73
         %80 = OpAccessChain %_ptr_Uniform_uint %gate_params %int_2
         %81 = OpLoad %uint %80
         %82 = OpShiftLeftLogical %int %int_1 %81
         %83 = OpBitcast %uint %82
               OpStore %total_states %83
         %84 = OpLoad %uint %i
         %85 = OpLoad %uint %total_states
         %87 = OpUGreaterThanEqual %bool %84 %85
               OpSelectionMerge %89 None
               OpBranchConditional %87 %88 %89
         %88 = OpLabel
               OpReturn
         %89 = OpLabel
         %92 = OpAccessChain %_ptr_Uniform_uint %gate_params %int_1
         %93 = OpLoad %uint %92
         %94 = OpShiftLeftLogical %int %int_1 %93
         %95 = OpBitcast %uint %94
               OpStore %stride %95
         %97 = OpLoad %uint %i
         %98 = OpAccessChain %_ptr_Uniform_uint %gate_params %int_1
         %99 = OpLoad %uint %98
        %100 = OpShiftRightLogical %uint %97 %99
        %102 = OpBitwiseAnd %uint %100 %uint_1
               OpStore %target_bit %102
        %103 = OpAccessChain %_ptr_Uniform_uint %gate_params %int_0
        %104 = OpLoad %uint %103
               OpSelectionMerge %109 None
               OpSwitch %104 %108 0 %105 1 %106 2 %107
        %108 = OpLabel
               OpBranch %109
        %105 = OpLabel
        %110 = OpLoad %uint %target_bit
        %111 = OpIEqual %bool %110 %uint_0
               OpSelectionMerge %113 None
               OpBranchConditional %111 %112 %113
        %112 = OpLabel
        %115 = OpLoad %uint %i
        %116 = OpLoad %uint %stride
        %117 = OpBitwiseOr %uint %115 %116
               OpStore %j %117
        %124 = OpLoad %uint %i
        %126 = OpAccessChain %_ptr_Uniform_Complex_0 %state_vector %int_0 %124
        %127 = OpLoad %Complex_0 %126
        %128 = OpCompositeExtract %float %127 0
        %129 = OpAccessChain %_ptr_Function_float %a_2 %int_0
               OpStore %129 %128
        %130 = OpCompositeExtract %float %127 1
        %131 = OpAccessChain %_ptr_Function_float %a_2 %int_1
               OpStore %131 %130
        %133 = OpLoad %uint %j
        %134 = OpAccessChain %_ptr_Uniform_Complex_0 %state_vector %int_0 %133
        %135 = OpLoad %Complex_0 %134
        %136 = OpCompositeExtract %float %135 0
        %137 = OpAccessChain %_ptr_Function_float %b_1 %int_0
               OpStore %137 %136
        %138 = OpCompositeExtract %float %135 1
        %139 = OpAccessChain %_ptr_Function_float %b_1 %int_1
               OpStore %139 %138
               OpStore %inv_sqrt2 %float_0_707106769
        %144 = OpLoad %Complex %a_2
               OpStore %param %144
        %146 = OpLoad %Complex %b_1
               OpStore %param_0 %146
        %147 = OpFunctionCall %Complex %complex_add_struct_Complex_f1_f11_struct_Complex_f1_f11_ %param %param_0
               OpStore %param_1 %147
        %150 = OpLoad %float %inv_sqrt2
               OpStore %param_2 %150
        %151 = OpFunctionCall %Complex %complex_scale_struct_Complex_f1_f11_f1_ %param_1 %param_2
               OpStore %new_a %151
        %154 = OpLoad %Complex %a_2
               OpStore %param_3 %154
        %156 = OpLoad %Complex %b_1
               OpStore %param_4 %156
        %157 = OpFunctionCall %Complex %complex_sub_struct_Complex_f1_f11_struct_Complex_f1_f11_ %param_3 %param_4
               OpStore %param_5 %157
        %160 = OpLoad %float %inv_sqrt2
               OpStore %param_6 %160
        %161 = OpFunctionCall %Complex %complex_scale_struct_Complex_f1_f11_f1_ %param_5 %param_6
               OpStore %new_b %161
        %162 = OpLoad %uint %i
        %163 = OpLoad %Complex %new_a
        %164 = OpAccessChain %_ptr_Uniform_Complex_0 %state_vector %int_0 %162
        %165 = OpCompositeExtract %float %163 0
        %167 = OpAccessChain %_ptr_Uniform_float %164 %int_0
               OpStore %167 %165
        %168 = OpCompositeExtract %float %163 1
        %169 = OpAccessChain %_ptr_Uniform_float %164 %int_1
               OpStore %169 %168
        %170 = OpLoad %uint %j
        %171 = OpLoad %Complex %new_b
        %172 = OpAccessChain %_ptr_Uniform_Complex_0 %state_vector %int_0 %170
        %173 = OpCompositeExtract %float %171 0
        %174 = OpAccessChain %_ptr_Uniform_float %172 %int_0
               OpStore %174 %173
        %175 = OpCompositeExtract %float %171 1
        %176 = OpAccessChain %_ptr_Uniform_float %172 %int_1
               OpStore %176 %175
               OpBranch %113
        %113 = OpLabel
               OpBranch %109
        %106 = OpLabel
        %178 = OpLoad %uint %target_bit
        %179 = OpIEqual %bool %178 %uint_0
               OpSelectionMerge %181 None
               OpBranchConditional %179 %180 %181
        %180 = OpLabel
        %183 = OpLoad %uint %i
        %184 = OpLoad %uint %stride
        %185 = OpBitwiseOr %uint %183 %184
               OpStore %j_0 %185
        %187 = OpLoad %uint %i
        %188 = OpAccessChain %_ptr_Uniform_Complex_0 %state_vector %int_0 %187
        %189 = OpLoad %Complex_0 %188
        %190 = OpCompositeExtract %float %189 0
        %191 = OpAccessChain %_ptr_Function_float %temp_i %int_0
               OpStore %191 %190
        %192 = OpCompositeExtract %float %189 1
        %193 = OpAccessChain %_ptr_Function_float %temp_i %int_1
               OpStore %193 %192
        %195 = OpLoad %uint %j_0
        %196 = OpAccessChain %_ptr_Uniform_Complex_0 %state_vector %int_0 %195
        %197 = OpLoad %Complex_0 %196
        %198 = OpCompositeExtract %float %197 0
        %199 = OpAccessChain %_ptr_Function_float %temp_j %int_0
               OpStore %199 %198
        %200 = OpCompositeExtract %float %197 1
        %201 = OpAccessChain %_ptr_Function_float %temp_j %int_1
               OpStore %201 %200
        %202 = OpLoad %uint %i
        %203 = OpLoad %Complex %temp_j
        %204 = OpAccessChain %_ptr_Uniform_Complex_0 %state_vector %int_0 %202
        %205 = OpCompositeExtract %float %203 0
        %206 = OpAccessChain %_ptr_Uniform_float %204 %int_0
               OpStore %206 %205
        %207 = OpCompositeExtract %float %203 1
        %208 = OpAccessChain %_ptr_Uniform_float %204 %int_1
               OpStore %208 %207
        %209 = OpLoad %uint %j_0
        %210 = OpLoad %Complex %temp_i
        %211 = OpAccessChain %_ptr_Uniform_Complex_0 %state_vector %int_0 %209
        %212 = OpCompositeExtract %float %210 0
        %213 = OpAccessChain %_ptr_Uniform_float %211 %int_0
               OpStore %213 %212
        %214 = OpCompositeExtract %float %210 1
        %215 = OpAccessChain %_ptr_Uniform_float %211 %int_1
               OpStore %215 %214
               OpBranch %181
        %181 = OpLabel
               OpBranch %109
        %107 = OpLabel
        %217 = OpLoad %uint %target_bit
        %218 = OpIEqual %bool %217 %uint_1
               OpSelectionMerge %220 None
               OpBranchConditional %218 %219 %220
        %219 = OpLabel
        %221 = OpLoad %uint %i
        %222 = OpLoad %uint %i
        %223 = OpAccessChain %_ptr_Uniform_float %state_vector %int_0 %222 %int_0
        %224 = OpLoad %float %223
        %225 = OpFNegate %float %224
        %226 = OpAccessChain %_ptr_Uniform_float %state_vector %int_0 %221 %int_0
               OpStore %226 %225
        %227 = OpLoad %uint %i
        %228 = OpLoad %uint %i
        %229 = OpAccessChain %_ptr_Uniform_float %state_vector %int_0 %228 %int_1
        %230 = OpLoad %float %229
        %231 = OpFNegate %float %230
        %232 = OpAccessChain %_ptr_Uniform_float %state_vector %int_0 %227 %int_1
               OpStore %232 %231
               OpBranch %220
        %220 = OpLabel
               OpBranch %109
        %109 = OpLabel
               OpReturn
               OpFunctionEnd
%complex_add_struct_Complex_f1_f11_struct_Complex_f1_f11_ = OpFunction %Complex None %9
          %a = OpFunctionParameter %_ptr_Function_Complex
          %b = OpFunctionParameter %_ptr_Function_Complex
         %13 = OpLabel
         %26 = OpAccessChain %_ptr_Function_float %a %int_0
         %27 = OpLoad %float %26
         %28 = OpAccessChain %_ptr_Function_float %b %int_0
         %29 = OpLoad %float %28
         %30 = OpFAdd %float %27 %29
         %32 = OpAccessChain %_ptr_Function_float %a %int_1
         %33 = OpLoad %float %32
         %34 = OpAccessChain %_ptr_Function_float %b %int_1
         %35 = OpLoad %float %34
         %36 = OpFAdd %float %33 %35
         %37 = OpCompositeConstruct %Complex %30 %36
               OpReturnValue %37
               OpFunctionEnd
%complex_sub_struct_Complex_f1_f11_struct_Complex_f1_f11_ = OpFunction %Complex None %9
        %a_0 = OpFunctionParameter %_ptr_Function_Complex
        %b_0 = OpFunctionParameter %_ptr_Function_Complex
         %17 = OpLabel
         %40 = OpAccessChain %_ptr_Function_float %a_0 %int_0
         %41 = OpLoad %float %40
         %42 = OpAccessChain %_ptr_Function_float %b_0 %int_0
         %43 = OpLoad %float %42
         %44 = OpFSub %float %41 %43
         %45 = OpAccessChain %_ptr_Function_float %a_0 %int_1
         %46 = OpLoad %float %45
         %47 = OpAccessChain %_ptr_Function_float %b_0 %int_1
         %48 = OpLoad %float %47
         %49 = OpFSub %float %46 %48
         %50 = OpCompositeConstruct %Complex %44 %49
               OpReturnValue %50
               OpFunctionEnd
%complex_scale_struct_Complex_f1_f11_f1_ = OpFunction %Complex None %19
        %a_1 = OpFunctionParameter %_ptr_Function_Complex
          %s = OpFunctionParameter %_ptr_Function_float
         %23 = OpLabel
         %53 = OpAccessChain %_ptr_Function_float %a_1 %int_0
         %54 = OpLoad %float %53
         %55 = OpLoad %float %s
         %56 = OpFMul %float %54 %55
         %57 = OpAccessChain %_ptr_Function_float %a_1 %int_1
         %58 = OpLoad %float %57
         %59 = OpLoad %float %s
         %60 = OpFMul %float %58 %59
         %61 = OpCompositeConstruct %Complex %56 %60
               OpReturnValue %61
               OpFunctionEnd
