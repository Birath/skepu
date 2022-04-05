#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapKernelTemplate_FPGA = R"~~~(

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef SHIFT_REG_SIZE 
#define SHIFT_REG_SIZE 8
#endif
__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((uses_global_work_offset(0)))
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}} {{SIZE_PARAMS}} {{STRIDE_PARAMS}} size_t skepu_n, size_t skepu_base)
{
	{{CONTAINER_PROXIES}}

	{{MAP_RESULT_TYPE}} shift_reg[SHIFT_REG_SIZE] = {0};

    #pragma ivdep
	#pragma loop_coalesce
	for (int skepu_i = 0; skepu_i < skepu_n + SHIFT_REG_SIZE + 1; skepu_i++) {
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
		#pragma unroll
		for (int i = 0; i < SHIFT_REG_SIZE - 1; i++) {
            shift_reg[i] = shift_reg[i + 1];
        }
		if (skepu_i < skepu_n) {
			shift_reg[SHIFT_REG_SIZE - 1] = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
		} else {
			shift_reg[SHIFT_REG_SIZE - 1] = 0;
		}

		if (skepu_i >= SHIFT_REG_SIZE - 1) {
			skepu_output[skepu_i - SHIFT_REG_SIZE + 1] = shift_reg[0];
		}
	}
}
)~~~";

const char *MapKernelTemplate_FPGA_NDRange = R"~~~(
__attribute(num_compute_units(MAP_CU_COUNT))
__attribute((reqd_work_group_size(BLOCK_SIZE,1,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}} {{SIZE_PARAMS}} {{STRIDE_PARAMS}} size_t skepu_n, size_t skepu_base)
{
	size_t skepu_i = get_global_id(0);
	size_t skepu_global_prng_id = get_global_id(0);
	size_t skepu_gridSize = get_local_size(0) * get_num_groups(0);
	{{CONTAINER_PROXIES}}
	{{STRIDE_INIT}}

	while (skepu_i < skepu_n)
	{
		{{INDEX_INITIALIZER}}
		{{CONTAINER_PROXIE_INNER}}
#if !{{USE_MULTIRETURN}}
		skepu_output[skepu_i * skepu_stride_0] = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
#else
		{{MULTI_TYPE}} skepu_out_temp = {{FUNCTION_NAME_MAP}}({{MAP_ARGS}});
		{{OUTPUT_ASSIGN}}
#endif
		skepu_i += skepu_gridSize;
	}
}
)~~~";


const std::string Constructor = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	static cl_kernel skepu_kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}

	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;

		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			std::ifstream binary_source_file
			("skepu_precompiled/{{KERNEL_NAME}}.aocx", std::ios::binary);
			if (!binary_source_file.is_open()) {
				std::cerr << "Failed to open binary kernel file " << "{{KERNEL_NAME}}.aocx" << '\n';
				return;
			}
			std::vector<unsigned char> binary_source(std::istreambuf_iterator<char>(binary_source_file), {});
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildBinaryProgram(device, binary_source);
			cl_kernel kernel = clCreateKernel(program, "{{KERNEL_NAME}}", &err);
			CL_CHECK_ERROR(err, "Error creating map kernel '{{KERNEL_NAME}}'");

			skepu_kernels(counter++, &kernel);
		}

		initialized = true;
	}

	{{TEMPLATE_HEADER}}
	static void map
	(
		size_t skepu_deviceID,
		{{HOST_KERNEL_PARAMS}} {{SIZES_TUPLE_PARAM}}
		size_t skepu_n, size_t skepu_base, skepu::StrideList<{{STRIDE_COUNT}}> skepu_strides
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernels(skepu_deviceID), {{KERNEL_ARGS}} {{SIZE_ARGS}} {{STRIDE_ARGS}} skepu_n, skepu_base);
		cl_int skepu_err = clEnqueueTask(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernels(skepu_deviceID), 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching Map kernel");
	}
};
)~~~";


std::string createMapKernelProgram_FPGA(SkeletonInstance &instance, UserFunction &mapFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs;
	std::stringstream SSStrideParams, SSStrideArgs, SSStrideInit;
	IndexCodeGen indexInfo = indexInitHelper_CL(mapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapFuncArgs << indexInfo.mapFuncParam;
	std::string multiOutputAssign = handleOutputs_CL(mapFunc, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, true);
	handleRandomParam_CL(mapFunc, sourceStream, SSMapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	
	size_t stride_counter = 0;
	for (size_t er = 0; er < std::max<size_t>(1, mapFunc.multipleReturnTypes.size()); ++er)
	{
		std::stringstream namesuffix;
		if (mapFunc.multipleReturnTypes.size()) namesuffix << "_" << stride_counter;
		SSStrideParams << "int skepu_stride_" << stride_counter << ", ";
		SSStrideArgs << "skepu_strides[" << stride_counter << "], ";
		SSStrideInit << "if (skepu_stride_" << stride_counter << " < 0) { skepu_output" << namesuffix.str() << " += (-skepu_n + 1) * skepu_stride_" << stride_counter << "; }\n";
		stride_counter++;
	}
	
	// Elementwise input data
	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSStrideParams << "int skepu_stride_" << stride_counter << ", ";
		SSStrideArgs << "skepu_strides[" << stride_counter << "], ";
		SSStrideInit << "if (skepu_stride_" << stride_counter << " < 0) { " << param.name << " += (-skepu_n + 1) * skepu_stride_" << stride_counter << "; }\n";
		SSKernelParamList << "__global " << param.typeNameOpenCL() << " const * restrict " << param.name << ", ";
		SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<" << param.resolvedTypeName << "> *" << param.name << ", ";
		SSKernelArgs << param.name << "->getDeviceDataPointer(), ";
		SSMapFuncArgs << param.name << "[skepu_i]";
		stride_counter++;
		first = false;
	}

	auto argsInfo = handleRandomAccessAndUniforms_CL(mapFunc, SSMapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	handleUserTypesConstantsAndPrecision_CL({&mapFunc}, sourceStream);
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);
	sourceStream << generateUserFunctionCode_CL(mapFunc) << MapKernelTemplate_FPGA;
	
	std::stringstream SSKernelName;
	SSKernelName << instance << "_" << transformToCXXIdentifier(ResultName) << "_MapKernel_" << mapFunc.uniqueName << "_arity_" << mapFunc.Varity;
	const std::string kernelName = SSKernelName.str();
	
	std::stringstream SSStrideCount;
	SSStrideCount << (mapFunc.elwiseParams.size() + std::max<size_t>(1, mapFunc.multipleReturnTypes.size()));
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_fpga_source.inl"};
	FSOutFile << templateString(Constructor,
	{
		{"{{OPENCL_KERNEL}}",          sourceStream.str()},
		{"{{KERNEL_NAME}}",            kernelName},
		{"{{FUNCTION_NAME_MAP}}",      mapFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",          SSKernelParamList.str()},
		{"{{HOST_KERNEL_PARAMS}}",     SSHostKernelParamList.str()},
		{"{{MAP_ARGS}}",               SSMapFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",      indexInfo.indexInit},
		{"{{KERNEL_CLASS}}",           "FPGAWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",            SSKernelArgs.str()},
		{"{{CONTAINER_PROXIES}}",      argsInfo.proxyInitializer},
		{"{{CONTAINER_PROXIE_INNER}}", argsInfo.proxyInitializerInner},
		{"{{SIZE_PARAMS}}",            indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",              indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",      indexInfo.sizesTupleParam},
		{"{{STRIDE_PARAMS}}",          SSStrideParams.str()},
		{"{{STRIDE_ARGS}}",            SSStrideArgs.str()},
		{"{{STRIDE_COUNT}}",           SSStrideCount.str()},
		{"{{STRIDE_INIT}}",            SSStrideInit.str()},
		{"{{MAP_RESULT_TYPE}}", 	   mapFunc.resolvedReturnTypeName},
		{"{{TEMPLATE_HEADER}}",        indexInfo.templateHeader},
		{"{{MULTI_TYPE}}",             mapFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",        (mapFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",          multiOutputAssign}
	});
			
	std::stringstream kernelStream{};
	kernelStream << templateString(sourceStream.str(), {
		{"{{KERNEL_NAME}}",            kernelName},
		{"{{FUNCTION_NAME_MAP}}",      mapFunc.uniqueName},
		{"{{KERNEL_PARAMS}}",          SSKernelParamList.str()},
		{"{{HOST_KERNEL_PARAMS}}",     SSHostKernelParamList.str()},
		{"{{MAP_ARGS}}",               SSMapFuncArgs.str()},
		{"{{INDEX_INITIALIZER}}",      indexInfo.indexInit},
		{"{{KERNEL_CLASS}}",           "FPGAWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",            SSKernelArgs.str()},
		{"{{CONTAINER_PROXIES}}",      argsInfo.proxyInitializer},
		{"{{CONTAINER_PROXIE_INNER}}", argsInfo.proxyInitializerInner},
		{"{{SIZE_PARAMS}}",            indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",              indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",      indexInfo.sizesTupleParam},
		{"{{STRIDE_PARAMS}}",          SSStrideParams.str()},
		{"{{STRIDE_ARGS}}",            SSStrideArgs.str()},
		{"{{STRIDE_COUNT}}",           SSStrideCount.str()},
		{"{{STRIDE_INIT}}",            SSStrideInit.str()},
		{"{{MAP_RESULT_TYPE}}", 	   mapFunc.resolvedReturnTypeName},
		{"{{TEMPLATE_HEADER}}",        indexInfo.templateHeader},
		{"{{MULTI_TYPE}}",             mapFunc.multiReturnTypeNameGPU()},
		{"{{USE_MULTIRETURN}}",        (mapFunc.multipleReturnTypes.size() > 0) ? "1" : "0"},
		{"{{OUTPUT_ASSIGN}}",          multiOutputAssign}
	});

	// Replace usage of size_t to match host platform size
	// Copied from skepu_opencl_helper
	// FIXME
	// Add error?
	std::string kernel_source = kernelStream.str();
	if (sizeof(size_t) <= sizeof(unsigned int))
		replaceTextInString(kernel_source, std::string("size_t "), "unsigned int ");
	else if (sizeof(size_t) <= sizeof(unsigned long))
		replaceTextInString(kernel_source, std::string("size_t "), "unsigned long ");

	// TEMP fix for get_device_id() in kernel
	replaceTextInString(kernel_source, "SKEPU_INTERNAL_DEVICE_ID", "0");
	std::ofstream openClKernel {dir + "/" + kernelName + ".cl"};
	openClKernel << kernel_source;

	return kernelName;
}
