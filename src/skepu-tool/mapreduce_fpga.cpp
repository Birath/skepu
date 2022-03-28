#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const char *MapReduceKernelTemplate_FPGA = R"~~~(
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef USER_FUNC_LATENCY
#define USER_FUNC_LATENCY 8
#endif
__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((uses_global_work_offset(0)))
__kernel void {{KERNEL_NAME}}({{KERNEL_PARAMS}} __global {{REDUCE_RESULT_TYPE}}* restrict skepu_output, {{SIZE_PARAMS}} {{STRIDE_PARAMS}} size_t skepu_n, size_t skepu_base)
{
	{{CONTAINER_PROXIES}}
	{{REDUCE_RESULT_TYPE}} shift_reg[USER_FUNC_LATENCY] = {0};

	int exit = (skepu_n % UNROLL == 0) ? (skepu_n / UNROLL) : (skepu_n / UNROLL) + 1;
	for (int i = 0; i < exit; i++) {
		{{REDUCE_RESULT_TYPE}} partial_result = 0;
		#pragma unroll 
		for (int j = 0; j < UNROLL; j++) {
			int skepu_i = i * UNROLL + j;
			{{INDEX_INITIALIZER}}
			{{CONTAINER_PROXIE_INNER}}
			if (skepu_i < skepu_n) {
				{{MAP_RESULT_TYPE}} tmpMap = {{FUNCTION_NAME_MAP}}({{MAP_PARAMS}});
				partial_result = {{FUNCTION_NAME_REDUCE}}(partial_result, tmpMap);
			}

			// partial_result = (index < skepu_n) ? {{FUNCTION_NAME_REDUCE}}(partial_result, input[index]) : partial_result;
		}

		{{REDUCE_RESULT_TYPE}} cur = {{FUNCTION_NAME_REDUCE}}(shift_reg[USER_FUNC_LATENCY-1], partial_result); 
		#pragma unroll
		for (int j = USER_FUNC_LATENCY - 1; j > 0; j--) { 
			shift_reg[j] = shift_reg[j-1]; 
		}
		shift_reg[0] = cur;
	}
	{{REDUCE_RESULT_TYPE}} result = 0;
	#pragma unroll 
	for (int i = 0; i < USER_FUNC_LATENCY; i++) {
		result = {{FUNCTION_NAME_REDUCE}}(shift_reg[i], result); 
	}   
	skepu_output[0] = result;

	// size_t skepu_global_prng_id = get_global_id(0);
	// size_t skepu_blockSize = get_local_size(0);
	// size_t skepu_tid = get_local_id(0);
	// size_t skepu_global_id = get_group_id(0) * skepu_blockSize + skepu_tid;
	// size_t skepu_i = get_group_id(0) * skepu_blockSize + skepu_tid;
	// size_t skepu_gridSize = skepu_blockSize * get_num_groups(0);
	// {{REDUCE_RESULT_TYPE}} skepu_result;
	// {{CONTAINER_PROXIES}}
	// {{STRIDE_INIT}}

	// if (skepu_i < skepu_n)
	// {
	// 	{{INDEX_INITIALIZER}}
	// 	{{CONTAINER_PROXIE_INNER}}
	// 	skepu_result = {{FUNCTION_NAME_MAP}}({{MAP_PARAMS}});
	// 	skepu_i += skepu_gridSize;
	// }

	// while (skepu_i < skepu_n)
	// {
	// 	{{INDEX_INITIALIZER}}
	// 	{{CONTAINER_PROXIE_INNER}}
	// 	{{MAP_RESULT_TYPE}} tempMap = {{FUNCTION_NAME_MAP}}({{MAP_PARAMS}});
	// 	skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, tempMap);
	// 	skepu_i += skepu_gridSize;
	// }

	// skepu_sdata[skepu_tid] = skepu_result;
	// barrier(CLK_LOCAL_MEM_FENCE);
	
	// if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	// if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }

	// if (skepu_tid == 0)
	// {
	// 	skepu_output[get_group_id(0)] = skepu_sdata[skepu_tid];
	// }
}
)~~~";

static const char *ReduceKernelTemplate_CL = R"~~~(
__kernel void {{KERNEL_NAME}}_ReduceOnly(__global {{REDUCE_RESULT_TYPE}}* skepu_input, __global {{REDUCE_RESULT_TYPE}}* skepu_output, size_t skepu_n, __local {{REDUCE_RESULT_TYPE}}* skepu_sdata)
{
	size_t skepu_blockSize = get_local_size(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * skepu_blockSize + get_local_id(0);
	size_t skepu_gridSize = skepu_blockSize * get_num_groups(0);
	{{REDUCE_RESULT_TYPE}} skepu_result;

	if (skepu_i < skepu_n)
	{
		skepu_result = skepu_input[skepu_i];
		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		skepu_result = {{FUNCTION_NAME_REDUCE}}(skepu_result, skepu_input[skepu_i]);
		skepu_i += skepu_gridSize;
	}

	skepu_sdata[skepu_tid] = skepu_result;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_n) { skepu_sdata[skepu_tid] = {{FUNCTION_NAME_REDUCE}}(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }

	if (skepu_tid == 0)
	{
		skepu_output[get_group_id(0)] = skepu_sdata[skepu_tid];
	}
}
)~~~";


const std::string Constructor = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	enum
	{
		KERNEL_MAPREDUCE = 0,
		KERNEL_REDUCE,
		KERNEL_COUNT
	};

	static cl_kernel kernels(size_t deviceID, size_t kerneltype, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8][KERNEL_COUNT]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID][kerneltype] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID][kerneltype];
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

			cl_kernel kernel_mapreduce = clCreateKernel(program, "{{KERNEL_NAME}}", &err);
			CL_CHECK_ERROR(err, "Error creating MapReduce kernel '{{KERNEL_NAME}}'");

			cl_kernel kernel_reduce = clCreateKernel(program, "{{KERNEL_NAME}}_ReduceOnly", &err);
			CL_CHECK_ERROR(err, "Error creating MapReduce_ReduceOnly kernel '{{KERNEL_NAME}}_ReduceOnly'");

			kernels(counter, KERNEL_MAPREDUCE, &kernel_mapreduce);
			kernels(counter, KERNEL_REDUCE, &kernel_reduce);
			counter++;
		}

		initialized = true;
	}

	{{TEMPLATE_HEADER}}
	static void mapReduce
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		{{HOST_KERNEL_PARAMS}}
		skepu::backend::DeviceMemPointer_CL<{{REDUCE_RESULT_CPU}}> *skepu_output,
		{{SIZES_TUPLE_PARAM}} size_t skepu_n, size_t skepu_base, skepu::StrideList<{{STRIDE_COUNT}}> skepu_strides,
		size_t skepu_sharedMemSize
	)
	{
		cl_kernel skepu_kernel = kernels(skepu_deviceID, KERNEL_MAPREDUCE);
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernel, {{KERNEL_ARGS}} skepu_output->getDeviceDataPointer(), {{SIZE_ARGS}} {{STRIDE_ARGS}} skepu_n, skepu_base);
		size_t size = 1;
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernel, 1, NULL, &size, &size, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapReduce kernel");
	}

	static void reduceOnly
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		skepu::backend::DeviceMemPointer_CL<{{REDUCE_RESULT_CPU}}> *skepu_input, skepu::backend::DeviceMemPointer_CL<{{REDUCE_RESULT_CPU}}> *skepu_output,
		size_t skepu_n, size_t skepu_sharedMemSize
	)
	{
		cl_kernel skepu_kernel = kernels(skepu_deviceID, KERNEL_REDUCE);
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernel, skepu_input->getDeviceDataPointer(), skepu_output->getDeviceDataPointer(), skepu_n);
		clSetKernelArg(skepu_kernel, 3, skepu_sharedMemSize, NULL);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernel, 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapReduce reduce-only kernel");
	}
};
)~~~";


std::string createMapReduceKernelProgram_FPGA(SkeletonInstance &instance, UserFunction &mapFunc, UserFunction &reduceFunc, std::string dir)
{
	std::stringstream sourceStream, SSKernelParamList, SSMapFuncArgs, SSHostKernelParamList, SSKernelArgs;
	std::stringstream SSStrideParams, SSStrideArgs, SSStrideInit;
	IndexCodeGen indexInfo = indexInitHelper_CL(mapFunc);
	bool first = !indexInfo.hasIndex;
	SSMapFuncArgs << indexInfo.mapFuncParam;
	
	handleRandomParam_CL(mapFunc, sourceStream, SSMapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	
	size_t stride_counter = 0;
	for (UserFunction::Param& param : mapFunc.elwiseParams)
	{
		if (!first) { SSMapFuncArgs << ", "; }
		SSStrideParams << "int skepu_stride_" << stride_counter << ", ";
		SSStrideArgs << "skepu_strides[" << stride_counter << "], ";
		SSStrideInit << "if (skepu_stride_" << stride_counter << " < 0) { " << param.name << " += (-skepu_n + 1) * skepu_stride_" << stride_counter << "; }\n";
		SSKernelParamList << "__global " << param.typeNameOpenCL() << " const * restrict " << param.name << ", ";
		SSHostKernelParamList << "skepu::backend::DeviceMemPointer_CL<const " << param.resolvedTypeName << "> * " << param.name << ", ";
		SSKernelArgs << param.name << "->getDeviceDataPointer(), ";
		SSMapFuncArgs << param.name << "[skepu_i]";
		stride_counter++;
		first = false;
	}
	
	auto argsInfo = handleRandomAccessAndUniforms_CL(mapFunc, SSMapFuncArgs, SSHostKernelParamList, SSKernelParamList, SSKernelArgs, first);
	handleUserTypesConstantsAndPrecision_CL({&mapFunc, &reduceFunc}, sourceStream);
	
	proxyCodeGenHelper_CL(argsInfo.containerProxyTypes, sourceStream);

	if (mapFunc.refersTo(reduceFunc))
		sourceStream << generateUserFunctionCode_CL(mapFunc);
	else if (reduceFunc.refersTo(mapFunc))
		sourceStream << generateUserFunctionCode_CL(reduceFunc);
	else
		sourceStream << generateUserFunctionCode_CL(mapFunc) << generateUserFunctionCode_CL(reduceFunc);

	sourceStream << MapReduceKernelTemplate_FPGA << ReduceKernelTemplate_CL;
	
	std::stringstream SSKernelName;
	SSKernelName << instance << "_" << transformToCXXIdentifier(ResultName) << "_MapReduceKernel_" << mapFunc.uniqueName << "_" << reduceFunc.uniqueName << "_arity_" << mapFunc.Varity << "uid_" << GlobalSkeletonIndex++;
	const std::string kernelName = SSKernelName.str();
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << mapFunc.numKernelArgsCL() + 2 + std::max<int>(0, indexInfo.dim - 1) + stride_counter;
	
	std::stringstream SSStrideCount;
	SSStrideCount << mapFunc.elwiseParams.size();
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor,
	{
		{"{{OPENCL_KERNEL}}",          sourceStream.str()},
		{"{{KERNEL_CLASS}}",           "FPGAWrapperClass_" + kernelName},
		{"{{KERNEL_ARGS}}",            SSKernelArgs.str()},
		{"{{KERNEL_ARG_COUNT}}",       SSKernelArgCount.str()},
		{"{{HOST_KERNEL_PARAMS}}",     SSHostKernelParamList.str()},
		{"{{CONTAINER_PROXIES}}",      argsInfo.proxyInitializer},
		{"{{CONTAINER_PROXIE_INNER}}", argsInfo.proxyInitializerInner},
		{"{{KERNEL_PARAMS}}",          SSKernelParamList.str()},
		{"{{MAP_PARAMS}}",             SSMapFuncArgs.str()},
		{"{{REDUCE_RESULT_TYPE}}",     reduceFunc.rawReturnTypeName},
		{"{{REDUCE_RESULT_CPU}}",      reduceFunc.resolvedReturnTypeName},
		{"{{MAP_RESULT_TYPE}}",        mapFunc.rawReturnTypeName},
		{"{{FUNCTION_NAME_MAP}}",      mapFunc.uniqueName},
		{"{{FUNCTION_NAME_REDUCE}}",   reduceFunc.uniqueName},
		{"{{KERNEL_NAME}}",            kernelName},
		{"{{INDEX_INITIALIZER}}",      indexInfo.indexInit},
		{"{{SIZE_PARAMS}}",            indexInfo.sizeParams},
		{"{{SIZE_ARGS}}",              indexInfo.sizeArgs},
		{"{{SIZES_TUPLE_PARAM}}",      indexInfo.sizesTupleParam},
		{"{{STRIDE_PARAMS}}",          SSStrideParams.str()},
		{"{{STRIDE_ARGS}}",            SSStrideArgs.str()},
		{"{{STRIDE_COUNT}}",           SSStrideCount.str()},
		{"{{STRIDE_INIT}}",            SSStrideInit.str()},
		{"{{TEMPLATE_HEADER}}",        indexInfo.templateHeader}
	});

	std::stringstream kernelStream{};
	kernelStream << templateString(sourceStream.str(), {
		{"{{KERNEL_NAME}}",            kernelName},
		{"{{CONTAINER_PROXIES}}",      argsInfo.proxyInitializer},
		{"{{CONTAINER_PROXIE_INNER}}", argsInfo.proxyInitializerInner},
		{"{{KERNEL_PARAMS}}",          SSKernelParamList.str()},
		{"{{MAP_PARAMS}}",             SSMapFuncArgs.str()},
		{"{{REDUCE_RESULT_TYPE}}",     reduceFunc.rawReturnTypeName},
		{"{{MAP_RESULT_TYPE}}",        mapFunc.rawReturnTypeName},
		{"{{FUNCTION_NAME_MAP}}",      mapFunc.uniqueName},
		{"{{FUNCTION_NAME_REDUCE}}",   reduceFunc.uniqueName},
		{"{{INDEX_INITIALIZER}}",      indexInfo.indexInit},
		{"{{SIZE_PARAMS}}",            indexInfo.sizeParams},
		{"{{STRIDE_PARAMS}}",          SSStrideParams.str()},
		{"{{STRIDE_INIT}}",            SSStrideInit.str()}
	});

	// Replace usage of size_t to match host platform size
	// Copied from skepu_opencl_helper
	// FIXME
	// Add error?
	std::string kernelSource = kernelStream.str();
	if (sizeof(size_t) <= sizeof(unsigned int))
		replaceTextInString(kernelSource, std::string("size_t "), "unsigned int ");
	else if (sizeof(size_t) <= sizeof(unsigned long))
		replaceTextInString(kernelSource, std::string("size_t "), "unsigned long ");
		
	// TEMP fix for get_device_id() in kernel
	replaceTextInString(kernelSource, "SKEPU_INTERNAL_DEVICE_ID", "0");
	std::ofstream kernelFile {dir + "/" + kernelName + ".cl"};
	kernelFile << kernelSource;

	return kernelName;
}
