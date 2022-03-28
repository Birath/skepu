#include <algorithm>

#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

static const char *ReduceKernelTemplate_CL = R"~~~(

#ifndef REDUCTION_UNROLL
#define REDUCTION_UNROLL 16
#endif
#ifndef USER_FUNC_LATENCY
#define USER_FUNC_LATENCY 8
#endif

__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((uses_global_work_offset(0)))
__kernel void {{KERNEL_NAME}}(__global {{REDUCE_RESULT_TYPE}} const* restrict input, __global {{REDUCE_RESULT_TYPE}}* restrict output, size_t size)
{
	{{REDUCE_RESULT_TYPE}} shift_reg[USER_FUNC_LATENCY] = {0};

	int exit = (size % REDUCTION_UNROLL == 0) ? (size / REDUCTION_UNROLL) : (size / REDUCTION_UNROLL) + 1;
	for (int i = 0; i < exit; i++) {
		{{REDUCE_RESULT_TYPE}} partial_result = 0;
		#pragma unroll 
		for (int j = 0; j < REDUCTION_UNROLL; j++) {
			int index = i * REDUCTION_UNROLL + j;
			partial_result = (index < size) ? {{FUNCTION_NAME_REDUCE}}(partial_result, input[index]) : partial_result;
		}

		shift_reg[USER_FUNC_LATENCY - 1] = {{FUNCTION_NAME_REDUCE}}(shift_reg[0], partial_result); 
		#pragma unroll
		for (int j = 0 ; j < USER_FUNC_LATENCY - 1; j++) { 
			shift_reg[j] = shift_reg[j + 1]; 
		}
	}
	{{REDUCE_RESULT_TYPE}} result = 0;
	#pragma unroll 
	for (int i = 0; i < USER_FUNC_LATENCY; i++) {
		result = {{FUNCTION_NAME_REDUCE}}(shift_reg[i], result); 
	}   
	output[0] = result;
}
)~~~";


const std::string Constructor1D = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
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
			CL_CHECK_ERROR(err, "Error creating reduce kernel '{{KERNEL_NAME}}'");

			kernels(counter++, &kernel);
		}

		initialized = true;
	}

	static void reduce(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		skepu::backend::cl_helpers::setKernelArgs(kernels(deviceID), input, output, n);
		size_t size = 1;
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernels(deviceID), 1, NULL, &size, &size, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Reduce kernel");
	}
};
)~~~";


std::string createReduce1DKernelProgram_FPGA(SkeletonInstance &instance, UserFunction &reduceFunc, std::string dir)
{
	std::stringstream sourceStream;

	if (reduceFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";

	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";

	// check for extra user-supplied opencl code for custome datatype
	for (UserType *RefType : reduceFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_ReduceKernel_" + reduceFunc.uniqueName;
	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(reduceFunc) << ReduceKernelTemplate_CL;

	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << templateString(Constructor1D,
	{
		{"{{OPENCL_KERNEL}}",        sourceStream.str()},
		{"{{KERNEL_CLASS}}",         "FPGAWrapperClass_" + kernelName},
		{"{{REDUCE_RESULT_TYPE}}",   reduceFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName},
		{"{{FUNCTION_NAME_REDUCE}}", reduceFunc.uniqueName}
	});

	std::stringstream kernelStream{};
	kernelStream << templateString(sourceStream.str(),
	{
		{"{{REDUCE_RESULT_TYPE}}",   reduceFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName},
		{"{{FUNCTION_NAME_REDUCE}}", reduceFunc.uniqueName}
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



const std::string Constructor2D = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	enum
	{
		KERNEL_ROWWISE = 0,
		KERNEL_COLWISE,
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

			cl_kernel rowwisekernel = clCreateKernel(program, "{{KERNEL_NAME}}_RowWise", &err);
			CL_CHECK_ERROR(err, "Error creating row-wise Reduce kernel '{{KERNEL_NAME}}'");

			cl_kernel colwisekernel = clCreateKernel(program, "{{KERNEL_NAME}}_ColWise", &err);
			CL_CHECK_ERROR(err, "Error creating col-wise Reduce kernel '{{KERNEL_NAME}}'");

			kernels(counter++, KERNEL_ROWWISE, &rowwisekernel);
			kernels(counter++, KERNEL_COLWISE, &colwisekernel);
		}

		initialized = true;
	}

	static void reduceRowWise(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_ROWWISE);
		skepu::backend::cl_helpers::setKernelArgs(kernel, input, output, n);
		clSetKernelArg(kernel, 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}

	static void reduceColWise(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_COLWISE);
		skepu::backend::cl_helpers::setKernelArgs(kernel, input, output, n);
		clSetKernelArg(kernel, 3, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Map kernel");
	}

	static void reduce(size_t deviceID, size_t localSize, size_t globalSize, cl_mem input, cl_mem output, size_t n, size_t sharedMemSize)
	{
		reduceRowWise(deviceID, localSize, globalSize, input, output, n, sharedMemSize);
	}
};
)~~~";

std::string createReduce2DKernelProgram_FPGA(SkeletonInstance &instance, UserFunction &rowWiseFunc, UserFunction &colWiseFunc, std::string dir)
{
	std::stringstream sourceStream;
	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_ReduceKernel_" + rowWiseFunc.uniqueName + "_" + colWiseFunc.uniqueName;

	if (rowWiseFunc.requiresDoublePrecision || colWiseFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";

	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";

	// check for extra user-supplied opencl code for custome datatype
	std::set<UserType*> referencedUTs;
	std::set_union(
		rowWiseFunc.ReferencedUTs.cbegin(), rowWiseFunc.ReferencedUTs.cend(),
		colWiseFunc.ReferencedUTs.cbegin(), colWiseFunc.ReferencedUTs.cend(), std::inserter(referencedUTs, referencedUTs.begin()));

	for (UserType *RefType : referencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);

	const std::string className = "FPGAWrapperClass_" + kernelName;

	sourceStream << KernelPredefinedTypes_CL;

	if (rowWiseFunc.refersTo(colWiseFunc))
		sourceStream << generateUserFunctionCode_CL(rowWiseFunc);
	else if (colWiseFunc.refersTo(rowWiseFunc))
		sourceStream << generateUserFunctionCode_CL(colWiseFunc);
	else
		sourceStream << generateUserFunctionCode_CL(rowWiseFunc) << generateUserFunctionCode_CL(colWiseFunc);

	sourceStream << templateString(ReduceKernelTemplate_CL,
	{
		{"{{REDUCE_RESULT_TYPE}}",   rowWiseFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName + "_RowWise"},
		{"{{FUNCTION_NAME_REDUCE}}", rowWiseFunc.uniqueName}
	});
	sourceStream << templateString(ReduceKernelTemplate_CL,
	{
		{"{{REDUCE_RESULT_TYPE}}",   colWiseFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",          kernelName + "_ColWise"},
		{"{{FUNCTION_NAME_REDUCE}}", colWiseFunc.uniqueName}
	});

	std::string finalSource = Constructor2D;
	replaceTextInString(finalSource, "{{OPENCL_KERNEL}}", sourceStream.str());
	replaceTextInString(finalSource, "{{KERNEL_NAME}}", kernelName);
	replaceTextInString(finalSource, "{{KERNEL_CLASS}}", className);

	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;

	// Replace usage of size_t to match host platform size
	// Copied from skepu_opencl_helper
	// FIXME
	// Add error?
	std::string kernelSource = sourceStream.str();
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
