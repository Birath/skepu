#include "code_gen.h"
#include "code_gen_cl.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

const std::string ScanKernel_FPGA = R"~~~(
#ifndef SCAN_UNROLL
#define SCAN_UNROLL 8
#endif

#ifndef SCAN_FUNC_LATENCY
#define SHIFT_REG_SIZE (16 * SCAN_UNROLL)
#else
#define SHIFT_REG_SIZE (SCAN_FUNC_LATENCY * SCAN_UNROLL)
#endif

#define SCAN_READ_OFFSET (8 * SCAN_UNROLL)

#if SHIFT_REG_SIZE <= SCAN_READ_OFFSET
#error SHIFT_REG_SIZE Must be greater than SCAN_WRITE_OFFSET
#endif

__attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((uses_global_work_offset(0)))
__kernel void {{KERNEL_NAME}}_Scan(__global {{SCAN_TYPE}} const* restrict skepu_input, __global {{SCAN_TYPE}}* restrict skepu_output, size_t const skepu_n, int const isInclusive)
{
	{{SCAN_TYPE}} shift_reg[SHIFT_REG_SIZE + 1] = {0};
	int write_delay = SHIFT_REG_SIZE - SCAN_READ_OFFSET + isInclusive;
	#pragma unroll SCAN_UNROLL
	for (int skepu_i = 0; skepu_i < skepu_n + write_delay; skepu_i++) {
        #pragma unroll
		for (int i = 0; i < SHIFT_REG_SIZE; i++) {
            shift_reg[i] = shift_reg[i + 1];
        }

        if (skepu_i < skepu_n) {
			shift_reg[SHIFT_REG_SIZE] = {{FUNCTION_NAME_SCAN}}(shift_reg[SHIFT_REG_SIZE - SCAN_READ_OFFSET], skepu_input[skepu_i]);
		} else { 
			shift_reg[SHIFT_REG_SIZE] = 0;
		}

        if (skepu_i >= write_delay) {
			{{SCAN_TYPE}} sum = 0;
			#pragma unroll
			for (int i = 0; i < SCAN_READ_OFFSET; i++) {
				sum = {{FUNCTION_NAME_SCAN}}(shift_reg[i], sum);
			}
			skepu_output[skepu_i - write_delay] = sum;
		}
	}
}
)~~~";


const std::string Constructor = R"~~~(
class {{KERNEL_CLASS}}
{
public:

	enum
	{
		KERNEL_SCAN = 0,
		KERNEL_SCAN_UPDATE,
		KERNEL_SCAN_ADD,
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
			("skepu_precompiled/{{KERNEL_DIR}}/{{KERNEL_NAME}}_fpga.aocx", std::ios::binary);
			if (!binary_source_file.is_open()) {
				std::cerr << "Failed to open binary kernel file " << "{{KERNEL_NAME}}_fpga.aocx" << '\n';
				return;
			}
			std::vector<unsigned char> binary_source(std::istreambuf_iterator<char>(binary_source_file), {});
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildBinaryProgram(device, binary_source);

			cl_kernel kernel_scan = clCreateKernel(program, "{{KERNEL_NAME}}_Scan", &err);
			CL_CHECK_ERROR(err, "Error creating Scan kernel '{{KERNEL_NAME}}'");

			kernels(counter, KERNEL_SCAN,        &kernel_scan);
			counter++;
		}

		initialized = true;
	}

	static void scan
	(
		size_t deviceID,
		skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *skepu_input, skepu::backend::DeviceMemPointer_CL<{{SCAN_TYPE}}> *skepu_output, 
		size_t skepu_n, int isInclusive
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_SCAN);
		skepu::backend::cl_helpers::setKernelArgs(kernel, skepu_input->getDeviceDataPointer(), skepu_output->getDeviceDataPointer(), skepu_n, isInclusive);
		cl_int err = clEnqueueTask(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching Scan kernel");
	}
};
)~~~";


std::string createScanKernelProgram_FPGA(SkeletonInstance &instance, UserFunction &scanFunc, std::string dir)
{
	std::stringstream sourceStream;

	if (scanFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";

	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";

	// check for extra user-supplied opencl code for custome datatype
	// TODO: Also check the referenced UFs for referenced UTs skepu::userstruct
	for (UserType *RefType : scanFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);

	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_CL(scanFunc) << ScanKernel_FPGA;

	const std::string kernelName = instance + "_" + transformToCXXIdentifier(ResultName) + "_ScanKernel_" + scanFunc.uniqueName;
	std::ofstream FSOutFile {dir + "/" + kernelName + "_fpga_source.inl"};
	FSOutFile << templateString(Constructor,
	{
		{"{{OPENCL_KERNEL}}", sourceStream.str()},
		{"{{KERNEL_CLASS}}",  "FPGAWrapperClass_" + kernelName},
		{"{{SCAN_TYPE}}",           scanFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",         kernelName},
		{"{{KERNEL_DIR}}",          ResultName},
		{"{{FUNCTION_NAME_SCAN}}",  scanFunc.uniqueName}
	});

	std::stringstream kernelStream{};
	kernelStream << templateString(sourceStream.str(), {
		{"{{KERNEL_NAME}}",         kernelName},
		{"{{SCAN_TYPE}}",           scanFunc.resolvedReturnTypeName},
		{"{{KERNEL_NAME}}",         kernelName},
		{"{{FUNCTION_NAME_SCAN}}",  scanFunc.uniqueName}
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
	std::ofstream kernelFile {dir + "/" + ResultName + "/" + kernelName + "_fpga.cl"};
	kernelFile << kernelSource;

	return kernelName;
}
