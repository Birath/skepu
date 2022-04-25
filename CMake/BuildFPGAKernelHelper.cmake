file(GLOB kernels "${KERNEL_DIR}/*.cl")
MESSAGE(STATUS "KERNELS: ${kernels}")
foreach (kernel ${kernels})
    get_filename_component(kernel_name ${kernel} NAME_WE)
    get_filename_component(kernel_path ${kernel} DIRECTORY)
    set(binary_output "${kernel_path}/${kernel_name}.aocx")
    unset(_kernel_byproducts)
    list(APPEND _kernel_byproducts ${binary_output} "${kernel_path}/${kernel_name}.aocr" "${kernel_name}" )

    # add_custom_command(OUTPUT ${binary_output}
    #     COMMAND "$ENV{INTELFPGAOCLSDKROOT}/bin/aoc" -q -march=emulator ${kernel} -o ${binary_output}
    #     DEPENDS ${kernel}
    #     BYPRODUCTS ${_kernel_byproducts}
    #     VERBATIM
    # )
    # set_source_files_properties(${binary_output}
    #     PROPERTIES
    #         GENERATED TRUE)
    # add_custom_target(${kernel_name} 
    #     DEPENDS ${binary_output} ${_target_name}
    # )	
    # list(APPEND _fpga_kernel_targets ${kernel_name})
    execute_process(COMMAND "$ENV{INTELFPGAOCLSDKROOT}/bin/aoc" -q -march=emulator ${kernel} -o ${binary_output} RESULT_VARIABLE kernel_compile_result)
    MESSAGE(STATUS "Compilation result: ${kernel_compile_result}")
    if (kernel_compile_result)
    	MESSAGE(FATAL_ERROR "OpenCL kernel compilation failed")
    else()
    	list(APPEND _built_kernels ${binary_output})
    endif()

endforeach()