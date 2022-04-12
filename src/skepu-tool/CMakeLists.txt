set(LLVM_LINK_COMPONENTS Support)

add_clang_executable(skepu-tool
  skepu.cpp
  visitor.cpp
  data_structures.cpp
  code_gen.cpp
  code_gen_cl.cpp
  code_gen_cu.cpp
  mapreduce_cl.cpp
  mapreduce_cu.cpp
  mapreduce_fpga.cpp
  map_cl.cpp
  map_cu.cpp
  map_fpga.cpp
  scan_cl.cpp
  scan_cu.cpp
  scan_fpga.cpp
  reduce_cl.cpp
  reduce_fpga.cpp
  reduce_cu.cpp
  mapoverlap_cl.cpp
  mapoverlap_cu.cpp
  mapoverlap_fpga.cpp
  mappairs_cl.cpp
  mappairs_cu.cpp
  mappairsreduce_cl.cpp
  mappairsreduce_cu.cpp
  call_cl.cpp
  call_cu.cpp)

clang_target_link_libraries(skepu-tool
	PRIVATE
		clangTooling
		clangBasic
		clangAST
		clangASTMatchers)

target_compile_options(skepu-tool PRIVATE
  -g 
)

install(TARGETS skepu-tool
	EXPORT skepu-toolTargets
	RUNTIME
	DESTINATION bin)
install(EXPORT skepu-toolTargets
	DESTINATION lib/cmake/SkePU
	NAMESPACE SkePU::)
