set(_zerofg_shader_compiler
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/compile_vulkan13_shader.py")
set(_zerofg_shader_headers)

# The engine's shaders, compiled for a Vulkan 1.3 environment.
set(_zerofg_rc3_shader_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rc3")
set(_zerofg_rc3_shader_output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/rc3")
function(zerofg_compile_rc3_shader output_name source_name symbol)
  set(_output "${_zerofg_rc3_shader_output_dir}/${output_name}.h")
  add_custom_command(
    OUTPUT "${_output}"
    COMMAND "${Python3_EXECUTABLE}" "${_zerofg_shader_compiler}"
            "${_zerofg_rc3_shader_source_dir}/${source_name}"
            "${_output}" "${symbol}" ${ARGN}
    DEPENDS "${_zerofg_rc3_shader_source_dir}/${source_name}"
            "${_zerofg_rc3_shader_source_dir}/rc3_common.glsl"
            "${_zerofg_rc3_shader_source_dir}/d4c_common.glsl"
            "${_zerofg_shader_compiler}"
    VERBATIM
    COMMENT "Compiling ZeroFG ${output_name} for Vulkan 1.3")
  set(_zerofg_shader_headers ${_zerofg_shader_headers} "${_output}"
      PARENT_SCOPE)
endfunction()

zerofg_compile_rc3_shader(rc3_crop crop.comp zerofg_rc3_crop_spv)
zerofg_compile_rc3_shader(rc3_d0 d0_frontend.comp zerofg_rc3_d0_spv)
zerofg_compile_rc3_shader(rc3_d0a d0a_lattice.comp zerofg_rc3_d0a_spv)
zerofg_compile_rc3_shader(rc3_d0b d0b_luma_full.comp zerofg_rc3_d0b_spv)
zerofg_compile_rc3_shader(rc3_d0p d0p_photo.comp zerofg_rc3_d0p_spv)
zerofg_compile_rc3_shader(rc3_d1 d1_evidence.comp zerofg_rc3_d1_spv)
zerofg_compile_rc3_shader(rc3_d2_h0 d2_refine.comp zerofg_rc3_d2_h0_spv
                          -DZFG_H1_ENABLED=0)
zerofg_compile_rc3_shader(rc3_d3 d3_cells.comp zerofg_rc3_d3_spv)
zerofg_compile_rc3_shader(rc3_d4 d4_fine.comp zerofg_rc3_d4_spv)
zerofg_compile_rc3_shader(rc3_d4sg d4_fine.comp zerofg_rc3_d4sg_spv
                          -DZFG_SUBGROUP=1)
zerofg_compile_rc3_shader(rc3_d4c d4c_camera.comp zerofg_rc3_d4c_spv)
zerofg_compile_rc3_shader(rc3_d4cc d4c_cost.comp zerofg_rc3_d4cc_spv)
zerofg_compile_rc3_shader(rc3_d4cp d4c_pick.comp zerofg_rc3_d4cp_spv)
zerofg_compile_rc3_shader(rc3_d5a d5a_resolve.comp zerofg_rc3_d5a_spv)
zerofg_compile_rc3_shader(rc3_d5ah d5a_resolve.comp zerofg_rc3_d5ah_spv
                          -DZFG_HALF=1)

set_source_files_properties(${_zerofg_shader_headers} PROPERTIES GENERATED TRUE)

add_library(zerofg STATIC
  src/zerofg.cc
  src/rc3/rc3_context.cc
  ${_zerofg_shader_headers}
)

target_include_directories(zerofg
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${_zerofg_rc3_shader_output_dir}
)

# While ZeroFG lives inside XenDroid, reuse Xenia's vendored Vulkan headers.
target_include_directories(zerofg
  PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/Vulkan-Headers/include
)

target_compile_features(zerofg PUBLIC cxx_std_20)
