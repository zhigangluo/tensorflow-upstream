licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [":friends"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

cc_library(
    name = "llvm_gpu_backend",
    srcs = [
        "dump_ir_pass.cc",
        "utils.cc",
    ] + if_cuda_is_configured(if_cuda([
        "nvptx_backend_lib.cc",
    ])) + if_rocm_is_configured(if_rocm([
        "amdgpu_backend_lib.cc",
    ])),
    hdrs = [
        "dump_ir_pass.h",
        "utils.h",
    ] + if_cuda_is_configured(if_cuda([
        "nvptx_backend_lib.h",
    ])) + if_rocm_is_configured(if_rocm([
        "amdgpu_backend_lib.h",
    ])),
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@llvm//:analysis",
        "@llvm//:asm_printer",
        "@llvm//:bit_reader",
        "@llvm//:bit_writer",
        "@llvm//:code_gen",
        "@llvm//:core",
        "@llvm//:instrumentation",
        "@llvm//:ipo",
        "@llvm//:ir_reader",
        "@llvm//:linker",
        "@llvm//:mc",
        "@llvm//:nvptx_code_gen",
        "@llvm//:amdgpu_code_gen",
        "@llvm//:objc_arc",
        "@llvm//:support",
        "@llvm//:target",
        "@llvm//:transform_utils",
    ],
)

cc_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.cc"],
    data = [
        "tests_data/saxpy.ll",
    ],
    deps = [
        ":llvm_gpu_backend",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm//:core",
        "@llvm//:support",
    ],
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
