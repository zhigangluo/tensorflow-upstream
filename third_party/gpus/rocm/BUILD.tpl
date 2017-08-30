licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])


config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_headers",
    hdrs = [
        "rocm/rocm_config.h",
        %{rocm_headers}
    ],
    includes = [
        ".",
        "rocm/include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocmrt_static",
    srcs = ["rocm/lib/%{rocmrt_static_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkopts = select({
        ":freebsd": [],
        "//conditions:default": ["-ldl"],
    }) + [
        "-lpthread",
        %{rocmrt_static_linkopt}
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_driver",
    srcs = ["rocm/lib/%{rocm_driver_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocmrt",
    srcs = ["rocm/lib/%{rocmrt_lib}"],
    data = ["rocm/lib/%{rocmrt_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocblas",
    srcs = ["rocm/lib/%{rocblas_lib}"],
    data = ["rocm/lib/%{rocblas_lib}"],
    includes = [
        ".",
        "rocm/include",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)


cc_library(
    name = "rocm",
    visibility = ["//visibility:public"],
    deps = [
        ":rocblas",
        ":rocm_headers",
        ":rocmrt",
    ],
)

#
#cc_library(
#    name = "libdevice_root",
#    data = [":rocm-nvvm"],
#    visibility = ["//visibility:public"],
#)

%{rocm_include_genrules}
