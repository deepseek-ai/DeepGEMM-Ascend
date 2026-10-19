# DeepGEMM Ascend

DeepGEMM Ascend is a port of DeepGEMM to the HUAWEI Ascend platform. It is fully API-compatible with DeepGEMM and supports BF16, FP8, FP4 GEMM, MQA logits, and MegaMoE. On Ascend platforms, users can simply install the package and use the same APIs and development workflow as DeepGEMM on other supported platforms.

DeepGEMM Ascend provides a lightweight abstraction over the Ascend MAD (matrix multiply-add) primitives, hiding much of the complexity associated with fractal layouts, alignment constraints, address calculations, and verbose low-level parameters. This enables GEMM kernels to remain both concise and efficient. DeepGEMM Ascend makes extensive use of Ascend-specific optimization techniques, such as sparse data loading and coroutine-based pipelining, to approach the performance limits of the Ascend hardware. These implementations can also serve as references for extreme performance optimization on the Ascend platform.

Despite its lightweight codebase, DeepGEMM Ascend can achieve peak hardware performance across a wide range of matrix shapes.

DeepGEMM Ascend 是华为昇腾平台上的 DeepGEMM 实现。它完全兼容 DeepGEMM 的 API，支持 BF16、FP8、FP4 GEMM、MQA logits 和 MegaMoE 算子。DeepGEMM Ascend 和 DeepGEMM 使用相同的包名，用户只需安装当前包，即可沿用其他平台上 DeepGEMM 的 API 和开发流程。

DeepGEMM Ascend 对昇腾平台的矩阵乘加原语（MAD）提供了一个轻量的抽象，能够隐藏分形矩阵布局、对齐约束、地址计算、参数转换等细节，让 GEMM kernel 的实现保持简洁高效。DeepGEMM Ascend 广泛采用昇腾平台特有的优化技术，包括稀疏数据加载、基于协程的流水线等，以接近硬件的性能极限。这些实现也可以作为昇腾平台极致性能优化的参考。

DeepGEMM Ascend 代码轻量，并且在多种矩阵形状上能达到硬件极限性能。

## News

- 2026.09.30: Initial release of DeepGEMM Ascend, with support for Ascend 950 devices. The kernels are designed to achieve near-peak hardware performance.

## Quick Start

### Requirements

- HUAWEI Ascend NPU (developed and validated on the Ascend 950 series)
- CANN 9.20 toolkit providing `bin/bisheng`, `bin/ld.lld`
- Torch NPU package `torch_npu`
- Python 3.10 or higher
- Compilers and standard libraries with C++20 `<format>` support
- `tilelang`, used by the HC prenorm kernel (declared as a package dependency)
- `tree-sitter` and `tree-sitter-cpp`, used to generate Python type stubs when building from source

### Development

```bash
# Submodule must be cloned
git clone --recursive https://github.com/deepseek-ai/DeepGEMM-Ascend.git
cd DeepGEMM-Ascend

# Link some essential includes and build the C++ extension
cat develop.sh
./develop.sh
```

### Installation

```bash
pip install . --no-build-isolation
```

## Interfaces

### Kernel Interface

Please refer to [DeepGEMM's interfaces](https://github.com/deepseek-ai/DeepGEMM#interfaces) for the kernel APIs.

> [!NOTE]
> The scaling factor format on Ascend differs from NVIDIA's: each pair of UE8M0 scaling factors along the K dimension is packed into an `int16`, and the packed values are stored in MN-major order for optimal hardware efficiency.

Dense BF16, FP8, FP4 and mixed FP8×FP4 GEMMs accept `c` with shape `[M, N]` or `[1, N]`
and the same dtype as `d`. Full C uses atomic add. A contiguous `[1, N]` bias is loaded
and added in the AIV epilogue before the output cast.
The `bf16_gemm_nn` and `fp8_fp4_gemm_nn` interfaces take `A[M, K]` and `B[K, N]`:

```python
deep_gemm.bf16_gemm_nn(a, b, c=bias, d=d)
```

### Utilities

The library provides some utility functions besides the above kernels:

- `deep_gemm.set_num_sms` / `get_num_sms`: set/get the number of AI cores the kernels may use
- `deep_gemm.set_npu_arch` / `get_npu_arch`: override/query the `dav-*` NPU architecture used for JIT compilation, `0` queries the device
- `deep_gemm.set_mk_alignment_for_contiguous_layout` / `get_mk_alignment_for_contiguous_layout`: set/get the group-level M/K alignment for contiguous layout
- `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout`: get the theoretical minimum M/K alignment
- `deep_gemm.use_deterministic_algorithms` / `get_deterministic_algorithms`: enable/disable deterministic algorithms
- `deep_gemm.transform_sf_into_required_layout`: transform scaling factors into the required layout
- `deep_gemm.transform_k_grouped_sf_into_required_layout`: transform K-grouped scaling factors into the required layout
- `deep_gemm.get_paged_mqa_logits_metadata`: build the scheduling metadata for the paged MQA kernels
- `deep_gemm.aclnn_fp8_fp4_gemm_{nt, nn, tn, tt}` and `deep_gemm.aclnn_bf16_gemm_{nt, nn, tn, tt}`: ACLNN reference GEMMs used to cross-check the kernels in tests

### Environment Variables

Ascend home path is given by `ASCEND_HOME_PATH` or `ASCEND_TOOLKIT_HOME`.

Each `DG_JIT_*` variable falls back to the corresponding global `DJ_JIT_*` variable when unset, and is snapshotted when the JIT runtime is first created.

- General
    - `DG_JIT_DEBUG`: `0` or `1`, enable JIT debugging features, including compiler commands, load-time reporting, and the selected config per shape; `0` by default
    - `DG_PRINT_CONFIGS`: `0` or `1`, print the selected config for each shape, `0` by default
- JIT cache
    - `DG_JIT_CACHE_DIR`: string, cache directory (or a `:`-separated list of directories) for compiled kernels; lookup searches all paths front-to-back (first hit wins) and a cache miss compiles into the first path, `$HOME/.dj` by default
- Compiler output and artifacts
    - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print compiler and disassembler commands, `0` by default
    - `DG_JIT_PRINT_LOAD_TIME`: `0` or `1`, print kernel load time, `0` by default
    - `DG_JIT_KERNEL_DEBUG_INFO`: `0` or `1`, add Bisheng kernel debug information, `0` by default
- Debugging
    - `DG_JIT_LAUNCH_TIMEOUT`: integer, Ascend kernel launch timeout in seconds; `300` by default, `0` disables it
    - `ASCEND_LAUNCH_BLOCKING`: synchronize the device after every launch, turning asynchronous launch errors into in-line failures

## Performance

Measured on Ascend 950DT (CANN 9.20) using `bench_msprof` with cold L2. Shapes and cases follow the DeepGEMM test suite and cover inference and training workloads of the DeepSeek model series; see `tests/` for reproduction.

### Dense GEMM

Dense GEMM reaches up to **99.8%** of the hardware limit across types:

| Type | M | N | K | Latency (us) | Computation (TFLOPS) | Hardware limit (TFLOPS) | Utilization (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FP4×FP4 | 4096 | 7168 | 16384 | 565.6 | 1701 | 1730 | 98.3 |
| FP8×FP4 | 4096 | 7168 | 16384 | 1117.8 | 861 | 865 | 99.5 |
| FP8×FP8 | 4096 | 7168 | 16384 | 1117.6 | 861 | 865 | 99.5 |
| BF16×BF16 | 4096 | 7168 | 16384 | 2229.9 | 431 | 432 | 99.8 |

FP8 GEMM for inference:

| M | N | K | Latency (us) | Computation (TFLOPS) | Memory bandwidth (GB/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 2112 | 7168 | 15.0 | 258 | 1205 |
| 128 | 576 | 7168 | 10.1 | 105 | 564 |
| 128 | 24576 | 1536 | 20.6 | 468 | 2068 |
| 128 | 32768 | 512 | 10.4 | 415 | 1829 |
| 128 | 7168 | 16384 | 54.8 | 549 | 2456 |
| 128 | 4096 | 7168 | 19.0 | 397 | 1797 |
| 128 | 7168 | 2048 | 10.1 | 373 | 1670 |
| 4096 | 2112 | 7168 | 156.9 | 790 | 319 |
| 4096 | 576 | 7168 | 54.6 | 619 | 689 |
| 4096 | 24576 | 1536 | 362.2 | 854 | 137 |
| 4096 | 32768 | 512 | 164.3 | 837 | 129 |
| 4096 | 7168 | 16384 | 1117.6 | 861 | 186 |
| 4096 | 4096 | 7168 | 282.9 | 850 | 234 |
| 4096 | 7168 | 2048 | 143.4 | 838 | 181 |

### M-Grouped GEMM

Grouped GEMM for DeepSeek MoE experts (FP8×FP4, BF16 output). `#Groups` is the number of experts, and `M per group` the average tokens each receives.

| #Groups | M per group | N | K | Latency (us) | Computation (TFLOPS) | Memory bandwidth (GB/s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4 | 8192 | 6144 | 7168 | 3722.4 | 818 | 104 |
| 4 | 8192 | 7168 | 3072 | 1778.0 | 856 | 98 |
| 4 | 8192 | 4096 | 4096 | 1356.1 | 855 | 148 |
| 4 | 8192 | 4096 | 2048 | 680.6 | 852 | 148 |
| 8 | 4096 | 6144 | 7168 | 3689.4 | 831 | 136 |
| 8 | 4096 | 7168 | 3072 | 1778.9 | 862 | 130 |
| 8 | 4096 | 4096 | 4096 | 1356.5 | 861 | 180 |
| 8 | 4096 | 4096 | 2048 | 681.0 | 858 | 179 |

### Small-M grouped GEMM

For decode workloads with few tokens per expert, opt into `deep_gemm.set_mk_alignment_for_contiguous_layout(16)` **before constructing the padded inputs and cumulative group layout**. Values 16, 64, 128 and 256 are supported; the default remains 256. The setting is process-wide, so keep it consistent with the tensors being used and restore 256 before reusing layouts constructed for the default. K-grouped MX requires alignment at least 64. Weights retain their existing ND format.

Measured on Ascend 950DT. These are matched local measurements, not comparisons against README numbers collected on another software stack.

Each pair uses the same physical card, seed, logical operands, group counts, ND weight layout and BF16 output. The baseline kernel is unmodified `3a35326` with alignment 256; the candidate explicitly opts into alignment 16 and repacks the activation/group layout accordingly. We use the corrected read-only L2 sweep from [issue #11](https://github.com/deepseek-ai/DeepGEMM-Ascend/issues/11) for **both** revisions: sum a preinitialized int32 buffer twice the reported L2 size (256 MiB on this host) before each call. Time only `gemm_impl`; exclude initialization, SF conversion, L2 sweep and host overhead. Each result is the median of three 30-sample window means following ten cold warmups per window. Raw exported profiler traces and all window means were retained. Eight devices run independent cases; each baseline/candidate pair runs sequentially on its assigned device.

The three primary columns follow `tests/common.py` / `tests/utils.py`: `Latency = kernel time`, `Computation = 2*M_padded_total*N*K/time`, `Memory bandwidth = count_bytes(A, B, D)/time`. Byte counts include native packed FP4/FP8 data, the actual packed UE8M0 scale tensors passed to the API, padding and BF16 output, and exclude layout metadata as the existing test does. Both revisions use identical scale representation. GB/s is this tensor-byte estimate, not a hardware traffic counter. Since padding is intentionally reduced, nominal TFLOPS and sometimes GB/s decrease even when useful-token latency improves. The supplementary effective TFLOPS uses the same valid `M_total` on both sides. All arrows are baseline → candidate.

| Type | Groups | Valid M/group | Padded M/group | N | K | Latency (us) | Computation (TFLOPS) | Memory bandwidth (GB/s) | Speedup |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | ---: |
| FP4×FP4 | 32 | 4 | 256 → 16 | 3072 | 5120 | 159.155 → 92.520 | 1619.17 → 174.08 | 2136.3 → 2939.1 | 1.720× |
| FP4×FP4 | 32 | 4 | 256 → 16 | 5120 | 1536 | 93.126 → 53.786 | 1383.59 → 149.72 | 2408.2 → 2590.9 | 1.731× |
| FP8×FP8 | 32 | 4 | 256 → 16 | 3072 | 5120 | 307.729 → 168.994 | 837.42 → 95.31 | 1990.8 → 3106.0 | 1.821× |
| FP8×FP8 | 32 | 4 | 256 → 16 | 5120 | 1536 | 160.765 → 100.070 | 801.47 → 80.47 | 2216.8 → 2653.9 | 1.607× |
| FP8×FP4 | 32 | 4 | 256 → 16 | 3072 | 5120 | 304.256 → 224.367 | 846.98 → 71.78 | 1186.4 → 1217.8 | 1.356× |
| FP8×FP4 | 32 | 4 | 256 → 16 | 5120 | 1536 | 155.131 → 118.272 | 830.58 → 68.09 | 1486.2 → 1181.6 | 1.312× |
| FP4×FP4 | 32 | 1 | 256 → 16 | 3072 | 5120 | 159.107 → 92.396 | 1619.65 → 174.32 | 2136.9 → 2943.0 | 1.722× |
| FP8×FP4 | 8 | 3 | 256 → 16 | 4096 | 2048 | 45.903 → 34.223 | 748.52 → 62.75 | 1236.4 → 1080.3 | 1.341× |

The six M=4 cases are reference shapes; the final two rows are additional small-shape checks. Default-alignment large-shape checks showed −0.020% to +0.028% latency change. Ordinary GEMM tuning is not included.

Reproduce with `python tests/benchmark_small_m_grouped_gemm.py --dtype fp4 --groups 32 --m 4 --n 3072 --k 5120 --alignment 16 --output /tmp/small-m-candidate`. For the baseline, run the same script against `3a35326` with `--alignment 256`, applying only the read-sum benchmark fix to that checkout. Other paths use `--dtype fp8` or `--dtype mixed`. The script uses the repository profiler with the same read-sum sweep, 10 warmups and 30 measured calls per window, and reports both padded and effective TFLOPS. Run correctness checks with `python tests/test_small_m_grouped_gemm.py --alignment 16`, repeating for 64, 128 and 256.

### MQA Logits

MQA scoring for the DeepSeek Lightning Indexer. The kernel is FIX-pipe bound rather than compute bound, saturating the FIX pipe at 99% utilization.

| Type | Format | #Q Tokens | #K Tokens | #Heads | Dim | Latency (us) | Computation (TFLOPS) | Memory bandwidth (GB/s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Prefill | FP8 | 4096 | 8192 | 32 | 128 | 302.6 | 681 | 283 |
| Prefill | FP4 | 4096 | 8192 | 32 | 128 | 277.4 | 743 | 277 |
| Decode | FP8 | 256 | 8192 | 32 | 128 | 150.9 | 565 | 2006 |
| Decode | FP4 | 256 | 8192 | 32 | 128 | 124.2 | 708 | 1349 |

### MegaMoE

Mega MoE fuses EP dispatch, two grouped GEMMs, SwiGLU, and combine. Benchmarked over EP8 with top-k=6 (each token routed to 6 experts) and one shared expert; all values are averaged across 8 ranks.

| #Experts | Hidden | Intermediate | Tokens | Latency (us) | Computation (TFLOPS) | Memory bandwidth (GB/s) | Communication bandwidth (GB/s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 96 | 5120 | 2304 | 64 | 138.7 | 228.5 | 1850.0 | 37.5 |
| 96 | 5120 | 2304 | 256 | 225.5 | 562.5 | 1257.4 | 91.6 |
| 96 | 7168 | 3072 | 64 | 222.2 | 266.4 | 2136.9 | 32.8 |
| 96 | 7168 | 3072 | 256 | 359.2 | 659.2 | 1425.4 | 80.5 |
| 384 | 5120 | 2304 | 4096 | 2567.7 | 790.3 | 567.5 | 128.8 |
| 384 | 5120 | 2304 | 16384 | 9768.6 | 831.0 | 325.0 | 135.2 |
| 384 | 7168 | 3072 | 4096 | 4600.7 | 823.4 | 531.3 | 100.6 |
| 384 | 7168 | 3072 | 16384 | 17904.2 | 846.3 | 269.3 | 103.3 |

### HC Prenorm GEMM

Prenorm GEMM for the DeepSeek mHC (Manifold-Constrained Hyper-Connections) module, which nearly saturates the HBM write bandwidth.

| M | N | K | Latency (us) | Computation (TFLOPS) | Memory bandwidth (GB/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 13 | 24 | 28672 | 6.7 | 3 | 519 |
| 137 | 24 | 28672 | 9.6 | 20 | 1110 |
| 512 | 24 | 28672 | 15.4 | 46 | 2082 |
| 4096 | 24 | 28672 | 72.7 | 78 | 3276 |
| 8192 | 24 | 28672 | 136.7 | 82 | 3463 |

## Contributors

* Project Leads: Kexing Zhou, Zhean Xu, Chenggang Zhao
* GEMM Kernels: Kexing Zhou, Zhean Xu, Yunfan Xiao, Yuhao Meng
* MQA Logits: Anyi Xu, Zhean Xu, Kaifeng Chen
* mHC Kernel: Yuxuan Zhou, Chenggang Zhao, Ruifan Xu, Huanqi Cao, Chenhao Xu
* MegaMoE: Zhean Xu
* SF Layout Kernels: Guanglin Li
* Infrastructure: Kexing Zhou, Kuai Yu

## Acknowledgements

DeepGEMM-Ascend follows the design of the upstream [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) project, which is inspired by [CUTLASS](https://github.com/nvidia/cutlass). The JIT runtime is provided by [DeepJIT](https://github.com/deepseek-ai/DeepJIT). The mHC kernel is backed by [Tilelang](https://github.com/tile-ai/tilelang). We sincerely thank the developers of these projects for their contributions, and gratefully acknowledge Huawei for its technical support and engineering expertise throughout the development of DeepGEMM-Ascend.

## License

This code repository is released under the [MIT License](LICENSE).

## Citation

If you use DeepGEMM-Ascend in your work, please cite:

```bibtex
@misc{deepgemm_ascend2026,
  title     = {DeepGEMM-Ascend: Clean and Efficient BLAS Kernel Library on Ascend NPU},
  author    = {Kexing Zhou and Zhean Xu and Anyi Xu and Chenggang Zhao and
               Yuxuan Zhou and Yunfan Xiao and Guanglin Li and Kaifeng Chen and Yuhao Meng and
               Huanqi Cao and Ruifan Xu and Chenhao Xu and Kuai Yu},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/deepseek-ai/DeepGEMM-Ascend}
}
```
