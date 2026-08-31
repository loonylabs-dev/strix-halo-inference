# unroll-flag

model: `/mnt/shared/LLM/Qwen3.8-27B-UD-Q4_K_XL.gguf`

- **reference**: `/home/martinloreck/llama.cpp/build-rocm-patched/bin/llama-bench`, build `b10702-11-gc799f1014`
  - cmake: `-DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`
- **variant**: `/home/martinloreck/llama.cpp/build-rocm-altsdk-b10711-2-gc799f1014/bin/llama-bench`, build `b10711-2-gc799f1014`
  - cmake: `-DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_HIP_COMPILER=/home/martinloreck/rocm-sdks/rocm-10.1.0a20260830/llvm/bin/clang -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_HIP_FLAGS=--rocm-path=/home/martinloreck/rocm-sdks/rocm-10.1.0a20260830 -isystem /home/martinloreck/rocm-sdks/rocm-10.1.0a20260830/include -DCMAKE_PREFIX_PATH=/home/martinloreck/rocm-sdks/rocm-10.1.0a20260830 -DROCM_PATH=/home/martinloreck/rocm-sdks/rocm-10.1.0a20260830`

```
                      reference       unroll     change
tg128 @ d0              9.07 t/s      9.97 t/s      +9.9 %
pp512 @ d0            252.52 t/s    246.52 t/s      -2.4 %
tg128 @ d16384          8.65 t/s      9.27 t/s      +7.1 %
pp512 @ d16384        201.70 t/s    196.15 t/s      -2.8 %
tg128 @ d32768          8.29 t/s      8.62 t/s      +3.9 %
pp512 @ d32768        161.28 t/s    157.78 t/s      -2.2 %
tg128 @ d65536          7.65 t/s      7.56 t/s      -1.2 %
pp512 @ d65536        115.35 t/s    113.22 t/s      -1.8 %

medians of 4 interleaved rounds; every round ran both arms.
A difference smaller than the spread between rounds is not a
difference. The per-round values are in the JSON beside this.
```
