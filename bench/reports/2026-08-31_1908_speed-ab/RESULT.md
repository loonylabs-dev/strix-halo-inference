# gdn-fork — the RDNA 3.5 tuning fork buys nothing at the operating point

The 49 commits of gaetan-puleo/llama-cpp-strix-halo (GDN blocks, D256 FA,
expert MMQ), rebased conflict-free onto the production base so the tuning is
the only variable, measured at the serving geometry (-ub 512 -b 2048):
+6.0 % cold prefill inside a 10 % round spread, +1.9 % at d32768 inside 9 %,
decode unchanged. At depth — the operating point — that is no difference.
Decision per the pre-registered rule: no adoption; watch that ecosystem's
upstream PRs and re-measure on merge.

model: `@MODELS@/Qwen3.8-27B-UD-Q4_K_XL.gguf`

- **reference**: `@HOME@/llama.cpp/build-rocm-patched/bin/llama-bench`, build `b10702-11-gc799f1014`
  - cmake: `-DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`
- **variant**: `@HOME@/llama.cpp/build-rocm-gdnfork-b10711-49-g19790c073/bin/llama-bench`, build `b10711-49-g19790c073`
  - cmake: `-DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`

```
                      reference      variant     change
tg64 @ d0               9.40 t/s      9.45 t/s      +0.6 %
pp2048 @ d0           252.65 t/s    267.75 t/s      +6.0 %
tg64 @ d32768           8.59 t/s      8.21 t/s      -4.4 %
pp2048 @ d32768       160.25 t/s    163.34 t/s      +1.9 %

medians of 2 interleaved rounds; every round ran both arms.
A difference smaller than the spread between rounds is not a
difference. The per-round values are in the JSON beside this.
```
