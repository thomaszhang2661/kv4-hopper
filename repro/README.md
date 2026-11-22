# Reproduction map for `paper/main.tex` (11-19; additions 11-21)

For every table and figure of the paper: what it shows, which result files and analysis script it uses, and the
Slurm jobs that produced its numbers, with the exact `sbatch` line as Slurm recorded it (`sacct -o SubmitLine`,
queried 11-19 from the login node). This file is a map, not a script. Anything not found is marked **UNKNOWN**.

Sources: LaTeX comments in `paper/tab_baselines.tex` (`paper/main.tex` itself has no job ids in comments),
our internal audit notes and experiment log (11-10 to 11-19), `paper/make_figures.py`, `sacct`, and the first lines
of the job logs in `~/kv4-hopper/logs` on the cluster (GPU name, driver, `KV4_COMMIT`, resolved FlashInfer backend).

## How to read the job entries

- Each entry gives job id, Slurm job name, node, final state, the
  `KV4_COMMIT` label, and the SubmitLine verbatim (relative paths are relative to `~/kv4-hopper` on the cluster).
- **`KV4_COMMIT` is a label passed by hand, not a checkout.** The job ran whatever code had been synced to
  `~/kv4-hopper` when it started. One known mismatch: 11818189 says `08827d7` but ran `d0cfcd3` (our log, 11-11).
- **Commit hashes.** `KV4_COMMIT` labels are commits of our development repository, whose history is not
  published; this repository starts from a fresh history. The scripts here are the ones the paper's jobs used,
  with the later changes listed in the next item.
- **Slurm stores the batch script at submit time**; the Python it calls is read when the job starts. Scripts that changed after the jobs here:
  `vllm_serve.sbatch` (last change `b01164e`, 11-19), `vllm_bench.sbatch` (`085345e`, 11-18), `vllm_kv4.sbatch`
  (`f03f2e8` = old `5cba9b1`, 11-17).
- **`--export=ALL` also passes the submitting shell's environment**, which the SubmitLine does not show. This
  matters at least once (decode-led throughput, `ABACKEND=FLASHINFER`; see the e2e section).
- **Accounts.** Slurm account flags (`-A ...`) are removed from the lines below; add your own.
- **Job ids below 1000** (106-488, 11-10 14:07-14:19) are real Slurm ids on Explorer (logs `kv4bd_b32-287.out`
  etc.), not truncated.
- **Where results live.** Every result file the paper uses is in `results/` (11-19); raw `vllm bench serve` JSONs are gitignored (in the local working tree and on the cluster), only
  `results/serve_goodput_*.csv` and `results/vllm_serve_x/metrics_*.csv` are committed.

## Environments on the cluster

| Env | Path | Used by | Versions (where stated) |
|---|---|---|---|
| `kv4base` | `/scratch/$USER/envs/kv4base` (venv) | kernel benchmarks: `final_table`, `decode_fp4`, `decode_int4`, `xqa_compare`, `ablation_fp4`, `nvbit`, `kernel_checks`, `run_base`, `build_ext`, `build_bitdecoding`, `gpu_tests` (part 1) | built by `slurm/setup_baselines.sbatch` (jobs 11765866, 11766404): torch `2.14.1+cu126`, Triton 3.8.0, FlashInfer 0.7.3 (`flashinfer-python`, unpinned at build time; 0.7.3 per `logs/kv4setup-11766404.out` and CSV tags), Python 3.11; `module load cuda/12.8.0` (nvcc/ptxas 12.8 for our extensions, Triton and FlashInfer JIT; `-gencode arch=compute_90a,code=sm_90a`). BitDecoding `hopper` branch `584d0d3` + CUTLASS `3fe62887` built into it (`slurm/build_bitdecoding.sbatch`, job 285). NVBit 1.8.1 (`slurm/build_nvbit.sbatch`, job 11804022, cuda 12.8 + cmake 3.30.2) |
| `vllm` | `/scratch/$USER/envs/vllm` (venv) | everything inside vLLM: `vllm_kv4`, `vllm_bench`, `vllm_bench_paper`, `vllm_serve`, `vllm_smoke`, `pr67789_check`, `gpu_tests` (part 2) | `slurm/setup_envs.sbatch vllm` (job 11827621) + `kv4plugin` (job 11828413, editable `integrations/vllm_plugin`): vLLM 0.33.0 (PyPI cu130 wheel, no git commit), torch 2.14.0, Triton 3.8.0, transformers 5.20.2, flashinfer-python 0.7.2.post1 (from `logs/kv4env-11827621.out`); `module load cuda/13.2.0` (plugin JIT with nvcc 13.2, `TORCH_CUDA_ARCH_LIST=9.0a`); CUDA 13 forward-compat libs from `/scratch/$USER/envs/cuda13compat` (`setup_envs.sbatch compat`, job 11827834, conda-forge `cuda-compat=13.0.2`) on driver 570 |
| `fp4` | `~/.conda/envs/fp4` (conda) | offline accuracy: `accuracy.sbatch` (KL / perplexity, calibration `bench/calib_kv.py`), `acc_models`, `qwen_diag`, `smooth2`, `scale_rule`, `run.sbatch`, `e2e`, `hf_download` | torch 2.6 + Triton (`slurm/run.sbatch` comment), transformers 5.21 (`slurm/accuracy.sbatch` header); exact pins **UNKNOWN** (env from the earlier FP4 project, no build script in this repo) |
| `foscar` | `/scratch/$USER/envs/foscar` | `foscar.sbatch` | `setup_envs.sbatch foscar` (job 11827620): torch 2.14.1 cu126, FOSCAR at commit `9089e1f` (editable, `[cu12,bench]`) |
| `lmdeploy` | `/scratch/$USER/envs/lmdeploy` | `lmdeploy_bench.sbatch` | `setup_envs.sbatch lmdeploy` (jobs 11865707, 11866077): torch 2.11.0 cu128, torchvision 0.26.0, LMDeploy 0.18.0 |

Models and datasets come from `/scratch/$USER/hf_cache` with `HF_HUB_OFFLINE=1`. The in-vLLM recipe needs the
static K-mean files `results/calib/<model>.pt` (see "Calibration files" at the end).

## GPU rule

- **Speed numbers only on H200 SXM**: nodes d4052-d4055 (`nvidia-smi`: `NVIDIA H200, 570.86.15`). **Never
  d4077**, which is H200 NVL (`NVIDIA H200 NVL, 595.58.03`) and slower. Every speed job submits with
  `--exclude=d4077`.
- Partition names do not fix the GPU type: `-p sharing` jobs landed on SXM nodes too (11979594, 11983773,
  11987348, ...), and a speed job without `--exclude=d4077` landed on d4077 (11844434). Check `NodeList` after
  the run.
- Both configurations of a speed ratio come from one job on one node, except where noted below.
- H100 numbers: d4041 (`NVIDIA H100 80GB HBM3, 545.23.08`, partition `sharing`); FlashInfer does not load there.
- Accuracy runs used both SXM and NVL; node type is listed per job but does not enter accuracy comparisons.

---

## Table `tab:formats` - formats and bytes per 128-value row

Definitions only, no measurement. Source: `kv4/formats.py`; byte counts also in `bench/final_table.py`
(`record`). No job.

## Table `tab:nvbit` - NVBit instruction counts (B = 32, S = 16K, H200)

- Results: `results/nvbit_count_H200.txt` (written by `bench/nvbit_summary.py`, which `slurm/nvbit.sbatch` runs at
  the end over `logs/nvbit/*.log`). Bandwidth column ("measured fractions from the main-table CSVs"):
  `results/bench_flashinfer_H200_p32.csv` (FlashInfer NVFP4 and our Triton NVFP4, rows `ours_triton3.8.0`) and
  `results/bench_bitdecoding_H200_b32.csv` (BitDecoding).
- Implied issue utilization = instructions per byte x bandwidth fraction / 6.97 (`bench/nvbit_summary.py`).

- **11804022** (kv4nvbitbuild) - NVBit build (prerequisite). Node c0678, COMPLETED, KV4_COMMIT not set.
  ```
  sbatch --parsable -J kv4nvbitbuild slurm/build_nvbit.sbatch
  ```
- **11804023** (kv4nvbit_ours) - counts for our Triton kernels (row "Our first Triton NVFP4"). Node d4053, COMPLETED, KV4_COMMIT `bf04088`.
  ```
  sbatch --parsable --export=ALL,KV4_COMMIT=bf04088 -J kv4nvbit_ours -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --dependency=afterok:11804022 slurm/nvbit.sbatch ours_bf16 ours_fp8 ours_int4_tok ours_int4a_tok ours_nvfp4 ours_mxfp4
  ```
- **11804024** (kv4nvbit_ext) - counts for FlashInfer and BitDecoding (rows 2 and 3). Node d4053, COMPLETED, KV4_COMMIT `bf04088`.
  ```
  sbatch --parsable --export=ALL,KV4_COMMIT=bf04088 -J kv4nvbit_ext -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --dependency=afterok:11804022 slurm/nvbit.sbatch fi_bf16 fi_fp8 fi_nvfp4 bd_int4
  ```
- **11766914** (kv4fiH200a) - bandwidth of FlashInfer NVFP4 and our Triton NVFP4 at B = 32 (`bench_flashinfer_H200_p32.csv`, commit column `e68f610`). Node d4053, COMPLETED, KV4_COMMIT not set.
  ```
  sbatch --export=ALL -J kv4fiH200a -p gpu-short,gpu --gres=gpu:h200:1 --mem=32G -t 30 slurm/run_base.sbatch bench/bench_flashinfer.py --batch 32 --tag p32
  ```
- **287** (kv4bd_b32) - bandwidth of BitDecoding at B = 32 (`bench_bitdecoding_H200_b32.csv`, commit column `94edfd9`). Node d4054, COMPLETED, KV4_COMMIT not set.
  ```
  sbatch --export=ALL --dependency=afterok:285 --exclude=d4077 -J kv4bd_b32 -p gpu-short,gpu --gres=gpu:h200:1 --mem=32G -t 30 slurm/run_base.sbatch bench/bench_bitdecoding.py --batch 32 --tag b32
  ```
- **285** (kv4bdbuild) - BitDecoding build (prerequisite of 287). Node c0683, COMPLETED, KV4_COMMIT not set.
  ```
  sbatch --parsable --export=ALL slurm/build_bitdecoding.sbatch
  ```

The text's "3.8 instructions per value for FP4 against 2.6 for INT4" (Section 4, an earlier build):

- **11811702** (kv4nvbit_cuda) - NVBit of the CUDA kernels at commit eb90677. Node d4054, COMPLETED, KV4_COMMIT `eb90677`.
  ```
  sbatch --parsable -J kv4nvbit_cuda -p gpu-short,gpu,gpu-interactive --gres=gpu:h200:1 --exclude=d4077 -t 00:30:00 --export=ALL,KV4_COMMIT=eb90677 slurm/nvbit.sbatch cuda_int4a cuda_nvfp4 cuda_mxfp4
  ```

## Figure `fig:pipeline` - kernel structure

TikZ drawing in `paper/main.tex`; no data, no job.

## Figure `fig:kernel` and Table `tab:kernel` - speed-up over FlashInfer FA2 FP8, 11 shapes

- Results: `results/final_table_H200_v1.csv` (written directly by `bench/final_table.py`; times include the split
  merge; our split = best of 1-64 per shape). Figure: `paper/make_figures.py` `kernel()` reads the same CSV
  (FlashInfer BF16, FlashInfer NVFP4, ours NVFP4, ours INT4 against `flashinfer_fp8`). The table shows 6 of the 11
  shapes.
- One job, one GPU (log `logs/kv4final-11812198.out`: `NVIDIA H200, 570.86.15`).

- **11812198** (kv4final) - every cell of the table and every bar of the figure. Node d4052, COMPLETED, KV4_COMMIT `e44d7b6`.
  ```
  sbatch --parsable -J kv4final -p gpu-short,gpu,gpu-interactive --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=e44d7b6 slurm/final_table.sbatch --tag v1
  ```

Numbers in the bullets under the tables (Section 5) come from other jobs:

- **11812087** (kv4fp4) - NVFP4 69% / 76% kernel-only bandwidth, no merge (`decode_fp4_H200_v5sleep.csv`); the paper now quotes 67-75% from 11812198. Node d4052, COMPLETED, KV4_COMMIT `6bb450c`.
  ```
  sbatch --parsable -J kv4fp4 -p gpu-short,gpu,gpu-interactive --gres=gpu:h200:1 --exclude=d4077 -t 00:40:00 --export=ALL,KV4_COMMIT=6bb450c slurm/decode_fp4.sbatch --tag v5sleep --shapes 32x16384,128x16384,8x65536 --splits 1,2,4 --no-baselines
  ```
- **11812098** (kv4fp4) - same, `_v6unroll.csv`. Node d4054, COMPLETED, KV4_COMMIT `7c86ad7`.
  ```
  sbatch --parsable -J kv4fp4 -p gpu-short,gpu,gpu-interactive --gres=gpu:h200:1 --exclude=d4077 -t 00:40:00 --export=ALL,KV4_COMMIT=7c86ad7 slurm/decode_fp4.sbatch --tag v6unroll --shapes 32x16384,128x16384,8x65536 --splits 1,2,4 --no-baselines
  ```
- **11811692** (kv4dec) - INT4 kernel-only bandwidth (`decode_int4_H200_v4.csv`). Node d4054, COMPLETED, KV4_COMMIT `3d6183e`.
  ```
  sbatch --parsable -J kv4dec -p gpu-short,gpu,gpu-interactive --gres=gpu:h200:1 --exclude=d4077 -t 00:45:00 --export=ALL,KV4_COMMIT=3d6183e slurm/decode_int4.sbatch --tag v4
  ```
- **11812802** (kv4final_h100) - H100: NVFP4 79-86%, INT4 82-90% at B >= 8 (`final_table_H100_v1_h100.csv`). Node d4041, COMPLETED, KV4_COMMIT `5e07a78`.
  ```
  sbatch --parsable -J kv4final_h100 -p sharing --gres=gpu:h100:1 -t 00:50:00 --export=ALL,KV4_COMMIT=5e07a78 slurm/final_table.sbatch --tag v1_h100 --no-flashinfer --shapes 1x16384,1x65536,8x65536,32x16384,64x16384,128x16384
  ```
- **11827870** (kv4rcp) - full recipe 1.77-2.17x of FP8 under CUDA graphs, sink merge + q rotation cost (`final_table_recipe_H200_v2.csv`; `_v1.csv` is byte-identical, see kernel audit). Node d4053, COMPLETED, KV4_COMMIT `ef3d6ea`.
  ```
  sbatch -J kv4rcp -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:50:00 --export=ALL,KV4_COMMIT=ef3d6ea,KV4_SCRIPTS=bench/final_table_recipe.py slurm/kernel_checks.sbatch
  ```
- **11854031** (kv4t53) - rerun of the 16x128K +9% under graphs, and the PR #53060 kernel-level numbers (`final_table_recipe_H200_pr53060.csv`). Node d4054, COMPLETED, KV4_COMMIT `b8c9517`.
  ```
  sbatch -J kv4t53 -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 01:30:00 --export=ALL,KV4_COMMIT=b8c9517 slurm/run_base.sbatch bench/final_table_recipe.py --tag pr53060 --pr53060 --no-int4
  ```
- **11827635** (kv4fos) - FOSCAR comparison on H200, 15 shapes (`foscar_compare_H200_v1.csv`; ran the old unfused q preparation). Node d4052, COMPLETED, KV4_COMMIT `e236e3a`.
  ```
  sbatch -J kv4fos -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --dependency=afterok:11827620 --export=ALL,KV4_COMMIT=e236e3a slurm/foscar.sbatch
  ```
- **11827620** (kv4env) - FOSCAR env build (prerequisite of 11827635, dependency). Node c0631, COMPLETED, KV4_COMMIT not set.
  ```
  sbatch -J kv4env slurm/setup_envs.sbatch foscar
  ```
- **11829653** (kv4fos) - FOSCAR comparison on H100 (`foscar_compare_H100_v1_h100.csv`). Node d4041, COMPLETED, KV4_COMMIT `b2b40b1`.
  ```
  sbatch -J kv4fos -p sharing --gres=gpu:h100:1 -t 01:00:00 --export=ALL,KV4_COMMIT=b2b40b1 slurm/foscar.sbatch --tag v1_h100
  ```
- **11813785** (kv4paged) - paged = contiguous speed, 0.99-1.035x (no CSV; log only). Node d4054, COMPLETED, KV4_COMMIT `38ea5c0`.
  ```
  sbatch --parsable -J kv4paged -p gpu-short,gpu,gpu-interactive --gres=gpu:h200:1 --exclude=d4077 --mem=32G -t 00:20:00 --export=ALL,KV4_COMMIT=38ea5c0,KV4_SCRIPT=bench/paged_check.py slurm/decode_fp4.sbatch
  ```
- **11811205** (kv4dec_h100) - numerical pitfall (1): 1024-offset error 1.6-2.2e-2 on random data (FAILED is the expected outcome: the check failed). Node d4041, FAILED, KV4_COMMIT `9fdc237`.
  ```
  sbatch --parsable -J kv4dec_h100 -p sharing --gres=gpu:h100:1 --export=ALL,KV4_COMMIT=9fdc237 slurm/decode_int4.sbatch --tag v1_h100
  ```
- **11811604** (kv4dec_h100) - pitfall (1) after the fix: 1.3-2.3e-4. Node d4041, COMPLETED, KV4_COMMIT `e36eca6`.
  ```
  sbatch --parsable -J kv4dec_h100 -p sharing --gres=gpu:h100:1 --export=ALL,KV4_COMMIT=e36eca6 slurm/decode_int4.sbatch --tag v4_h100 --no-baselines --shapes 32x16384
  ```

Setup text, "the fixed rule picks the same split as the per-shape search at 9 of 11 shapes": derived by comparing
the best split counts of 11812198 with the rule `min(16, floor(264/(B*Hkv)))`. The dedicated fixed-rule job never
ran: 11967173 and 11972579 were cancelled (GPU yield, 11-18) and 11988677 is still PENDING:

- **11988677** (kv4ftab) - `final_table.py --rule` (pending; not a source of any paper number). Node None assigned, PENDING, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4ftab -p gpu --gres=gpu:h200:1 --exclude=d4077 -t 01:20:00 --export=ALL,KV4_COMMIT=d31691c slurm/final_table.sbatch --tag rule --rule
  ```

## Table `tab:xqa` - against FlashInfer's XQA FP8 kernel, 11 shapes

- Results: `results/xqa_compare_H200_v1.csv` (written by `bench/xqa_compare.py`; XQA at pages 16/32/64/128, best
  page shown; ours = `cuda_nvfp4` best split, `cuda_nvfp4_rule` = fixed rule, used for "within 4% except 1x128K").
  No separate analysis script; the table takes the minimum XQA time per shape.

- **11979307** (kv4xqa) - every cell (FA2 FP8, XQA FP8 per page, ours best split and fixed rule, XQA-vs-FA2 error). Node d4052, COMPLETED, KV4_COMMIT `ae41fa8`.
  ```
  sbatch -J kv4xqa -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:55:00 --export=ALL,KV4_COMMIT=ae41fa8 slurm/xqa_compare.sbatch --tag v1
  ```

(11979306 was an NVL quick look, cancelled before it ran; not used.)

## Table `tab:ablation` - removal ablation of our NVFP4 kernel

- Results: `results/ablation_fp4_H200_nvl1.csv` (written by `bench/ablation_fp4.py`; `vs_full` column = the
  table's ratios; copy row = device copy of the same bytes). The tag `nvl1` is only a file name: the job was
  submitted to `sharing` but ran on d4052 (H200 SXM).

- **11979594** (kv4abl) - every cell. Node d4052, COMPLETED, KV4_COMMIT `1b911cf`.
  ```
  sbatch -J kv4abl -p sharing --gres=gpu:h200:1 -t 00:45:00 --export=ALL,KV4_COMMIT=1b911cf slurm/ablation_fp4.sbatch --tag nvl1
  ```

## Table `tab:e2e` and Figure `fig:vllm` - pure decode step inside vLLM (Llama-3.1-8B)

- Results: `results/vllm_forward_xqa/Llama-3.1-8B_{fp8_fi,fp8_fi_xqa,nvfp4_fi,plugin_recipe}.json`
  (`vllm bench forward` medians, driven by `bench/vllm_forward_matrix.py`). The table uses `fp8_fi`, `nvfp4_fi`,
  `plugin_recipe`; `fp8_fi_xqa` (`VLLM_HAS_FLASHINFER_CUBIN=1`) is the "at most 0.3%" check. Figure:
  `paper/make_figures.py` `vllm()` reads the same three JSONs.
- Log `logs/kv4fwd-11979733.out`: `host d4054`, `NVIDIA H200, 570.86.15`; BACKEND lines say `decode_backend=xqa`
  for `fp8_fi` and `fp8_fi_xqa`, `flashinfer-native` for `nvfp4_fi`. The `bf16` config in this job failed
  (32x32K does not fit), see `tab:baselines`. Checked 11-19: all 45 cells match the JSONs.

- **11979733** (kv4fwd) - every cell of the table and every bar of the figure. Node d4054, COMPLETED, KV4_COMMIT `a0de09b`.
  ```
  sbatch -J kv4fwd -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --mem=96G -c 16 -t 01:45:00 --export=ALL,KV4_COMMIT=a0de09b,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --model meta-llama/Llama-3.1-8B --configs bf16,fp8_fi,fp8_fi_xqa,nvfp4_fi,plugin_recipe --out-dir results/vllm_forward_xqa
  ```

Calibration file used: `results/calib/Llama-3.1-8B.pt` (job 11844479, below).

Text around the table (Section 6):

- **11845880** (kv4fwd) - "mean removal and sinks cost 1.2-3.2% against NVFP4 with only layer 0 in BF16 (both in one earlier job)": `plugin_recipe` vs `plugin_nv_l0`; an older job (commit 4cfb97e), not the table's job. Node d4054, COMPLETED, KV4_COMMIT `4cfb97e`.
  ```
  sbatch -J kv4fwd -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:40:00 --export=ALL,KV4_COMMIT=4cfb97e,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --configs plugin_recipe,plugin_nv_l0,fp8_fi
  ```
- **11850998** (kv4cfwd) - Qwen3-32B speed 1.00-1.02x / 1.05-1.18x. Node d4052, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cfwd -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --mem=96G -c 16 -t 01:30:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/Qwen3-32B.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-32B --max-model-len 33024 --configs fp8_fi,plugin_recipe --cases b1_16k=1*1@16384 b1_32k=1*1@32768 b8_16k=8*1@16384 b8_32k=8*1@32768 b16_16k=16*1@16384 b32_8k=32*1@8192 b64_4k=64*1@4096
  ```
- **11851106** (kv4cfwd) - Qwen3-30B-A3B speed 0.94-0.95x / 1.11-1.32x. Node d4052, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cfwd -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --mem=96G -c 16 -t 01:30:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/Qwen3-30B-A3B.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-30B-A3B --max-model-len 33024 --configs fp8_fi,plugin_recipe --cases b1_16k=1*1@16384 b1_32k=1*1@32768 b8_32k=8*1@32768 b32_16k=32*1@16384 b32_32k=32*1@32768 b64_16k=64*1@16384 b128_8k=128*1@8192
  ```
- **11853196** (kv4cfwd) - Qwen3-30B-A3B recheck at B = 1 (0.946 / 0.952). Node d4055, COMPLETED, KV4_COMMIT `36137ae`.
  ```
  sbatch -J kv4cfwd -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --mem=96G -c 16 -t 01:00:00 --export=ALL,KV4_COMMIT=36137ae,KV4_CALIB=results/calib/Qwen3-30B-A3B.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-30B-A3B --max-model-len 33024 --configs fp8_fi,plugin_recipe,plugin_recipe_s32 --out-dir results/vllm_forward_s32 --cases b1_16k=1*1@16384 b1_32k=1*1@32768 b8_32k=8*1@32768
  ```
- **11851200** (kv4vb) - end-to-end TPOT / TTFT / offline throughput (8K and 16K in, 512 out), recipe side; KV capacity 2,867,200. Node d4052, COMPLETED, KV4_COMMIT `fd375df`.
  ```
  sbatch -J kv4vb -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:59:00 --export=ALL,KV4_VLLM=1,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_FORMAT=nvfp4,KV4_KMEAN=1,KV4_SINKS=1,KV4_BOUNDARY=0,KV4_COMMIT=fd375df,RTAG=_sxm_recipe3 slurm/vllm_bench.sbatch ultraquant_4bit
  ```
- **11851201** (kv4vb) - same, FlashInfer FP8 side; KV capacity 1,761,600 (two consecutive jobs on one node, d4052). Node d4052, COMPLETED, KV4_COMMIT `fd375df`.
  ```
  sbatch -J kv4vb -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:59:00 --export=ALL,ABACKEND=FLASHINFER,KV4_COMMIT=fd375df,RTAG=_sxm_FI3 slurm/vllm_bench.sbatch fp8
  ```
- **11848187** (kv4vb) - decode-led throughput (16K in, 2048 out), recipe side: 16,804.68 tok/s. Node d4054, COMPLETED, KV4_COMMIT `1a056b8`.
  ```
  sbatch -J kv4vb -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:59:00 --export=ALL,KV4_VLLM=1,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_FORMAT=nvfp4,KV4_KMEAN=1,KV4_SINKS=1,KV4_BOUNDARY=0,KV4_COMMIT=1a056b8,RTAG=_sxm_recipe2 slurm/vllm_bench.sbatch ultraquant_4bit
  ```
- **11846510** (kv4vb) - decode-led throughput, FlashInfer FP8 side: 14,347.61 tok/s. The SubmitLine has no `ABACKEND`; FlashInfer came from the submitting shell (`--export=ALL`): the result file on the cluster is `results/vllm/throughput_Llama-3.1-8B_fp8_sxm_FLASHINFER_i16384_o2048.json` (11-13 03:57). A rerun must add `ABACKEND=FLASHINFER` to `--export`. Node d4052, COMPLETED, KV4_COMMIT `6f29121`.
  ```
  sbatch -J kv4vb -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:59:00 --export=ALL,KV4_VLLM=0,KV4_COMMIT=6f29121,RTAG=_sxm slurm/vllm_bench.sbatch fp8
  ```
- **11846746** (kv4vb) - decode-led throughput, default FA3 FP8 side: 12,038.86 tok/s (file `..._fp8_sxm_i16384_o2048.json`, 04:04). Same SubmitLine as 11846510, different node. Node d4054, COMPLETED, KV4_COMMIT `6f29121`.
  ```
  sbatch -J kv4vb -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:59:00 --export=ALL,KV4_VLLM=0,KV4_COMMIT=6f29121,RTAG=_sxm slurm/vllm_bench.sbatch fp8
  ```
- **11956067** (kv4paperC) - 64x32K: decode 38.7 vs 52.2 ms, whole request 86.8 vs 85.0 s, prefill 77.0 vs 71.7 s (one job). Node d4052, COMPLETED, KV4_COMMIT `f2b426e` (old) -> `e108e24` (new).
  ```
  sbatch -J kv4paperC -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -c 16 --mem=96G -t 01:10:00 --export=ALL,KV4_COMMIT=f2b426e,PART=C slurm/vllm_bench_paper.sbatch
  ```
- **11945784** (kv4tprof) - per-kernel profile at 8x16K: our decode kernel 1.50 ms vs 2.88 ms; attention-related 2.00 vs 3.02 ms; steps 7.55 vs 8.43 ms. Node d4052, COMPLETED, KV4_COMMIT `3508193` (old) -> `bfcf59d` (new).
  ```
  sbatch -J kv4tprof -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 00:45:00 --export=ALL,KV4_COMMIT=3508193,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/debug/vllm_decode_profile.py,PROF_MODE=diff,PROF_DIFF=33:65,PROF_S=16000,PROF_B=8 slurm/vllm_kv4.sbatch recipe,fp8_fi
  ```
- **11850980** (kv4tprof) - prefill 9.29 vs 8.66 s for 8 x 32,000 tokens; ~226 ms attention, ~230 ms copies, 72 ms dequant; write kernel 98.6 vs 97.8 ms (log not in `~/kv4-hopper/logs`; numbers from our internal log only, per the e2e audit). Node d4052, COMPLETED, KV4_COMMIT `b45c626`.
  ```
  sbatch -J kv4tprof -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:45:00 --export=ALL,KV4_COMMIT=b45c626,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/debug/vllm_decode_profile.py,PROF_MODE=prefill,PROF_S=32000,PROF_B=8 slurm/vllm_kv4.sbatch recipe,fp8_fi
  ```
- **11844434** (kv4fwd) - "FA3 FP8 30.2 vs FlashInfer FP8 21.6 ms at 32x32K" and "UltraQuant 1.9-4.6x slower" - **H200 NVL**, marked so in the paper; SubmitLine has no `--configs`, so it ran the script's defaults at commit 94e5251. Node d4077, COMPLETED, KV4_COMMIT `94e5251`.
  ```
  sbatch -J kv4fwd -p gpu-short,gpu,sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=94e5251,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch
  ```
- **11840475** (kv4ppl) - prefill-precision perplexity +1.51% vs +0.68% (old all-layer MXFP4 plugin, **H200 NVL**). Node d4077, COMPLETED, KV4_COMMIT `d0e1426`.
  ```
  sbatch -J kv4ppl -p sharing --gres=gpu:h200:1 -t 00:50:00 --export=ALL,KV4_COMMIT=d0e1426,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --configs auto,fp8,plugin_bf16,plugin_fp8,stock --windows 16 --len 8192 --chunk 2048
  ```
- **11840255** (kv4pa) - prefill attention microbenchmark (FA3 FP8 not faster than FA3 BF16; H200 NVL), backs the prefill bullet's wording. Node d4077, COMPLETED, KV4_COMMIT `20ab790`.
  ```
  sbatch -J kv4pa -p sharing --gres=gpu:h200:1 -t 00:20:00 --export=ALL,KV4_COMMIT=20ab790,KV4_SCRIPT=bench/debug/prefill_attn_micro.py slurm/vllm_kv4.sbatch
  ```

Section 9 (limitations) speed numbers that belong here:

- **11873291** (kv4tprof) - B = 1 per-kernel profile at 1x16K: 382 vs 702 us, merge 269 us, totals 848 vs 847 us, step 5.13 vs 5.40 ms (1.05x). Node d4055, COMPLETED, KV4_COMMIT `6fa8a08`.
  ```
  sbatch --parsable -J kv4tprof -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 00:45:00 --export=ALL,KV4_COMMIT=6fa8a08,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/debug/vllm_decode_profile.py,PROF_MODE=diff,PROF_S=16000,PROF_B=1 slurm/vllm_kv4.sbatch recipe,fp8_fi
  ```
- **11881741** (kv4lmd) - LMDeploy TurboMind INT4 / INT8 / BF16 (one job). Node d4054, COMPLETED, KV4_COMMIT `b5bd85f`.
  ```
  sbatch --parsable -J kv4lmd -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=b5bd85f slurm/lmdeploy_bench.sbatch
  ```
- **11886726** (kv4lmd) - LMDeploy rerun, within 1.3%. Node d4055, COMPLETED, KV4_COMMIT `add6eee`.
  ```
  sbatch --parsable -J kv4lmd -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=add6eee slurm/lmdeploy_bench.sbatch
  ```

## Table `tab:serve` - serving under load (Llama-3.1-8B, H200 SXM, FP8 with XQA decode)

- Raw: `results/vllm_serve_x/Llama-3.1-8B_{recipe,fp8_xqa}_{reason,longctx}_i*_o*_x0.5_r<rate>[_tag].json`
  (gitignored; local working tree and the cluster) and `results/vllm_serve_x/metrics_*.csv` (committed; KV use, waiting,
  preemptions from `/metrics`).
- Table numbers: `results/serve_goodput_xqa_11-19.csv`, made by `bench/analyze_serve.py` with request-bootstrap
  intervals (our log, 11-19: "analyze_serve --boot 1000"). **UNKNOWN: the exact command line and where it ran**
  (no Slurm job found; probably run locally on the JSONs; committed in 8f9f8e8 and 9799ed3). Tag column: empty =
  seed 0, `s1` = seed 1, `diag` = the seed-0 repeat, `a4` = `--api-server-count 4`.
- Every job below has both configurations (`recipe`, `fp8_xqa`) one after the other unless noted; every FP8 server
  log (`logs/serve-<job>-*`) shows `decode_backend=xqa`. Each job starts its own servers, so each row's two
  columns come from one job and one node.

- **11982322** (kv4srv) - reasoning-like 0.75 and 1.0. Node d4052, COMPLETED, KV4_COMMIT `a0de09b`.
  ```
  sbatch -J kv4srv -p gpu --gres=gpu:h200:1 --exclude=d4077 -t 02:00:00 --export=ALL,KV4_COMMIT=a0de09b,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,RATES=0.75 1 slurm/vllm_serve.sbatch
  ```
- **11988711** (kv4srv) - reasoning-like 1.25. Node d4054, COMPLETED, KV4_COMMIT `28364ea`.
  ```
  sbatch -J kv4srv -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 02:00:00 --export=ALL,KV4_COMMIT=28364ea,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,RATES=1.25 slurm/vllm_serve.sbatch
  ```
- **11989408** (kv4srv) - reasoning-like 1.35 and 1.4, seed 0 (FP8 0.89 at 1.35, 0.82 at 1.4 with 750 preemptions; ours 0.82 and 1.00). Node d4052, COMPLETED, KV4_COMMIT `8896dc4`.
  ```
  sbatch -J kv4srv -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 02:00:00 --export=ALL,KV4_COMMIT=8896dc4,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,RATES=1.35 1.4 slurm/vllm_serve.sbatch
  ```
- **11981959** (kv4srv) - reasoning-like 1.5, seed 0 (FP8 0.69, ours 0.81). Node d4053, COMPLETED, KV4_COMMIT `a0de09b`.
  ```
  sbatch -J kv4srv -p gpu --gres=gpu:h200:1 --exclude=d4077 -t 02:00:00 --export=ALL,KV4_COMMIT=a0de09b,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,RATES=1.5 slurm/vllm_serve.sbatch
  ```
- **11990246** (kv4srv) - repeat of seed 0 at 1.35 and 1.5, recipe only (`CONFIGS=recipe`, `RTAG=_diag`; ours 1.00 and 1.00). Node d4052, COMPLETED, KV4_COMMIT `4425ecd`.
  ```
  sbatch -J kv4srv -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 01:20:00 --export=ALL,KV4_COMMIT=4425ecd,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,RATES=1.35 1.5,CONFIGS=recipe,RTAG=_diag slurm/vllm_serve.sbatch
  ```
- **11990702** (kv4srv) - seed 1 at 1.35 and 1.5 (FP8 0.94 / 0.75, ours 0.86 / 1.00; API-server CPU diagnostics). Node d4053, COMPLETED, KV4_COMMIT `752b45e`.
  ```
  sbatch -J kv4srv -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 01:50:00 --export=ALL,KV4_COMMIT=752b45e,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,RATES=1.35 1.5,SEED=1,RTAG=_s1 slurm/vllm_serve.sbatch
  ```
- **11991941** (kv4srv) - "4 API servers" rows: seed 0, `APISRV=4`. Node d4053, COMPLETED, KV4_COMMIT `b01164e`.
  ```
  sbatch -J kv4srv -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 01:50:00 --export=ALL,KV4_COMMIT=b01164e,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,RATES=1.35 1.5,APISRV=4,SEED=0,RTAG=_a4 slurm/vllm_serve.sbatch
  ```
- **11981961** (kv4srv) - long context (16K in, 1K out) 0.5, 1.0, 1.5. Node d4053, COMPLETED, KV4_COMMIT `a0de09b`.
  ```
  sbatch -J kv4srv -p gpu --gres=gpu:h200:1 --exclude=d4077 -t 02:00:00 --export=ALL,KV4_COMMIT=a0de09b,W=_longctx,IN=16384,OUT=1024,RANGE=0.5,NP=150,NPT=300,RATES=0.5 1 1.5 slurm/vllm_serve.sbatch
  ```

Not used in the table: 11972580 (reasoning-like 1.25: submitted before the `fp8_xqa` default, so its FP8 side is native decode; its recipe side is valid but unused),
the 11-15/11-17 batches (FP8 fell back to FlashInfer's native decode behind the proxy). Seeds 1 and 2 with 4 API
servers (11991942, 11991943) are pending.

## Table `tab:baselines` - speed and accuracy of each KV option (`paper/tab_baselines.tex`)

Analysis: MATH-500 means and paired tests with `bench/analyze_reasoning_runs.py --boot 2000` (our log, 11-19);
the exact invocations are **UNKNOWN** (not recorded; the docstring shows the pattern). Speed ratios, perplexity
ratios and KL are read from the files directly.

**Speed columns** (speed-up over FP8 = `fp8_fi` / config, `vllm bench forward`):

- **11979733** (kv4fwd) - FP8, vLLM NVFP4 and ours (same job as `tab:e2e`); B >= 8 range over b8_16k ... b128_8k, B = 1 over b1_16k, b1_32k. Node d4054, COMPLETED, KV4_COMMIT `a0de09b`.
  ```
  sbatch -J kv4fwd -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --mem=96G -c 16 -t 01:45:00 --export=ALL,KV4_COMMIT=a0de09b,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --model meta-llama/Llama-3.1-8B --configs bf16,fp8_fi,fp8_fi_xqa,nvfp4_fi,plugin_recipe --out-dir results/vllm_forward_xqa
  ```
- **11987693** (kv4fwd) - BF16 cells (marked *): BF16 / FP8 0.974, 0.964 at B = 1; 0.864, 0.813, 0.627 at 8x16K, 8x32K, 32x16K; ours there 1.12-1.63 / 1.01-1.04. The `bf16` config exits 1 at b64_16k (cache does not fit), so **no BF16 JSON exists**: the BF16 medians are only in `logs/kv4fwd-11987693.out`; `results/vllm_forward_xqa_bf16/` holds only `fp8_fi` and `plugin_recipe`. Node d4052, COMPLETED, KV4_COMMIT `d409c20`.
  ```
  sbatch -J kv4fwd -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=96G -c 16 -t 01:00:00 --export=ALL,KV4_COMMIT=d409c20,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --model meta-llama/Llama-3.1-8B --configs bf16,fp8_fi,plugin_recipe --out-dir results/vllm_forward_xqa_bf16 --cases b1_16k=1*1@16384 b1_32k=1*1@32768 b8_16k=8*1@16384 b8_32k=8*1@32768 b32_16k=32*1@16384 b64_16k=64*1@16384 b16_64k=16*1@65536 b128_8k=128*1@8192
  ```

**MATH-500 columns.** BF16 and ours = Table `tab:invllm` (same files and jobs, see there). vLLM NVFP4 and its
layer-0 row (`nvfp4`, `nvfp4@0`, three runs each; files `results/mathruns/vllm_reasoning_*_500nv*_nvfp4*.csv`,
`..._tf500nv*`):

- **11987348** (kv4vnv) - Qwen2.5-7B-Instruct `nvfp4,nvfp4@0`, run 1 (default order). Node d4054, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes nvfp4,nvfp4@0 --n 500 --new 4096 --out results/vllm_reasoning_q25_7bi_500nv.csv
  ```
- **11987684** (kv4vnv) - Qwen2.5-7B-Instruct run 2 (shuffle 1). Node d4077, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes nvfp4,nvfp4@0 --n 500 --new 4096 --shuffle-seed 1 --out results/vllm_reasoning_q25_7bi_500nvs1.csv
  ```
- **11987689** (kv4vnv) - Qwen2.5-7B-Instruct run 3 (shuffle 2). Node d4052, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes nvfp4,nvfp4@0 --n 500 --new 4096 --shuffle-seed 2 --out results/vllm_reasoning_q25_7bi_500nvs2.csv
  ```
- **11987816** (kv4vnv) - R1-Distill-Llama-8B `nvfp4`, run 1. Node d4054, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dtypes nvfp4 --n 500 --new 4096 --out results/vllm_reasoning_r1llama8b_tf500nv.csv
  ```
- **11987851** (kv4vnv) - R1-Distill-Llama-8B run 2. Node d4077, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dtypes nvfp4 --n 500 --new 4096 --shuffle-seed 1 --out results/vllm_reasoning_r1llama8b_tf500nvs1.csv
  ```
- **11987854** (kv4vnv) - R1-Distill-Llama-8B run 3. Node d4052, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dtypes nvfp4 --n 500 --new 4096 --shuffle-seed 2 --out results/vllm_reasoning_r1llama8b_tf500nvs2.csv
  ```
- **11987889** (kv4vnv) - Phi-4 `nvfp4`, run 1. Node d4052, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model microsoft/phi-4 --dtypes nvfp4 --n 500 --new 2048 --out results/vllm_reasoning_phi-4_500nv.csv
  ```
- **11987891** (kv4vnv) - Phi-4 run 2. Node d4053, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model microsoft/phi-4 --dtypes nvfp4 --n 500 --new 2048 --shuffle-seed 1 --out results/vllm_reasoning_phi-4_500nvs1.csv
  ```
- **11987892** (kv4vnv) - Phi-4 run 3. Node d4077, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model microsoft/phi-4 --dtypes nvfp4 --n 500 --new 2048 --shuffle-seed 2 --out results/vllm_reasoning_phi-4_500nvs2.csv
  ```

FP8 row (one run each, `decode_backend=xqa`):

- **11987938** (kv4vnv) - Qwen2.5-7B-Instruct FP8 1.6% (`vllm_reasoning_q25_7bi_500fp8_fp8.csv`). Node d4077, COMPLETED, KV4_COMMIT `b571da8`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=b571da8,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes fp8 --n 500 --new 4096 --out results/vllm_reasoning_q25_7bi_500fp8.csv
  ```
- **11987943** (kv4vnv) - R1-Distill-Llama-8B FP8 67.0% (`results/mathruns/vllm_reasoning_r1llama8b_tf500fp8_fp8.csv`). Node d4077, COMPLETED, KV4_COMMIT `b571da8`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=b571da8,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dtypes fp8 --n 500 --new 4096 --out results/vllm_reasoning_r1llama8b_tf500fp8.csv
  ```
- **11987962** (kv4vnv) - Phi-4 FP8 76.0%. Its `--out` lacks the `vllm_reasoning_` prefix, so the file is `results/phi-4_500fp8_fp8.csv`. Node d4077, COMPLETED, KV4_COMMIT `b571da8`.
  ```
  sbatch -J kv4vnv -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=b571da8,KV4_ATTN_BACKEND=FLASHINFER,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model microsoft/phi-4 --dtypes fp8 --n 500 --new 2048 --out results/phi-4_500fp8.csv
  ```

**Delta-ppl columns** (vLLM, wikitext-2, 16 x 8192; results in the job logs, `RESULT` lines of
`bench/vllm_prompt_ppl.py`):

- **11983773** (kv4ppl) - Llama-3.1-8B: BF16 5.4809, FP8 +0.94%, vLLM NVFP4 +2.62%, NVFP4 layer 0 BF16 +2.56%, ours +0.06% (all rows one job). Node d4054, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4ppl -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --configs auto,fp8,nvfp4,nvfp4_l0,plugin_recipe --windows 16 --len 8192 --chunk 2048
  ```
- **11983776** (kv4ppl) - Qwen2.5-7B-Instruct: BF16 6.2692, vLLM NVFP4 1542.5 (x246), layer 0 BF16 +78%, ours +0.05%. Node d4077, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4ppl -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=4860ba7,KV4_CALIB=results/calib/Qwen2.5-7B-Instruct.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --configs auto,fp8_l0,nvfp4,nvfp4_l0,plugin_recipe --windows 16 --len 8192 --chunk 2048
  ```
- **11987940** (kv4ppl) - Qwen2.5-7B-Instruct FP8 230.7 (x36.8). Node d4077, COMPLETED, KV4_COMMIT `b571da8`.
  ```
  sbatch -J kv4ppl -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=b571da8,KV4_ATTN_BACKEND=FLASHINFER,KV4_CALIB=results/calib/Qwen2.5-7B-Instruct.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --configs auto,fp8 --windows 16 --len 8192 --chunk 2048
  ```

**KL column** (offline, FP32, CPU, 8 x 2048; cross-job noise up to ~4%):

- **11983769** (kv4base) - FP8 0.0037 and "4-bit, one scale per tensor" 0.297 (`baseB2_llama-3_1-8b_sink0.csv`). Node c0683, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4base -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=4860ba7,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 0 --configs fp8,int4t,fp4t,nvfp4_g1,nvfp4 --out results/baseB2_llama-3_1-8b_sink0.csv
  ```
- **11986756** (kv4stepD) - vLLM NVFP4 0.0282 and its layer-0 row 0.0262 (`nvfp4_g1`, `nvfp4_g1@0`; `stepD_llama-3_1-8b_sink0.csv`). Node d0100, COMPLETED, KV4_COMMIT `23f0997`.
  ```
  sbatch -J kv4stepD -p short -c 20 --mem=64G -t 01:30:00 --export=ALL,KV4_COMMIT=23f0997,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 0 --configs nvfp4_g1,nvfp4_g1@0 --out results/stepD_llama-3_1-8b_sink0.csv
  ```
- **11981935** (kv4sinkS) - ours 0.0103 (`nvfp4_khs@0`, sink 4; `sinkS_llama-3_1-8b_sink4.csv`). Node c0676, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4sinkS -p short -c 20 --mem=64G -t 02:30:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs nvfp4_kh,nvfp4_khs,nvfp4_khs@0 --out results/sinkS_llama-3_1-8b_sink4.csv
  ```

## Table `tab:ladder` - each recipe step, mean KL (offline, FP32, CPU, 8 x 2048)

- Results (all in the repo since 11-19): `results/baseB2_<model>_sink0.csv` for (a) `fp4t`, (b) `nvfp4` and the FP8 row;
  `results/sinkS_<model>_sink0.csv` for (c) `nvfp4_kh`; `results/sinkS_<model>_sink4.csv` for (d) `nvfp4_kh`,
  (e) `nvfp4_khs`, (f) `nvfp4_khs@0`. No analysis script; the mean KL is read from the CSV.
- (e) -> (f) is within one job; other steps cross jobs and CPU nodes (repeat noise 2.8-4%).

- **11983769** (kv4base) - Llama-3.1-8B (a), (b), FP8. Node c0683, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4base -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=4860ba7,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 0 --configs fp8,int4t,fp4t,nvfp4_g1,nvfp4 --out results/baseB2_llama-3_1-8b_sink0.csv
  ```
- **11983770** (kv4base) - Qwen2.5-7B-Instruct (a), (b), FP8. Node c0690, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4base -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=4860ba7,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 0 --configs fp8,int4t,fp4t,nvfp4_g1,nvfp4 --out results/baseB2_qwen2_5-7b-instruct_sink0.csv
  ```
- **11983771** (kv4base) - Qwen3-8B-Base (a), (b), FP8. Node c0676, COMPLETED, KV4_COMMIT `4860ba7`.
  ```
  sbatch -J kv4base -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=4860ba7,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen3-8B-Base --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 0 --configs fp8,int4t,fp4t,nvfp4_g1,nvfp4 --out results/baseB2_qwen3-8b-base_sink0.csv
  ```
- **11981932** (kv4sinkS) - Llama-3.1-8B (c), sink 0. Node c0690, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4sinkS -p short -c 20 --mem=64G -t 02:30:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 0 --configs nvfp4_kh,nvfp4_khs,nvfp4_khs@0 --out results/sinkS_llama-3_1-8b_sink0.csv
  ```
- **11981935** (kv4sinkS) - Llama-3.1-8B (d)-(f), sink 4. Node c0676, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4sinkS -p short -c 20 --mem=64G -t 02:30:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs nvfp4_kh,nvfp4_khs,nvfp4_khs@0 --out results/sinkS_llama-3_1-8b_sink4.csv
  ```
- **11981938** (kv4sinkS) - Qwen2.5-7B-Instruct (c), sink 0. Node c0690, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4sinkS -p short -c 20 --mem=64G -t 02:30:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 0 --configs nvfp4_kh,nvfp4_khs,nvfp4_khs@0 --out results/sinkS_qwen2_5-7b-instruct_sink0.csv
  ```
- **11981941** (kv4sinkS) - Qwen2.5-7B-Instruct (d)-(f), sink 4. Node d0022, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4sinkS -p short -c 20 --mem=64G -t 02:30:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs nvfp4_kh,nvfp4_khs,nvfp4_khs@0 --out results/sinkS_qwen2_5-7b-instruct_sink4.csv
  ```
- **11981944** (kv4sinkS) - Qwen3-8B-Base (c), sink 0. Node c0676, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4sinkS -p short -c 20 --mem=64G -t 02:30:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen3-8B-Base --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 0 --configs nvfp4_kh,nvfp4_khs,nvfp4_khs@0 --out results/sinkS_qwen3-8b-base_sink0.csv
  ```
- **11981947** (kv4sinkS) - Qwen3-8B-Base (d)-(f), sink 4. Node d0029, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4sinkS -p short -c 20 --mem=64G -t 02:30:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen3-8B-Base --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs nvfp4_kh,nvfp4_khs,nvfp4_khs@0 --out results/sinkS_qwen3-8b-base_sink4.csv
  ```

Text under the ladder ("What each step fixes"):

- Sink sweep (sink 1, 2, 8, 16): 11981933-34, 11981936-37 (Llama), 11981939-40, 11981942-43 (Qwen2.5),
  11981945-46, 11981948-49 (Qwen3); same command as the sink-0/4 jobs above with `--sink N` and the
  matching `--out`.
- Rotation after mean removal (12% / 14% / 8%), layer 0 alone (Qwen2.5 0.250 vs 0.0146; FP8 3.79 -> 0.033),
  global scale 1.0 (<= 1.3%): the (D) jobs below and 11983769-71.
- Context length (0.0093 at 2K to 0.0113 at 16K): `results/ctxC_llama31_8b_ctx*.csv`.

- **11986745** (kv4stepD) - (D) Llama-3.1-8B, sink 4. Node d0126, COMPLETED, KV4_COMMIT `ce5f0c8`.
  ```
  sbatch -J kv4stepD -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=ce5f0c8,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs fp8,fp8@0,nvfp4,nvfp4@0,nvfp4_ks,nvfp4_ks@0,nvfp4_khs@0 --out results/stepD_llama-3_1-8b_sink4.csv
  ```
- **11986746** (kv4stepD) - (D) Qwen2.5-7B-Instruct, sink 4. Node c0637, COMPLETED, KV4_COMMIT `ce5f0c8`.
  ```
  sbatch -J kv4stepD -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=ce5f0c8,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs fp8,fp8@0,nvfp4,nvfp4@0,nvfp4_ks,nvfp4_ks@0,nvfp4_khs@0 --out results/stepD_qwen2_5-7b-instruct_sink4.csv
  ```
- **11986747** (kv4stepD) - (D) Qwen3-8B-Base, sink 4. Node d0100, COMPLETED, KV4_COMMIT `ce5f0c8`.
  ```
  sbatch -J kv4stepD -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=ce5f0c8,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen3-8B-Base --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs fp8,fp8@0,nvfp4,nvfp4@0,nvfp4_ks,nvfp4_ks@0,nvfp4_khs@0 --out results/stepD_qwen3-8b-base_sink4.csv
  ```
- **11981950** (kv4ctxC) - context sweep 2K. Node d0142, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4ctxC -p short -c 28 --mem=200G -t 05:00:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=28,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 16 --sink 4 --configs fp8,nvfp4_khs@0 --chunk 512 --out results/ctxC_llama31_8b_ctx2048.csv
  ```
- **11981952** (kv4ctxC) - context sweep 4K. Node c0675, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4ctxC -p short -c 28 --mem=200G -t 05:00:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=28,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 4096 --windows 8 --sink 4 --configs fp8,nvfp4_khs@0 --chunk 512 --out results/ctxC_llama31_8b_ctx4096.csv
  ```
- **11981953** (kv4ctxC) - context sweep 8K. Node d0149, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4ctxC -p short -c 28 --mem=200G -t 05:00:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=28,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 8192 --windows 4 --sink 4 --configs fp8,nvfp4_khs@0 --chunk 512 --out results/ctxC_llama31_8b_ctx8192.csv
  ```
- **11981954** (kv4ctxC) - context sweep 16K. Node c0677, COMPLETED, KV4_COMMIT `d31691c`.
  ```
  sbatch -J kv4ctxC -p short -c 28 --mem=200G -t 05:00:00 --export=ALL,KV4_COMMIT=d31691c,OMP_NUM_THREADS=28,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 16384 --windows 2 --sink 4 --configs fp8,nvfp4_khs@0 --chunk 512 --out results/ctxC_llama31_8b_ctx16384.csv
  ```

## Table `tab:offline` - offline KL and perplexity increase (16 x 4096, sink 4, BF16 on GPU)

- Results: `results/acc_kl_<model>_ctx4096_sink4.csv` (columns "NVFP4, K rotated" = `nvfp4_kh`, FP8 = `fp8`) and
  `results/acc_kl_<model>_ctx4096_sink4_smooth.csv` ("K mean-removed + rotated" = `nvfp4_khs`), for
  `llama31_8b`, `mistral-7b-v0_3`, `qwen3-8b-base`. Values read directly; no analysis script.
- These jobs ran on H200 SXM in BF16 (`fp4` env); the batch scripts `acc_models.sbatch`, `qwen_diag.sbatch`
  contain the model loop and configs (SubmitLine has no arguments).

- **11818189** (kv4acc) - Llama-3.1-8B sink 4: "K rotated" and FP8 (`acc_kl_llama31_8b_ctx4096_sink4.csv`). Label `08827d7`, but the code had been synced to `d0cfcd3` before it started (our log, 11-11). Node d4055, COMPLETED, KV4_COMMIT `08827d7`.
  ```
  sbatch --parsable -J kv4acc -p gpu-short,gpu,gpu-interactive --gres=gpu:h200:1 --exclude=d4077 -t 00:50:00 --export=ALL,KV4_COMMIT=08827d7 slurm/accuracy.sbatch --ctx 4096 --windows 16 --sink 4 --out results/acc_kl_llama31_8b_ctx4096_sink4.csv
  ```
- **11822460** (kv4accm) - Mistral-7B-v0.3 and Qwen3-8B-Base, sink 0 and 4: "K rotated" and FP8 (`acc_kl_{mistral-7b-v0_3,qwen3-8b-base}_ctx4096_sink{0,4}.csv`). Node d4055, COMPLETED, KV4_COMMIT `91e0644`.
  ```
  sbatch -J kv4accm -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:50:00 --export=ALL,KV4_COMMIT=91e0644 slurm/acc_models.sbatch
  ```
- **11823492** (kv4qdiag) - "K mean-removed + rotated" for all three models (`acc_kl_*_ctx4096_sink4_smooth.csv`; also `kv_stats.py`). Node d4055, COMPLETED, KV4_COMMIT `67078bd`.
  ```
  sbatch -J kv4qdiag -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 01:00:00 --export=ALL,KV4_COMMIT=67078bd slurm/qwen_diag.sbatch
  ```

Text next to the table:

- **11827107** (kv4sm2) - "a mean from the first 512 tokens is 8-14% worse" and "without the sink Llama is 25% worse" (`acc_kl_*_smooth2.csv`). Node d4055, COMPLETED, KV4_COMMIT `f426d15`.
  ```
  sbatch -J kv4sm2 -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 01:00:00 --export=ALL,KV4_COMMIT=f426d15 slurm/smooth2.sbatch
  ```
- **11866079** (kv4expC) - FP32 recomputation (FP8 -17-18%, 4-bit -2-6%): `expC_llama31_8b_fp32_sink0_fp8_nvfp4.csv`. Node d0113, COMPLETED, KV4_COMMIT `1f9468b`.
  ```
  sbatch -J kv4expC -p short -c 20 --mem=60G -t 01:30:00 --export=ALL,KV4_COMMIT=1f9468b,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 4096 --windows 16 --sink 0 --configs bf16,fp8,nvfp4 --out results/expC_llama31_8b_fp32_sink0_fp8_nvfp4.csv
  ```
- **11866081** (kv4expC) - FP32: `expC_..._sink0_nvfp4_kh_mxfp4_kh.csv`. Node d0032, COMPLETED, KV4_COMMIT `1f9468b`.
  ```
  sbatch -J kv4expC -p short -c 20 --mem=60G -t 01:30:00 --export=ALL,KV4_COMMIT=1f9468b,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 4096 --windows 16 --sink 0 --configs bf16,nvfp4_kh,mxfp4_kh --out results/expC_llama31_8b_fp32_sink0_nvfp4_kh_mxfp4_kh.csv
  ```
- **11866082** (kv4expC) - FP32: `expC_..._sink0_mxfp4_uq_kh.csv`. Node d0032, COMPLETED, KV4_COMMIT `1f9468b`.
  ```
  sbatch -J kv4expC -p short -c 20 --mem=60G -t 01:30:00 --export=ALL,KV4_COMMIT=1f9468b,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 4096 --windows 16 --sink 0 --configs bf16,mxfp4_uq_kh --out results/expC_llama31_8b_fp32_sink0_mxfp4_uq_kh.csv
  ```
- **11866083** (kv4expC) - FP32: `expC_llama31_8b_fp32_sink4.csv`. Node c3147, COMPLETED, KV4_COMMIT `1f9468b`.
  ```
  sbatch -J kv4expC -p short -c 20 --mem=60G -t 01:30:00 --export=ALL,KV4_COMMIT=1f9468b,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 4096 --windows 16 --sink 4 --configs bf16,fp8,nvfp4_khs --out results/expC_llama31_8b_fp32_sink4.csv
  ```
- **11866802** (kv4expC) - FP32: `expC_..._sink0_int4a_kh.csv`. Node c0621, COMPLETED, KV4_COMMIT `8acc684`.
  ```
  sbatch -J kv4expC -p short -c 20 --mem=60G -t 02:00:00 --export=ALL,KV4_COMMIT=8acc684,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 4096 --windows 16 --sink 0 --configs bf16,int4a_kh --out results/expC_llama31_8b_fp32_sink0_int4a_kh.csv
  ```
- **11866803** (kv4expC) - FP32: `expC_..._sink0_int4a_kvh.csv`. Node d0142, COMPLETED, KV4_COMMIT `8acc684`.
  ```
  sbatch -J kv4expC -p short -c 20 --mem=60G -t 02:00:00 --export=ALL,KV4_COMMIT=8acc684,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 4096 --windows 16 --sink 0 --configs bf16,int4a_kvh --out results/expC_llama31_8b_fp32_sink0_int4a_kvh.csv
  ```

## Table `tab:invllm` - inside vLLM, full recipe vs BF16 KV

- Results: `results/vllm_reasoning_<tag>_<config>.csv` (per-problem correctness from
  `bench/vllm_reasoning_check.py`; Qwen2.5-7B-Instruct copies in `results/mathruns/`; R1-Distill-Qwen-7B and
  ProLong RULER CSVs in `results/` too). Means, paired permutation p (20000 draws, seed 0) and two-level bootstrap 95%
  intervals: `bench/analyze_reasoning_runs.py --boot 2000`; exact invocations **UNKNOWN** (the accuracy audit
  documents the globs per row). Do not include `phi-4_500_mathcal_*` in the Phi-4 row.
- "ours" in every row = `ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0`
  (order of the `KV4_*` keys varies by job); BF16 = `auto`.

Qwen2.5-7B-Instruct, 5 runs (all NVL d4077; G = 7 padded to 8):

- **11846762** (kv4vr) - run 1 (`q25_7bi_500`). Node d4077, COMPLETED, KV4_COMMIT `1d17c6c`.
  ```
  sbatch -J kv4vr -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=1d17c6c,KV4_CALIB=results/calib/Qwen2.5-7B-Instruct.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 500 --new 4096 --out results/vllm_reasoning_q25_7bi_500.csv
  ```
- **11847271** (kv4rep) - run 2 (`500r2`). Node d4077, COMPLETED, KV4_COMMIT `2302800`.
  ```
  sbatch -J kv4rep -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=2302800,KV4_CALIB=results/calib/Qwen2.5-7B-Instruct.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_SINKS=1,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_KMEAN=1;KV4_SINKS=1 --n 500 --new 4096 --out results/vllm_reasoning_q25_7bi_500r2.csv
  ```
- **11847273** (kv4rep) - run 3 (`500r3`). Node d4077, COMPLETED, KV4_COMMIT `2302800`.
  ```
  sbatch -J kv4rep -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=2302800,KV4_CALIB=results/calib/Qwen2.5-7B-Instruct.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_SINKS=1,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_KMEAN=1;KV4_SINKS=1 --n 500 --new 4096 --out results/vllm_reasoning_q25_7bi_500r3.csv
  ```
- **11847365** (kv4rep) - run 4 (`500s1`, shuffle 1). Node d4077, COMPLETED, KV4_COMMIT `077887d`.
  ```
  sbatch -J kv4rep -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=077887d,KV4_CALIB=results/calib/Qwen2.5-7B-Instruct.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_KMEAN=1;KV4_SINKS=1 --n 500 --new 4096 --shuffle-seed 1 --out results/vllm_reasoning_q25_7bi_500s1.csv
  ```
- **11847368** (kv4rep) - run 5 (`500s2`, shuffle 2). Node d4077, COMPLETED, KV4_COMMIT `077887d`.
  ```
  sbatch -J kv4rep -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=077887d,KV4_CALIB=results/calib/Qwen2.5-7B-Instruct.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_KMEAN=1;KV4_SINKS=1 --n 500 --new 4096 --shuffle-seed 2 --out results/vllm_reasoning_q25_7bi_500s2.csv
  ```

DeepSeek-R1-Distill-Qwen-7B, 5 runs (run 1 is two jobs: BF16 in 11846911, ours in 11846985):

- **11846911** (kv4vr) - run 1, BF16 (and `fp8@0`). Node d4077, COMPLETED, KV4_COMMIT `1d17c6c`.
  ```
  sbatch -J kv4vr -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=1d17c6c,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dtypes auto,fp8@0 --n 500 --new 4096 --out results/vllm_reasoning_r1q7b_500.csv
  ```
- **11846985** (kv4vr) - run 1, ours. Node d4077, COMPLETED, KV4_COMMIT `1d17c6c`.
  ```
  sbatch -J kv4vr -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=1d17c6c,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Qwen-7B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dtypes ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 500 --new 4096 --out results/vllm_reasoning_r1q7b_500.csv
  ```
- **11847272** (kv4rep) - run 2. Node d4077, COMPLETED, KV4_COMMIT `2302800`.
  ```
  sbatch -J kv4rep -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=2302800,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Qwen-7B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_KMEAN=1;KV4_SINKS=1 --n 500 --new 4096 --out results/vllm_reasoning_r1q7b_500r2.csv
  ```
- **11847274** (kv4rep) - run 3. Node d4077, COMPLETED, KV4_COMMIT `2302800`.
  ```
  sbatch -J kv4rep -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=2302800,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Qwen-7B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_KMEAN=1;KV4_SINKS=1 --n 500 --new 4096 --out results/vllm_reasoning_r1q7b_500r3.csv
  ```
- **11847366** (kv4rep) - run 4 (shuffle 1). Node d4077, COMPLETED, KV4_COMMIT `077887d`.
  ```
  sbatch -J kv4rep -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=077887d,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Qwen-7B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_KMEAN=1;KV4_SINKS=1 --n 500 --new 4096 --shuffle-seed 1 --out results/vllm_reasoning_r1q7b_500s1.csv
  ```
- **11847404** (kv4rep) - run 5 (shuffle 2). Node d4077, COMPLETED, KV4_COMMIT `077887d`.
  ```
  sbatch -J kv4rep -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=077887d,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Qwen-7B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0;KV4_KMEAN=1;KV4_SINKS=1 --n 500 --new 4096 --shuffle-seed 2 --out results/vllm_reasoning_r1q7b_500s2.csv
  ```

DeepSeek-R1-Distill-Llama-8B, 3 runs (tokenizer-fixed reruns, files `*_tf500*`; run 1 crosses node types: BF16 on
NVL, ours on SXM):

- **11870900** (kv4r1l) - run 1, BF16 (and FP8 67.0, used in Section 7). Node d4077, COMPLETED, KV4_COMMIT `d6e7162`.
  ```
  sbatch --parsable -J kv4r1l -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=d6e7162,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Llama-8B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n 500 --new 4096 --dtypes auto,fp8 --out results/vllm_reasoning_r1llama8b_tf500a.csv
  ```
- **11870901** (kv4r1l) - run 1, ours (and `fp8@0` 68.8). Node d4054, COMPLETED, KV4_COMMIT `d6e7162`.
  ```
  sbatch --parsable -J kv4r1l -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=d6e7162,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Llama-8B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n 500 --new 4096 --dtypes fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --out results/vllm_reasoning_r1llama8b_tf500b.csv
  ```
- **11870902** (kv4r1l) - run 2 (shuffle 1). Node d4054, COMPLETED, KV4_COMMIT `d6e7162`.
  ```
  sbatch --parsable -J kv4r1l -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=d6e7162,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Llama-8B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n 500 --new 4096 --dtypes auto,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --shuffle-seed 1 --out results/vllm_reasoning_r1llama8b_tf500s1.csv
  ```
- **11870903** (kv4r1l) - run 3 (shuffle 2). Node d4055, COMPLETED, KV4_COMMIT `d6e7162`.
  ```
  sbatch --parsable -J kv4r1l -p sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=d6e7162,KV4_CALIB=results/calib/DeepSeek-R1-Distill-Llama-8B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --n 500 --new 4096 --dtypes auto,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --shuffle-seed 2 --out results/vllm_reasoning_r1llama8b_tf500s2.csv
  ```

Phi-4, 3 runs:

- **11850995** (kv4cov) - run 1 (NVL). Node d4077, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cov -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/phi-4.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model microsoft/phi-4 --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 500 --new 2048 --out results/vllm_reasoning_phi-4_500.csv
  ```
- **11851473** (kv4cov) - run 2 (shuffle 1, SXM). Node d4052, COMPLETED, KV4_COMMIT `635e8ce`.
  ```
  sbatch -J kv4cov -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=635e8ce,KV4_CALIB=results/calib/phi-4.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model microsoft/phi-4 --dtypes auto,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 500 --new 2048 --shuffle-seed 1 --out results/vllm_reasoning_phi-4_500s1.csv
  ```
- **11852303** (kv4cov) - run 3 (shuffle 2, SXM). Node d4054, COMPLETED, KV4_COMMIT `635e8ce`.
  ```
  sbatch -J kv4cov -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=635e8ce,KV4_CALIB=results/calib/phi-4.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model microsoft/phi-4 --dtypes auto,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 500 --new 2048 --shuffle-seed 2 --out results/vllm_reasoning_phi-4_500s2.csv
  ```

Qwen3-32B thinking on, MATH-500 first 200, 16K tokens, 3 runs:

- **11852038** (kv4think) - run 1 (shuffle -1). Node d4054, COMPLETED, KV4_COMMIT `9ed0a22`.
  ```
  sbatch -J kv4think -p gpu,gpu-short --gres=gpu:h200:1 --mem=96G -c 16 -t 01:59:00 --export=ALL,KV4_COMMIT=9ed0a22,KV4_CALIB=results/calib/Qwen3-32B.pt,KV4_MAX_LEN=17408,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-32B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 200 --new 16384 --shuffle-seed -1 --out results/vllm_reasoning_Qwen3-32B_think200s-1.csv
  ```
- **11852040** (kv4think) - run 2 (shuffle 1). Node d4054, COMPLETED, KV4_COMMIT `9ed0a22`.
  ```
  sbatch -J kv4think -p gpu,gpu-short --gres=gpu:h200:1 --mem=96G -c 16 -t 01:59:00 --export=ALL,KV4_COMMIT=9ed0a22,KV4_CALIB=results/calib/Qwen3-32B.pt,KV4_MAX_LEN=17408,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-32B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 200 --new 16384 --shuffle-seed 1 --out results/vllm_reasoning_Qwen3-32B_think200s1.csv
  ```
- **11852041** (kv4think) - run 3 (shuffle 2). Node d4054, COMPLETED, KV4_COMMIT `9ed0a22`.
  ```
  sbatch -J kv4think -p gpu,gpu-short --gres=gpu:h200:1 --mem=96G -c 16 -t 01:59:00 --export=ALL,KV4_COMMIT=9ed0a22,KV4_CALIB=results/calib/Qwen3-32B.pt,KV4_MAX_LEN=17408,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-32B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 200 --new 16384 --shuffle-seed 2 --out results/vllm_reasoning_Qwen3-32B_think200s2.csv
  ```

Single runs:

- **11850990** (kv4cov) - Qwen3-32B thinking off (`KV4_NO_THINK=1`). Node d4077, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cov -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/Qwen3-32B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py,KV4_NO_THINK=1 slurm/vllm_kv4.sbatch --model Qwen/Qwen3-32B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 500 --new 2048 --out results/vllm_reasoning_Qwen3-32B_500.csv
  ```
- **11851265** (kv4cov) - Qwen3-30B-A3B thinking off. Node d4077, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cov -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:59:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/Qwen3-30B-A3B.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py,KV4_NO_THINK=1 slurm/vllm_kv4.sbatch --model Qwen/Qwen3-30B-A3B --dtypes auto,fp8@0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_KMEAN=1;KV4_SINKS=1;KV4_BOUNDARY=0 --n 500 --new 2048 --out results/vllm_reasoning_Qwen3-30B-A3B_500.csv
  ```
- **11846336** (kv4lr) - ProLong RULER-style, BF16 side (`vllm_ruler_prolong_auto.csv`). Node d4054, COMPLETED, KV4_COMMIT `c7a8e58`.
  ```
  sbatch -J kv4lr -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:59:00 --export=ALL,KV4_COMMIT=c7a8e58,KV4_SCRIPT=bench/vllm_ruler.py slurm/vllm_kv4.sbatch --configs auto --out results/vllm_ruler_prolong.csv
  ```
- **11847442** (kv4lr) - ProLong RULER-style, ours (`vllm_ruler_prolong_recipe.csv`); paired with 11846336 (pairing command **UNKNOWN**). Node d4054, COMPLETED, KV4_COMMIT `c7a8e58`.
  ```
  sbatch -J kv4lr -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:59:00 --export=ALL,KV4_COMMIT=c7a8e58,KV4_CALIB=results/calib/Llama-3-8B-ProLong-512k-Instruct.pt,KV4_SCRIPT=bench/vllm_ruler.py slurm/vllm_kv4.sbatch --configs recipe --out results/vllm_ruler_prolong.csv
  ```

Text after the table (vLLM perplexity, ours vs FP8; and the layer-0-only ablation):

- **11845691** (kv4ppl) - Llama-3.1-8B +0.06% (recipe; also `plugin_nv_l0`). Node d4077, COMPLETED, KV4_COMMIT `3b5f2bb`.
  ```
  sbatch -J kv4ppl -p sharing --gres=gpu:h200:1 -t 00:30:00 --export=ALL,KV4_COMMIT=3b5f2bb,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --configs auto,plugin_nv_l0,plugin_recipe --windows 16 --len 8192 --chunk 2048
  ```
- **11840475** (kv4ppl) - Llama-3.1-8B FP8 +0.94% (older source; 11983773 gives both in one job). Node d4077, COMPLETED, KV4_COMMIT `d0e1426`.
  ```
  sbatch -J kv4ppl -p sharing --gres=gpu:h200:1 -t 00:50:00 --export=ALL,KV4_COMMIT=d0e1426,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --configs auto,fp8,plugin_bf16,plugin_fp8,stock --windows 16 --len 8192 --chunk 2048
  ```
- **11850992** (kv4cppl) - Qwen3-32B +0.07% vs +0.59%. Node d4052, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cppl -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:45:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/Qwen3-32B.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-32B --configs auto,fp8,plugin_recipe --windows 16 --len 8192 --chunk 2048
  ```
- **11850996** (kv4cppl) - Phi-4 +0.15% vs +0.30%. Node d4052, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cppl -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:45:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/phi-4.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model microsoft/phi-4 --configs auto,fp8,plugin_recipe --windows 16 --len 8192 --chunk 2048
  ```
- **11851460** (kv4cppl) - OLMo-2-1124-7B +0.06% vs +0.34% (16 x 4000). Node d4077, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cppl -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:45:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/OLMo-2-1124-7B.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model allenai/OLMo-2-1124-7B --configs auto,fp8,plugin_recipe --windows 16 --len 4000 --chunk 2000
  ```
- **11851344** (kv4cppl) - Qwen3-30B-A3B +0.07% vs +0.04%. Node d4077, COMPLETED, KV4_COMMIT `14e3e45`.
  ```
  sbatch -J kv4cppl -p gpu-short,gpu,sharing --gres=gpu:h200:1 --mem=96G -c 16 -t 00:45:00 --export=ALL,KV4_COMMIT=14e3e45,KV4_CALIB=results/calib/Qwen3-30B-A3B.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-30B-A3B --configs auto,fp8,plugin_recipe --windows 16 --len 8192 --chunk 2048
  ```
- **11847198** (kv4vr) - Qwen2.5-7B "NVFP4 with only layer 0 in BF16 69.3% over 5 runs", run 1 (`500abl`); runs 2-5 are the `KV4_FORMAT=nvfp4;KV4_BOUNDARY=0` config of 11847271, 11847273, 11847365, 11847368. Node d4077, COMPLETED, KV4_COMMIT `0ef4df1`.
  ```
  sbatch -J kv4vr -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=0ef4df1,KV4_CALIB=results/calib/Qwen2.5-7B-Instruct.pt,KV4_SCRIPT=bench/vllm_reasoning_check.py slurm/vllm_kv4.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtypes ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_BOUNDARY=0,ultraquant_4bit!KV4_VLLM=1;KV4_FORMAT=nvfp4;KV4_SINKS=1;KV4_BOUNDARY=0 --n 500 --new 4096 --out results/vllm_reasoning_q25_7bi_500abl.csv
  ```

Other Section 7 numbers (layer-0 paragraph, in-vLLM FP8 first 200 problems): 11834597 (R1-Distill-Qwen-7B),
11834598 (R1-Distill-Qwen-1.5B), 11835749 (Qwen2.5-7B-Instruct), 11839926 (R1-1.5B `fp8@0` on FlashInfer),
11837607 (R1-Distill-Qwen-14B, NVL); bias-center runs 11960777/83/84/85 and 11965404/12 (H200 NVL). Offline
reasoning (H100) 11829267, 11831292, 11831644, 11831645. Their SubmitLines are in `sacct` and not repeated here;
**not cross-checked** in this map.

## Table `tab:vllm` - findings reported to vLLM

No measured cell except the numbers in the finding column:

- #67176, "0.20 cuts KL by 40-56% offline": `results/acc_kl_*_ctx4096_sink{0,4}_smooth2.csv` (`mxc0.156_khs` vs
  `mxc0.200_khs`, 3 models x sink 0/4 with mean removal) plus the earlier c sweep without mean removal.
  "the perplexity increase by 66-79% inside vLLM": job logs of 11879844-46.

- **11827107** (kv4sm2) - offline KL at c = 0.156 / 0.177 / 0.200 with mean removal, 3 models, sink 0 and 4. Node d4055, COMPLETED, KV4_COMMIT `f426d15`.
  ```
  sbatch -J kv4sm2 -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 01:00:00 --export=ALL,KV4_COMMIT=f426d15 slurm/smooth2.sbatch
  ```
- **11821950** (kv4scale) - Llama-3.1-8B c sweep (`scale_rule_mse_llama31_8b.csv`, `acc_kl_llama31_8b_ctx4096_csweep.csv`). Node d4055, COMPLETED, KV4_COMMIT `6c258b2`.
  ```
  sbatch -J kv4scale -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 -t 00:50:00 --export=ALL,KV4_COMMIT=6c258b2 slurm/scale_rule.sbatch
  ```
- **11879844** (kv4uqc) - in-vLLM perplexity, Llama-3.1-8B, BF16 / c = 0.156 / 0.177 / 0.20 (stock UltraQuant). Node d4077, COMPLETED, KV4_COMMIT `a8c3fa9`.
  ```
  sbatch --parsable -J kv4uqc -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=a8c3fa9,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model meta-llama/Llama-3.1-8B --configs auto,stock,stock_c177,stock_c20 --windows 16 --len 8192 --chunk 2048
  ```
- **11879845** (kv4uqc) - same, Mistral-7B-v0.3. Node d4077, COMPLETED, KV4_COMMIT `a8c3fa9`.
  ```
  sbatch --parsable -J kv4uqc -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=a8c3fa9,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model mistralai/Mistral-7B-v0.3 --configs auto,stock,stock_c177,stock_c20 --windows 16 --len 8192 --chunk 2048
  ```
- **11879846** (kv4uqc) - same, Qwen3-8B-Base. Node d4077, COMPLETED, KV4_COMMIT `a8c3fa9`.
  ```
  sbatch --parsable -J kv4uqc -p sharing --gres=gpu:h200:1 -t 00:59:00 --export=ALL,KV4_COMMIT=a8c3fa9,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model Qwen/Qwen3-8B-Base --configs auto,stock,stock_c177,stock_c20 --windows 16 --len 8192 --chunk 2048
  ```

- #41343 / #63081: qualitative; the data posted there are the Section 7 FP8 runs listed above.
- #46461 -> #67789, "0/4 -> 4/4 on both triggers":

- **11893478** (kv4pr) - stock vLLM 0.33.0 vs #67789's two backend files, prefix caching and chunked prefill triggers (H200 NVL). Node d4077, COMPLETED, KV4_COMMIT `9a00fde`.
  ```
  sbatch --parsable -J kv4pr -p sharing --gres=gpu:h200:1 -t 00:55:00 --export=ALL,KV4_COMMIT=9a00fde slurm/pr67789_check.sbatch
  ```

## Figure in `paper/figures/` that the paper does not use

`fig_serving.pdf` was removed on 11-19. `paper/make_figures.py` `serving()` (only with `KV4_SERVE_FIG=1`) plots
`results/serve_goodput_11-17.csv` (job 11967005: FP8 with FlashInfer's native decode, superseded by `tab:serve`);
`main.tex` never included it.

- **11967005** (kv4ana) - `analyze_serve.py` over `results/vllm_serve` (native-decode batch) -> `serve_goodput_11-17.csv`. Node c0585, COMPLETED, KV4_COMMIT not set.
  ```
  sbatch -J kv4ana -p short -t 00:30:00 -c 2 --mem=16G -o logs/%x-%j.out --wrap python3 bench/analyze_serve.py results/vllm_serve > results/serve_goodput_11-17.csv
  ```

## Calibration files (prerequisite of every recipe run in vLLM)

`KV4_CALIB=results/calib/<model>.pt` (static per-head K mean, `bench/calib_kv.py`, 8 x 2048 wikitext tokens).
Whether a file was regenerated between its creation and a later job is **UNKNOWN**; only one creating job per
file was found.

- **11844479** (kv4cal) - Llama-3.1-8B.pt. Node d1026, COMPLETED, KV4_COMMIT `96646a9`.
  ```
  sbatch -J kv4cal -p gpu-short,gpu --gres=gpu:1 --mem=64G -t 00:30:00 --export=ALL,KV4_COMMIT=96646a9,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --out results/calib/Llama-3.1-8B.pt
  ```
- **11844480** (kv4cal) - Qwen2.5-7B-Instruct.pt. Node d4054, COMPLETED, KV4_COMMIT `96646a9`.
  ```
  sbatch -J kv4cal -p gpu-short,gpu --gres=gpu:1 --mem=64G -t 00:30:00 --export=ALL,KV4_COMMIT=96646a9,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model Qwen/Qwen2.5-7B-Instruct --out results/calib/Qwen2.5-7B-Instruct.pt
  ```
- **11846899** (kv4cal) - DeepSeek-R1-Distill-Qwen-7B.pt. Node d1011, COMPLETED, KV4_COMMIT `64c42ba`.
  ```
  sbatch -J kv4cal -p courses-gpu --gres=gpu:v100-sxm2:1 --mem=64G -t 01:00:00 --export=ALL,KV4_COMMIT=64c42ba,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --out results/calib/DeepSeek-R1-Distill-Qwen-7B.pt
  ```
- **11846900** (kv4cal) - Llama-3-8B-ProLong-512k-Instruct.pt. Node d1011, COMPLETED, KV4_COMMIT `64c42ba`.
  ```
  sbatch -J kv4cal -p courses-gpu --gres=gpu:v100-sxm2:1 --mem=64G -t 01:00:00 --export=ALL,KV4_COMMIT=64c42ba,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model princeton-nlp/Llama-3-8B-ProLong-512k-Instruct --out results/calib/Llama-3-8B-ProLong-512k-Instruct.pt
  ```
- **11849500** (kv4cal) - Qwen3-32B.pt. Node d4054, COMPLETED, KV4_COMMIT `5d52ad6`.
  ```
  sbatch -J kv4cal -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --mem=128G -t 01:00:00 --export=ALL,KV4_COMMIT=5d52ad6,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model Qwen/Qwen3-32B --out results/calib/Qwen3-32B.pt
  ```
- **11849501** (kv4cal) - Qwen3-30B-A3B.pt. Node d4052, COMPLETED, KV4_COMMIT `5d52ad6`.
  ```
  sbatch -J kv4cal -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --mem=128G -t 01:00:00 --export=ALL,KV4_COMMIT=5d52ad6,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model Qwen/Qwen3-30B-A3B --out results/calib/Qwen3-30B-A3B.pt
  ```
- **11849502** (kv4cal) - phi-4.pt. Node d4054, COMPLETED, KV4_COMMIT `5d52ad6`.
  ```
  sbatch -J kv4cal -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --mem=128G -t 01:00:00 --export=ALL,KV4_COMMIT=5d52ad6,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model microsoft/phi-4 --out results/calib/phi-4.pt
  ```
- **11849503** (kv4cal) - OLMo-2-1124-7B.pt. Node d1011, COMPLETED, KV4_COMMIT `5d52ad6`.
  ```
  sbatch -J kv4cal -p courses-gpu --gres=gpu:v100-sxm2:1 --mem=64G -t 01:30:00 --export=ALL,KV4_COMMIT=5d52ad6,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model allenai/OLMo-2-1124-7B --out results/calib/OLMo-2-1124-7B.pt
  ```
- **11852172** (kv4cal) - DeepSeek-R1-Distill-Llama-8B.pt. Node d1004, COMPLETED, KV4_COMMIT `22a172d`.
  ```
  sbatch -J kv4cal -p courses-gpu --gres=gpu:v100-sxm2:1 --mem=64G -t 01:30:00 --export=ALL,KV4_COMMIT=22a172d,KV4_SCRIPT=bench/calib_kv.py slurm/accuracy.sbatch --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --out results/calib/DeepSeek-R1-Distill-Llama-8B.pt
  ```

## 11-21 additions (after the 11-19 map)
Same conventions as above (SubmitLines from `sacct`, queried 11-21; account flags removed).

### Table `tab:capacity` - highest sustainable request rate on one H200 (Llama-3.1-8B, Qwen3-32B, Qwen2.5-72B FP8 weights)

Result file: `results/serve_goodput_x_11-21.csv` (`bench/analyze_serve.py results/vllm_serve_x --boot 1000`, job 12045665; columns include the model). Workload `_reason` (1024 in, 8192 out, range 0.5), `max(150, 600 s x rate)` requests, 4 API-server processes (`APISRV=4`); `RTAG=_s1a4` / `_s2a4` mark seed 1 / 2. KV capacities (tokens) are the `GPU KV cache size` lines of the serving logs. All runs on H200 SXM; FP8 logs show `decode_backend=xqa`. Brackets are [highest rate that held, lowest that failed). One job timed out without a result (12038595, 72B ours 0.275; rerun as 12040416).

```
12036489 (d4055): sbatch --parsable -J kv4srv8 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -H -t 01:15:00 --export=ALL,KV4_COMMIT=ae55f51,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,CONFIGS=recipe,RATES=1.6 1.65 slurm/vllm_serve.sbatch
12036490 (d4055): sbatch --parsable -J kv4srv8 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -H -t 00:45:00 --export=ALL,KV4_COMMIT=ae55f51,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,CONFIGS=recipe,RATES=1.7 slurm/vllm_serve.sbatch
12036885 (d4055): sbatch -J kv4srv8 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -H -t 00:45:00 --export=ALL,KV4_COMMIT=ae55f51,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,CONFIGS=fp8_xqa,RATES=1.3 slurm/vllm_serve.sbatch
12040496 (d4055): sbatch --parsable -J kv4srv8 -p gpu --gres=gpu:h200:1 --exclude=d4077 --mem=24G -t 00:45:00 --export=ALL,KV4_COMMIT=ce485b9,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,CONFIGS=fp8_xqa,RATES=1.25 slurm/vllm_serve.sbatch
12040497 (d4055): sbatch --parsable -J kv4srv8 -p gpu --gres=gpu:h200:1 --exclude=d4077 --mem=24G -t 00:45:00 --export=ALL,KV4_COMMIT=ce485b9,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=1,RTAG=_s1a4,CONFIGS=recipe,RATES=1.6 slurm/vllm_serve.sbatch
12044817 (d4055): sbatch --parsable -J kv4srv8 -p gpu --gres=gpu:h200:1 --exclude=d4077 --mem=24G -t 01:00:00 --export=ALL,KV4_COMMIT=e0a826c,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=2,RTAG=_s2a4,RATES=1.25 slurm/vllm_serve.sbatch
12034670 (d4053): sbatch --parsable -J kv4srv32 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 01:59:00 --dependency=afterok:12034663 --export=ALL,KV4_COMMIT=4e40127,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,MODEL=Qwen/Qwen3-32B,RATES=0.25 0.35 slurm/vllm_serve.sbatch
12036905 (d4055): sbatch --parsable -J kv4srv32 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 00:50:00 --export=ALL,KV4_COMMIT=ec2f284,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,MODEL=Qwen/Qwen3-32B,CONFIGS=fp8_xqa,RATES=0.3 slurm/vllm_serve.sbatch
12034672 (d4055): sbatch --parsable -J kv4srv32 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 01:59:00 --dependency=afterok:12034663 --export=ALL,KV4_COMMIT=4e40127,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,MODEL=Qwen/Qwen3-32B,RATES=0.45 0.55 slurm/vllm_serve.sbatch
12037600 (d4052): sbatch --parsable -J kv4srv32 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -t 00:50:00 --export=ALL,KV4_COMMIT=6f85d05,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,MODEL=Qwen/Qwen3-32B,CONFIGS=fp8_xqa,RATES=0.275 slurm/vllm_serve.sbatch
12040415 (d4055): sbatch --parsable -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -J kv4srv32 -t 01:30:00 --export=ALL,KV4_COMMIT=7746287,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,MODEL=Qwen/Qwen3-32B,SEED=0,RTAG=_a4,CONFIGS=recipe,RATES=0.4 slurm/vllm_serve.sbatch
12041167 (d4054): sbatch --parsable -J kv4srv32 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -t 00:59:00 --export=ALL,KV4_COMMIT=7cb44cf,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,MODEL=Qwen/Qwen3-32B,SEED=1,RTAG=_s1a4,CONFIGS=recipe,RATES=0.4 slurm/vllm_serve.sbatch
12040417 (d4055): sbatch --parsable -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -J kv4srv32 -t 00:59:00 --export=ALL,KV4_COMMIT=7746287,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,MODEL=Qwen/Qwen3-32B,SEED=1,RTAG=_s1a4,CONFIGS=fp8_xqa,RATES=0.25 slurm/vllm_serve.sbatch
12034671 (d4052): sbatch --parsable -J kv4srv72 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 -t 01:59:00 --dependency=afterok:12034664 --export=ALL,KV4_COMMIT=4e40127,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,MODEL=RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic,CALIB=results/calib/Qwen2.5-72B-Instruct.pt,RATES=0.15 0.25 slurm/vllm_serve.sbatch
12038594 (d4055): sbatch --parsable -J kv4srv72 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -t 01:59:00 --export=ALL,KV4_COMMIT=8395a0e,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,MODEL=RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic,CALIB=results/calib/Qwen2.5-72B-Instruct.pt,SEED=0,RTAG=_a4,CONFIGS=fp8_xqa,RATES=0.1 0.125 slurm/vllm_serve.sbatch
12038595 (d4053): sbatch --parsable -J kv4srv72 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -t 00:59:00 --export=ALL,KV4_COMMIT=8395a0e,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,MODEL=RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic,CALIB=results/calib/Qwen2.5-72B-Instruct.pt,SEED=0,RTAG=_a4,CONFIGS=recipe,RATES=0.275 slurm/vllm_serve.sbatch
12038596 (d4052): sbatch --parsable -J kv4srv72 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -t 00:59:00 --export=ALL,KV4_COMMIT=8395a0e,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,MODEL=RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic,CALIB=results/calib/Qwen2.5-72B-Instruct.pt,SEED=1,RTAG=_s1a4,CONFIGS=recipe,RATES=0.25 slurm/vllm_serve.sbatch
12040416 (d4055): sbatch --parsable -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -J kv4srv72 -t 01:59:00 --export=ALL,KV4_COMMIT=7746287,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,MODEL=RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic,CALIB=results/calib/Qwen2.5-72B-Instruct.pt,SEED=0,RTAG=_a4,CONFIGS=recipe,RATES=0.275 slurm/vllm_serve.sbatch
12040419 (d4055): sbatch --parsable -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -J kv4srv72 -t 00:59:00 --export=ALL,KV4_COMMIT=7746287,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,MODEL=RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic,CALIB=results/calib/Qwen2.5-72B-Instruct.pt,SEED=1,RTAG=_s1a4,CONFIGS=fp8_xqa,RATES=0.125 slurm/vllm_serve.sbatch
12045665 (c0674): sbatch -J kv4ana -p short -t 00:30:00 --mem=24G -o logs/kv4ana-%j.out --wrap python3 bench/analyze_serve.py results/vllm_serve_x --boot 1000 > results/serve_goodput_x_11-21.csv; echo done
```

### Capacity paragraph (Section 7) - per-step speed with FP8 weights and 72B accuracy

Llama-3.1-8B with FP8 weights (`--weight-quant fp8`): `results/vllm_forward_w8/` (12034667). Qwen2.5-72B per-step: `results/vllm_forward_72b/` (12037666). Qwen2.5-72B in-vLLM perplexity (wikitext-2, 16 x 8192): `results/vllm_ppl_qwen2_5-72b-instruct-fp8.txt`, the RESULT lines of 12037667's log.

```
12034667 (d4055): sbatch --parsable -J kv4fwd -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=96G -c 16 -t 01:00:00 --export=ALL,KV4_COMMIT=4e40127,KV4_CALIB=results/calib/Llama-3.1-8B.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --model meta-llama/Llama-3.1-8B --configs fp8_fi_xqa,plugin_recipe --weight-quant fp8 --out-dir results/vllm_forward_w8
12037666 (d4053): sbatch --parsable -J kv4fwd72 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -c 16 -t 00:50:00 --export=ALL,KV4_COMMIT=6a0c66a,KV4_CALIB=results/calib/Qwen2.5-72B-Instruct.pt,KV4_SCRIPT=bench/vllm_forward_matrix.py slurm/vllm_kv4.sbatch --model RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic --max-model-len 32768 --configs fp8_fi_xqa,plugin_recipe --out-dir results/vllm_forward_72b --cases b1_16k=1*1@16384 b1_31k=1*1@31744 b8_16k=8*1@16384 b8_31k=8*1@31744 b16_16k=16*1@16384 b32_8k=32*1@8192
12037667 (d4055): sbatch --parsable -J kv4ppl72 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=34G -c 16 -t 01:00:00 --export=ALL,KV4_COMMIT=6a0c66a,KV4_CALIB=results/calib/Qwen2.5-72B-Instruct.pt,KV4_SCRIPT=bench/vllm_prompt_ppl.py slurm/vllm_kv4.sbatch --model RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic --configs auto,fp8,fp8_l0,plugin_recipe
```

### Limitations, FP8-baselines item - NVBit of XQA and the C1 prototype (K on FP8 tensor cores)

NVBit opcode histograms: `results/nvbit_xqa/` (12042910: XQA pages 16 / 64, our NVFP4 and MXFP4; 12043372: the prototype). Prototype code: `kv4/cuda/decode_fp4_k8.cu`, `kv4/cuda/decode_fp4_k8.py`, `bench/k8_proto.py`, `slurm/k8_proto.sbatch`. Speed: `results/k8_proto_H200_v1.csv` (12043330) and `results/k8_proto_H200_v2.csv` (12043414; NVFP4, prototype and XQA in one job, the numbers the paper quotes). KL with one K scale per 32 dims: `results/c1_kl_*_sink4.csv` (12042972-12042974, CPU, FP32).

```
12042910 (d4054): sbatch -J kv4nvbit -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=32G -t 00:45:00 --export=ALL,KV4_COMMIT=018e322 slurm/nvbit.sbatch xqa_p16 xqa_p64 cuda_nvfp4 cuda_mxfp4
12043372 (d4053): sbatch --parsable -J kv4nvbit -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=48G -t 00:45:00 --export=ALL,KV4_COMMIT=411e981 slurm/nvbit.sbatch k8_nvfp4 cuda_nvfp4
12043330 (d4053): sbatch --parsable -J kv4k8 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=48G -t 00:50:00 --export=ALL,KV4_COMMIT=fe91f7b slurm/k8_proto.sbatch --tag v1
12043414 (d4052): sbatch --parsable -J kv4k8 -p gpu-short --gres=gpu:h200:1 --exclude=d4077 --mem=48G -t 00:59:00 --export=ALL,KV4_COMMIT=7600c97 slurm/k8_proto.sbatch --tag v2
12042972 (d0113): sbatch --parsable -J kv4c1kl -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=2be7ffc,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model meta-llama/Llama-3.1-8B --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs fp8@0,nvfp4_khs@0,nvfp4g32K_khs@0,mxc0.200K_khs@0 --out results/c1_kl_llama-3_1-8b_sink4.csv
12042973 (CPU node): sbatch --parsable -J kv4c1kl -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=2be7ffc,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen2.5-7B-Instruct --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs fp8@0,nvfp4_khs@0,nvfp4g32K_khs@0,mxc0.200K_khs@0 --out results/c1_kl_qwen2_5-7b-instruct_sink4.csv
12042974 (CPU node): sbatch --parsable -J kv4c1kl -p short -c 20 --mem=64G -t 03:00:00 --export=ALL,KV4_COMMIT=2be7ffc,OMP_NUM_THREADS=20,KV4_SCRIPT=bench/accuracy_kl.py slurm/accuracy.sbatch --model Qwen/Qwen3-8B-Base --dtype fp32 --device cpu --ctx 2048 --windows 8 --sink 4 --configs fp8@0,nvfp4_khs@0,nvfp4g32K_khs@0,mxc0.200K_khs@0 --out results/c1_kl_qwen3-8b-base_sink4.csv
```

### Section 7, "Two runs of ours that did not reproduce"

Original runs: 12009230 (seed 2 at 1.25 / 1.5, 4 API servers; TIMEOUT after both rates) and 11981960 (seed 0 at 0.25 / 0.5, one API server); both rows stay in the result CSVs. Miss timeline: `bench/serve_miss_timeline.py` (12044811). Reruns: 12044817 (seed 2 at 1.25, 4 API servers, both configs), 12044818 (seed 0 at 0.25 / 0.5, 4 API servers, both configs), 12044819 (seed 0 at 0.25 / 0.5, one API server, ours, `RTAG=_rep`).

```
12044811 (c0633): sbatch -J kv4ana -p short -t 00:30:00 --mem=24G -o logs/kv4miss-%j.out --wrap python3 bench/serve_miss_timeline.py results/vllm_serve_x/Llama-3.1-8B_recipe_reason_i1024_o8192_x0.5_r0.25.json results/vllm_serve_x/Llama-3.1-8B_recipe_reason_i1024_o8192_x0.5_r0.5.json results/vllm_serve_x/Llama-3.1-8B_fp8_xqa_reason_i1024_o8192_x0.5_r0.5.json results/vllm_serve_x/Llama-3.1-8B_recipe_reason_i1024_o8192_x0.5_r1.25_s2.json results/vllm_serve_x/Llama-3.1-8B_recipe_reason_i1024_o8192_x0.5_r1.25_s1.json
12044817 (d4055): sbatch --parsable -J kv4srv8 -p gpu --gres=gpu:h200:1 --exclude=d4077 --mem=24G -t 01:00:00 --export=ALL,KV4_COMMIT=e0a826c,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=2,RTAG=_s2a4,RATES=1.25 slurm/vllm_serve.sbatch
12044818 (d4052): sbatch --parsable -J kv4srv8 -p gpu --gres=gpu:h200:1 --exclude=d4077 --mem=24G -t 01:30:00 --export=ALL,KV4_COMMIT=e0a826c,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,APISRV=4,SEED=0,RTAG=_a4,RATES=0.25 0.5 slurm/vllm_serve.sbatch
12044819 (d4052): sbatch --parsable -J kv4srv8 -p gpu --gres=gpu:h200:1 --exclude=d4077 --mem=24G -t 01:00:00 --export=ALL,KV4_COMMIT=e0a826c,W=_reason,IN=1024,OUT=8192,RANGE=0.5,NP=150,NPT=600,SEED=0,RTAG=_rep,CONFIGS=recipe,RATES=0.25 0.5 slurm/vllm_serve.sbatch
```

## UNKNOWN items

1. Exact command (and host) that produced `results/serve_goodput_xqa_11-19.csv` (`analyze_serve.py --boot 1000`
   per the plan; no Slurm job).
2. Exact `bench/analyze_reasoning_runs.py` invocations (globs, `--ref`, `--boot`) for `tab:invllm` and the MATH
   columns of `tab:baselines`; the accuracy audit lists the files per row but not the command lines.
3. How the ProLong RULER pairing (+0.6 points, p = 0.50, 450 paired trials) was computed.
4. Pinned versions of the `fp4` conda env beyond "torch 2.6 + Triton" and "transformers 5.21".
5. Whether `results/calib/*.pt` files were regenerated after the jobs listed above.
6. Logs of 11850980 (prefill profile) are not in `~/kv4-hopper/logs` (ran from `~/kv4-test`, per the e2e audit);
   its numbers are backed by our internal log only.
7. The fixed-split-rule column for the kernel table: never measured (11988677 pending); the "9 of 11" statement
   is an analysis of 11812198.

## Inconsistencies found while building this map

1. **`tab:serve` caption, "a repeat of seed 0 (same arrivals, same node)":** true at 1.35 (11989408 and the repeat
   11990246, both d4052), not at 1.5: the seed-0 run 11981959 ran on d4053, the repeat on d4052. our log, 11-19
   makes the same claim for 1.5.
2. **Decode-led throughput (Section 6):** 11846510 (FlashInfer) and 11846746 (FA3) have identical SubmitLines;
   the FlashInfer backend came from the submitting shell. The three jobs also ran on two nodes (d4052, d4054) and
   two commits (6f29121, 1a056b8); the paper says "separate jobs".
3. **"Mean removal and sinks cost 1.2-3.2% (same job)"** sits next to `tab:e2e` (11979733) but comes from
   11845880 (commit 4cfb97e, 11-13); 11979733 has no `plugin_nv_l0` config.
4. **BF16 speed cells of `tab:baselines`** exist only in the log of 11987693 (the `bf16` run exits 1 before
   writing its JSON).
5. Our internal hardware table says 11804024 "has no sacct row"; sacct now returns it (COMPLETED, d4053).
6. 11987962 (Phi-4 FP8) wrote `results/phi-4_500fp8_fp8.csv` without the usual prefix; easy to miss with the
   `vllm_reasoning_*` globs.
7. (Fixed 11-19.) The inputs that were on cluster only (`baseB2_*`, `sinkS_*`, `stepD_*`, `ctxC_*`,
   `vllm_forward_xqa_bf16/`, the R1-Distill-Qwen-7B and Qwen2.5 FP8 MATH CSVs, ProLong RULER CSVs,
   `final_table_recipe_H200_pr53060.csv`) are now in `results/`, so every table can be regenerated from the repo.
