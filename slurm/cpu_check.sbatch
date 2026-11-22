#!/bin/bash
# CPU-only pre-flight check (11-12): data loading, tokenizer, prompt building of a GPU job, on the short partition,
# so a GPU job does not wait hours in the queue to fail in its first minute.
#   sbatch -J kv4pre slurm/cpu_check.sbatch reasoning|longbench
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$SLURM_SUBMIT_DIR CUDA_VISIBLE_DEVICES=
cd "$SLURM_SUBMIT_DIR"
~/.conda/envs/fp4/bin/python - "$1" <<'PY'
import sys
which = sys.argv[1]
if which == "reasoning":
    import json, os
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from bench.reasoning_eval import PROMPT, last_boxed, norm
    root = snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset")
    probs = [json.loads(l) for l in open(os.path.join(root, "test.jsonl")) if l.strip()]
    tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", padding_side="left")
    t = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=probs[0]["problem"])}],
                                tokenize=False, add_generation_prompt=True)
    print(len(probs), "problems; keys", sorted(probs[0]), "; prompt tokens", len(tok(t).input_ids))
    print("answer", probs[0]["answer"], "-> norm", norm(probs[0]["answer"]), "| boxed test", last_boxed("x \\boxed{\\frac{1}{2}}"))
    from bench.accuracy_kl import CONFIGS
    for c in "bf16,fp8,nvfp4_kh,nvfp4_kh:4,int4a_kh,mxfp4_kh,mxfp4_uq_kh,nvfp4:4,nvfp4_khs:4,int4a_khs:4".split(","):
        assert c.partition(":")[0] in CONFIGS, c
    print("configs ok")
elif which == "longbench":
    import json
    from transformers import AutoTokenizer
    from bench.longbench import load_task, build_ids, TASKS, CFG
    prompts = json.loads((CFG / "dataset2prompt.json").read_text())
    tok = AutoTokenizer.from_pretrained("princeton-nlp/Llama-3-8B-ProLong-512k-Instruct")
    for task in TASKS.split(","):
        data = load_task(task)
        ids = build_ids(tok, task, data[0], prompts[task], 65536)
        print(f"{task:22s} {len(data):4d} items, first prompt {ids.shape[1]} tokens, answers {data[0]['answers'][:1]}")
PY
