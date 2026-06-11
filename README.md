# TON Tolk SFT Hello World

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adriancichy-softwaredevelopment/ton-tolk-sft-hello-world/blob/main/notebooks/TON_Tolk_SFT_HelloWorld.ipynb)

A complete, runnable **Supervised Fine-Tuning (SFT)** pipeline packaged as a Google Colab notebook.

The goal is to specialize a small open-weights LLM (Llama-3.2-3B-Instruct or Qwen2.5-Coder-7B-Instruct) on the **TON blockchain** ecosystem — focusing on the modern **Tolk** smart contract language, the **Acton** toolchain, message handling, storage patterns, and secure contract development.

## Quick Start

### Option A — From GitHub (recommended)
```bash
git clone https://github.com/adriancichy-softwaredevelopment/ton-tolk-sft-hello-world.git
cd ton-tolk-sft-hello-world
```

Then open `notebooks/TON_Tolk_SFT_HelloWorld.ipynb` in Google Colab:
- Go to [colab.research.google.com](https://colab.research.google.com)
- File → Open notebook → GitHub tab → select this repo

### Option B — Direct Colab
1. Open the notebook directly via the "Open in Colab" badge above, or
2. Download `notebooks/TON_Tolk_SFT_HelloWorld.ipynb` and upload it to Colab.

## Run Instructions (Colab)
1. **Runtime → Change runtime type → GPU** (T4 is free tier; L4 is better)
2. The notebook currently defaults to the **non-gated** `Qwen/Qwen2.5-3B-Instruct` (no license approval needed).

If you want to switch back to the gated `meta-llama/Llama-3.2-3B-Instruct`:
   - Accept the Llama 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
   - Create a read token at https://huggingface.co/settings/tokens
   - In Colab, use the **Secrets** panel (key icon) and add `HF_TOKEN`
   - Change `MODEL_NAME` at the top of the notebook.
3. Run all cells top-to-bottom.

The notebook includes:
- Automatic BF16 (L4) vs 4-bit (T4) loading
- LoRA fine-tuning with `SFTTrainer`
- Conversational `{"messages": [...]}` dataset format
- **Full Acton integration** — install the real toolchain and validate model-generated Tolk code with `acton build`, `acton test`, `acton check`, `acton fmt`, etc.
- Data scaling helpers + curated links to all the official docs we audited (Acton + TON Tolk docs, llms.txt files, agent skills, reference contracts)
- **Colab MCP integration** — connect local AI agents (Grok, Claude, Cursor, etc.) directly to the running notebook for closed-loop cell execution, inspection, and iteration (see the "Connect AI Agent via Colab MCP" section in the notebook)
- A generator script so you can easily modify and regenerate the notebook

## Regenerating the Notebook

If you edit `generate_notebook.py`, you can recreate the notebook with:

```bash
python generate_notebook.py
```

This will overwrite `notebooks/TON_Tolk_SFT_HelloWorld.ipynb`.

## Project Structure

```
ton-tolk-sft-hello-world/
├── README.md
├── LICENSE
├── .gitignore
├── generate_notebook.py          # Re-runs to (re)generate the notebook
├── notebooks/
│   └── TON_Tolk_SFT_HelloWorld.ipynb   # Main deliverable
```

## Files

| File | Purpose |
|------|---------|
| `notebooks/TON_Tolk_SFT_HelloWorld.ipynb` | Main deliverable — the complete SFT + Acton pipeline |
| `generate_notebook.py` | Generator script. Re-run this if you want to modify the notebook programmatically |
| `README.md` | This file |
| `.gitignore` | Sensible ignores for Python + Jupyter + model artifacts |

The notebook contains:
- Automatic BF16 (L4) vs 4-bit NF4 (T4) selection
- LoRA (PEFT) setup with `all-linear` targets
- A high-quality conversational seed dataset (8 examples) using the exact `{"messages": [...]}` format expected by `trl.SFTTrainer`
- Full `SFTConfig` + `SFTTrainer` usage (modern trl API)
- Inference with the trained adapter
- Optional merge to full weights
- Clear comments on how to bring your own real `.tolk` files and documentation

## Using Your Own Data (the important part)

The seed dataset is intentionally tiny so the notebook runs quickly as a "Hello World".

For a real specialized model you should:

- Collect dozens to thousands of high-quality (instruction, response) pairs
- Preferred sources:
  - Real `.tolk` source files (from `ton-blockchain/acton-contracts`, your projects, tolk-bench, etc.)
  - Official TON documentation (Tolk overview, Blueprint guide, Acton docs)
  - Curated explanations written by experts or a strong teacher model
- A simple but effective pattern:
  - For each contract file: one "Write a Tolk contract that..." example + one "Explain the following Tolk code..." example
- Keep a small held-out set for manual spot-checking of generations

The notebook includes a commented Google Drive loading skeleton you can adapt.

## Acton — The Full Modern Dev Kit (Comprehensive Audit)

We thoroughly reviewed:
- https://ton-blockchain.github.io/acton/docs/welcome
- https://ton-blockchain.github.io/acton/docs/installation
- https://ton-blockchain.github.io/acton/docs/agent-skills/overview
- Linked pages (quickstart, walkthrough, tutorial, acton-toml, commands/*, testing/*, debug, deploy, verify, scripting, lint, CI, dApp, etc.)
- `llms-full.txt` on the Acton site
- GitHub repos: ton-blockchain/acton (README, Dockerfile, CONTRIBUTING), ton-blockchain/skills (the agent skills), ton-blockchain/acton-contracts (reference contracts), related TON docs (Tolk sections + llms.txt)
- IDE support, Docker, GitHub Actions, etc.

**Result**: We now have a complete picture of the "full dev tools" stack for Tolk + Acton work.

### Core Installation & Daily Workflow
```bash
# One-liner (Linux/Colab/WSL)
curl -LsSf https://github.com/ton-blockchain/acton/releases/latest/download/acton-installer.sh | sh
acton --version
acton doctor

acton new my_project --template counter   # or jetton, nft, empty; add --app for React+Vite dApp scaffold
cd my_project
acton build
acton test
acton check
acton fmt --check
acton script scripts/deploy.tolk          # local emulation first
```

Other essentials:
- `acton init`, `acton up`, `acton wrapper [--ts]`, `acton wallet new --local --airdrop`, `acton verify`, `acton func2tolk`, `acton script --net testnet`, `acton test --coverage --ui`, `acton test --debug` (DAP), `acton retrace`, etc.
- `Acton.toml` is the rich single source of truth (contracts + deps + BoC support, [test] with coverage/mutation/fuzz/fork/gas/ui, [lint], [wrappers], [scripts], import mappings, etc.).

### Best Reference & Training Material (for your SFT dataset)
- Acton docs + https://ton-blockchain.github.io/acton/llms-full.txt
- TON Docs Tolk pages + https://docs.ton.org/llms.txt (and per-page .md)
- https://github.com/ton-blockchain/acton-contracts (production-grade Tolk + tests + scripts + gas baselines; includes counter, jetton-v2, NFT, w5, highload, multisig, dns, elector, config)
- **Agent Skills** (https://github.com/ton-blockchain/skills) — `acton/`, `tolk/`, `func2tolk/`, `ton-blockchain/` skills with `SKILL.md` + `references/` (checklists, command maps, idiomatic patterns, porting guides). Install: `npx skills add -g https://github.com/ton-blockchain/skills`. Perfect companion material for an SFT TON assistant.
- Official VS Code extension (ton-core.vscode-ton) for full Tolk LSP + Acton integration.

### Notebook Integration
The `notebooks/TON_Tolk_SFT_HelloWorld.ipynb` now contains an expanded **## 15. Acton Toolchain** section (after the inference/merge cells) with:
- Robust Colab install + `acton doctor`
- Quality gates (`fmt --check`, `check`, `build`, `test` + coverage/gas notes)
- Realistic demo that takes "model output", writes it into a real project `contracts/*.tolk`, then runs the full modern validation pipeline (`build` + `wrapper` + `check` + `fmt --check` + `test`)
- Documentation of the broader ecosystem, agent skills, llms-*.txt, acton-contracts, Docker, CI (setup-acton action + GitHub/GitLab examples), debugging (DAP, retrace, backtraces), dApp wrappers, etc.

**Colab practicalities**: Re-run the install cell on every new runtime. Build + test + lint + fmt work excellently. Advanced browser Test UI and live DAP stepping are constrained; use `--backtrace full`, coverage, gas snapshots, and `acton test` results as your primary signals. Deployment flows are possible via Acton wallets + scripts but should only be used after thorough local validation (and never blindly on mainnet).

Docker alternative for stronger isolation: `docker run --rm -v "$PWD":/workspace -w /workspace ghcr.io/ton-blockchain/acton:1.1.0 build ...` (see the repo Dockerfile).

This gives you a closed loop: fine-tune on high-quality TON/Tolk/Acton data → generate → validate with the real production toolchain inside the notebook.

See the notebook itself and the expanded Acton section for copy-paste cells.

## Precision & Memory Notes (Colab 2026)

- **L4 GPU**: Native `bfloat16` works great → higher quality, faster.
- **T4 GPU**: Falls back to 4-bit (NF4 + double quant) + LoRA. This is the classic QLoRA setup and fits comfortably.
- Gradient checkpointing is enabled.
- Typical trainable params with the default LoRA (r=16): ~1–2% of total model size.
- If you hit OOM: lower `max_length`, reduce `gradient_accumulation_steps`, or decrease LoRA `r`.

## Recommended Models

| Model                              | Size | Notes                                      | Gated?     | Colab friendliness |
|------------------------------------|------|--------------------------------------------|------------|--------------------|
| meta-llama/Llama-3.2-3B-Instruct   | 3B   | Excellent balance, very popular for SFT    | Yes        | Excellent (small)  |
| Qwen/Qwen2.5-Coder-7B-Instruct     | 7B   | Outstanding at code generation & reasoning | Usually no | Good on L4, tight on T4 |

You can switch `MODEL_NAME` at the top of the notebook.

## After Training — What to Do with the Adapter

- The saved `./ton-tolk-lora-adapter` folder is tiny (tens of MB) and contains only the trainable weights + tokenizer.
- You can load it later with `PeftModel.from_pretrained(base, adapter_path)`.
- For easier distribution / inference engines, merge the weights (cell provided).
- Push to the Hugging Face Hub (adapter-only is usually preferred).

## Important Safety Note

Even after fine-tuning, **always**:
- Compile and test generated Tolk contracts with the official Acton / Blueprint tooling
- Review for missing access controls, incorrect opcodes, gas issues, etc.
- Treat the model as a very knowledgeable pair programmer, not an infallible oracle

## Extending the Pipeline

Ideas for v2:
- Add evaluation cells that attempt to compile generated contracts (requires installing the Acton toolchain in Colab — possible but heavy)
- Integrate a small RAG step over TON docs before generation (for higher factual accuracy)
- Use `trl` DPO / ORPO once you have preference data
- Switch to Unsloth for 2-5× faster training while keeping almost identical code

## License & Credits

This notebook is intended as an educational template. The underlying models have their own licenses (check Meta Llama / Qwen terms). TON, Tolk, Acton, and Blueprint are projects of the TON Foundation and community.

Happy building on The Open Network! 🚀
