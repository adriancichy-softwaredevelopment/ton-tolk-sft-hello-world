#!/usr/bin/env python3
"""
Generate a complete, production-quality 'Hello World' SFT notebook
for fine-tuning a small LLM on TON / Tolk / Acton domain.
Run this script locally to produce notebooks/TON_Tolk_SFT_HelloWorld.ipynb
Then upload that notebook to Google Colab.
"""

import json
from pathlib import Path

def md_cell(text: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    }

def code_cell(source_lines: list[str], execution_count: int | None = None) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source_lines],
        "outputs": [],
        "execution_count": execution_count
    }

def build_notebook() -> dict:
    cells = []

    # ========== 0. TITLE ==========
    cells.append(md_cell("""
# TON / Tolk SFT Hello World — Supervised Fine-Tuning Pipeline

**Goal**: Fine-tune a small modern open-weights model (Llama-3.2-3B-Instruct or Qwen2.5-Coder-7B) into a specialized assistant for the TON blockchain ecosystem, with strong knowledge of:

- **Tolk** — the new primary smart contract language for TON (TypeScript-like, actor-model native)
- Acton framework & toolchain
- Smart contract structure, storage, messages, opcodes, getters
- Best practices for secure, gas-efficient TON contracts

**Environment**: Google Colab (T4 or L4 GPU recommended)

**Precision**:
- L4 → native BF16
- T4 → 4-bit (BitsAndBytes) + LoRA

**Stack**: `transformers` + `trl` (SFTTrainer) + `peft` (LoRA) + `bitsandbytes` + `datasets`
"""))

    # ========== 1. PREREQUISITES ==========
    cells.append(md_cell("""
## 1. Prerequisites & Colab Setup

1. **Runtime**: Runtime → Change runtime type → **GPU** (T4 or L4)
2. **HF Token** (recommended, required for gated models like Llama):
   - Go to https://huggingface.co/settings/tokens (read access)
   - In Colab: `Secrets` (left sidebar key icon) → Name: `HF_TOKEN`
   - (Not strictly needed for the current non-gated default model)
3. **Your own data** (later): Prepare folders with `.tolk` files + `.md`/`.txt` docs. We provide a seed dataset + loaders below.
4. This is a **Hello World** — the tiny seed dataset will cause the model to strongly memorize the style and a few patterns. Replace with 200–2000+ high-quality examples for real specialization.
"""))

    # ========== 2. INSTALL ==========
    cells.append(md_cell("## 2. Install Dependencies"))

    cells.append(code_cell([
        "# Upgrade pip quietly and install the core stack",
        "!pip install -q --upgrade pip",
        "",
        "# Core libraries — pin reasonably recent stable versions for reproducibility",
        "!pip install -q \\",
        '    "transformers>=4.46.0" \\',
        '    "trl>=0.12.0" \\',
        '    "peft>=0.13.0" \\',
        '    "bitsandbytes>=0.43.0" \\',
        '    "accelerate>=1.0.0" \\',
        '    "datasets>=3.0.0" \\',
        '    "huggingface_hub>=0.26.0"',
        "",
        "print('Installation complete.')"
    ]))

    # ========== 3. IMPORTS ==========
    cells.append(md_cell("## 3. Imports & Environment"))

    cells.append(code_cell([
        "import os",
        "import json",
        "import torch",
        "from datasets import Dataset",
        "from transformers import (",
        "    AutoTokenizer,",
        "    AutoModelForCausalLM,",
        "    BitsAndBytesConfig,",
        "    pipeline,",
        ")",
        "from peft import LoraConfig, get_peft_model, PeftModel",
        "from trl import SFTTrainer, SFTConfig",
        "",
        "print('torch:', torch.__version__)",
        "print('CUDA available:', torch.cuda.is_available())",
        "if torch.cuda.is_available():",
        "    print('GPU:', torch.cuda.get_device_name(0))",
        "    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))",
        "",
        "!nvidia-smi"
    ]))

    # ========== 4. HARDWARE + PRECISION ==========
    cells.append(md_cell("## 4. Hardware Detection & Precision Strategy"))

    cells.append(code_cell([
        "def get_precision_config():",
        "    if not torch.cuda.is_available():",
        "        print('WARNING: No GPU detected. Training will be extremely slow.')",
        "        return {'torch_dtype': torch.float32, 'quantization_config': None, 'use_4bit': False}",
        "",
        "    gpu_name = torch.cuda.get_device_name(0).upper()",
        "    supports_bf16 = torch.cuda.is_bf16_supported()",
        "",
        "    print(f'GPU detected: {gpu_name}')",
        "    print(f'Native BF16 support: {supports_bf16}')",
        "",
        "    if 'L4' in gpu_name or (supports_bf16 and 'T4' not in gpu_name):",
        "        print('→ Using native BF16 (no quantization)')",
        "        return {",
        "            'torch_dtype': torch.bfloat16,",
        "            'quantization_config': None,",
        "            'use_4bit': False",
        "        }",
        "    else:",
        "        print('→ Using 4-bit quantization (NF4) + FP16 compute (typical for T4)')",
        "        bnb_config = BitsAndBytesConfig(",
        "            load_in_4bit=True,",
        "            bnb_4bit_quant_type='nf4',",
        "            bnb_4bit_compute_dtype=torch.float16,",
        "            bnb_4bit_use_double_quant=True,",
        "        )",
        "        return {",
        "            'torch_dtype': torch.float16,",
        "            'quantization_config': bnb_config,",
        "            'use_4bit': True",
        "        }",
        "",
        "precision = get_precision_config()"
    ]))

    # ========== 5. MODEL SELECTION ==========
    cells.append(md_cell("""
## 5. Choose Base Model

**Recommended for this task** (temporary non-gated default):
- **Primary (small & Colab-friendly, non-gated)**: `Qwen/Qwen2.5-3B-Instruct`
- **Strong coder alternative**: `Qwen/Qwen2.5-Coder-7B-Instruct` (better at code, ~2× slower on T4)

**Note**: The previous default `meta-llama/Llama-3.2-3B-Instruct` is gated. You must accept the Llama license on HF and provide a valid `HF_TOKEN` to use it.
Qwen2.5 models are open-weights and load without approval.
"""))

    cells.append(code_cell([
        "# === CONFIGURE YOUR MODEL HERE ===",
        "MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'   # non-gated, strong small model. Alternative: 'Qwen/Qwen2.5-Coder-7B-Instruct'",
        "# To use Llama 3.2 instead (gated): MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'  (requires HF access approval + token)",
        "",
        "# For gated models (Llama), load token from Colab Secrets or env var",
        "HF_TOKEN = None",
        "try:",
        "    from google.colab import userdata",
        "    HF_TOKEN = userdata.get('HF_TOKEN')",
        "except Exception:",
        "    HF_TOKEN = os.environ.get('HF_TOKEN')",
        "",
        "print('Using model:', MODEL_NAME)",
        "print('HF token present:', bool(HF_TOKEN))"
    ]))

    # ========== 6. LOAD TOKENIZER ==========
    cells.append(code_cell([
        "tokenizer = AutoTokenizer.from_pretrained(",
        "    MODEL_NAME,",
        "    token=HF_TOKEN,",
        "    trust_remote_code=True,",
        ")",
        "",
        "# Ensure we have a pad token (many instruct models use EOS as pad)",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "",
        "print('Tokenizer loaded. Vocab size:', tokenizer.vocab_size)",
        "print('Chat template present:', tokenizer.chat_template is not None)"
    ]))

    # ========== 7. LOAD BASE MODEL ==========
    cells.append(md_cell("## 6. Load Base Model (with appropriate precision)"))

    cells.append(code_cell([
        "model_kwargs = {",
        "    'trust_remote_code': True,",
        "    'device_map': 'auto',",
        "    'token': HF_TOKEN,",
        "}",
        "if precision['quantization_config'] is not None:",
        "    model_kwargs['quantization_config'] = precision['quantization_config']",
        "else:",
        "    model_kwargs['torch_dtype'] = precision['torch_dtype']",
        "",
        "model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)",
        "",
        "# Enable gradient checkpointing for memory savings (highly recommended)",
        "model.gradient_checkpointing_enable()",
        "model.config.use_cache = False   # required for gradient checkpointing + training",
        "",
        "print('Model loaded successfully.')",
        "print('Trainable parameters (before LoRA):', sum(p.numel() for p in model.parameters() if p.requires_grad))"
    ]))

    # ========== 8. LoRA CONFIG ==========
    cells.append(md_cell("""
## 7. LoRA Configuration (PEFT)

We use LoRA (Low-Rank Adaptation) so only a tiny fraction of parameters are trained.
This is the key to making 3B–7B fine-tuning feasible on a single Colab GPU.
"""))

    cells.append(code_cell([
        "lora_config = LoraConfig(",
        "    r=16,                    # LoRA rank — 8-32 is common. Higher = more capacity, more VRAM.",
        "    lora_alpha=32,",
        "    lora_dropout=0.05,",
        "    bias='none',",
        "    task_type='CAUSAL_LM',",
        "    # 'all-linear' works well for most modern architectures.",
        "    # You can also target specific modules for Llama: ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']",
        "    target_modules='all-linear',",
        ")",
        "",
        "model = get_peft_model(model, lora_config)",
        "model.print_trainable_parameters()"
    ]))

    # ========== 9. SEED DATASET ==========
    cells.append(md_cell("""
## 8. Seed Training Dataset (Conversational Format)

The dataset uses the standard OpenAI/HF conversational format with a `"messages"` column.

Each example is a list of `{"role": "user" | "assistant" | "system", "content": "..."}`.

**This is a minimal Hello World seed (8 examples)**. It teaches the model:
- Correct Tolk syntax and idioms (structs, contract declaration, lazy loading, match, get methods, opcoded messages)
- Tone: precise, modern, production-ready TON developer
- Basic Acton/Blueprint awareness

**Next step for you**: Replace or massively expand this with real data extracted from:
- Official TON docs (tolk/overview, blueprint, etc.)
- `.tolk` source files from acton-contracts, tolk-bench, your own repos
- Curated Q&A pairs ("Write a contract that...", "Explain this storage layout...")
"""))

    # The actual seed data — realistic Tolk examples based on public TON docs
    seed_examples = [
        {
            "messages": [
                {"role": "system", "content": "You are an expert TON smart contract engineer specializing in Tolk and the Acton toolchain. You write clean, gas-efficient, secure contracts and always follow current TON best practices."},
                {"role": "user", "content": "Write a minimal Tolk smart contract that stores a counter integer. It must support two operations: increasing the counter by a given amount and resetting it to zero. Include the storage struct, message types with opcodes, the contract declaration, onInternalMessage handler, and a getter."},
                {"role": "assistant", "content": "```tolk\nimport \"@stdlib/gas-payments\" tolk 1.1\n\nstruct Storage {\n    counter: uint32;\n}\n\nstruct(0x7e8764ef) IncreaseCounter {\n    increaseBy: uint32;\n}\n\nstruct(0x3a752f06) ResetCounter {}\n\ntype AllowedMessage = IncreaseCounter | ResetCounter;\n\ncontract Counter {\n    storage: Storage;\n    incomingMessages: AllowedMessage;\n}\n\nfun Storage.load() {\n    return Storage.fromCell(contract.getData());\n}\n\nfun Storage.save(self) {\n    contract.setData(self.toCell());\n}\n\nfun onInternalMessage(in: InMessage) {\n    val msg = lazy AllowedMessage.fromSlice(in.body);\n    val storage = lazy Storage.load();\n\n    match (msg) {\n        IncreaseCounter => {\n            storage.counter += msg.increaseBy;\n            storage.save();\n            cashback(sender());\n        }\n        ResetCounter => {\n            storage.counter = 0;\n            storage.save();\n            cashback(sender());\n        }\n    }\n}\n\nget fun currentCounter(): uint32 {\n    val storage = lazy Storage.load();\n    return storage.counter;\n}\n```"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Define a Tolk storage struct for a simple TON wallet-like contract that holds a seqno (uint32) and a public key (uint256). Show how to load and save it."},
                {"role": "assistant", "content": "```tolk\nstruct Storage {\n    seqno: uint32;\n    public_key: uint256;\n}\n\nfun Storage.load() {\n    return Storage.fromCell(contract.getData());\n}\n\nfun Storage.save(self) {\n    contract.setData(self.toCell());\n}\n\n// Usage inside a function:\n// val storage = lazy Storage.load();\n// storage.seqno += 1;\n// storage.save();\n```"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Show the recommended way to declare message structs with opcodes in Tolk and how to parse an incoming message body."},
                {"role": "assistant", "content": "```tolk\n// 32-bit opcodes are conventional for TON messages\nstruct(0x7e8764ef) IncreaseCounter {\n    increaseBy: uint32;\n}\n\nstruct(0x3a752f06) ResetCounter {}\n\n// Union type for all supported messages\ntype AllowedMessage = IncreaseCounter | ResetCounter;\n\n// Inside onInternalMessage:\nfun onInternalMessage(in: InMessage) {\n    val msg = lazy AllowedMessage.fromSlice(in.body);\n\n    match (msg) {\n        IncreaseCounter => { /* ... */ }\n        ResetCounter => { /* ... */ }\n    }\n}\n```"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a Tolk getter that returns the current counter value from storage."},
                {"role": "assistant", "content": "```tolk\nget fun currentCounter(): uint32 {\n    val storage = lazy Storage.load();\n    return storage.counter;\n}\n\n// You can also return more complex data:\n// get fun getFullState(): (uint32, uint32) {\n//     val s = lazy Storage.load();\n//     return (s.counter, s.version);\n// }\n```"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the high-level structure of a Tolk smart contract? List the main sections you would normally write."},
                {"role": "assistant", "content": "A typical Tolk smart contract contains:\n\n1. Imports (stdlib, gas-payments, etc.)\n2. `struct Storage { ... }` — persistent state layout\n3. Message structs with explicit opcodes using `struct(0x...)`\n4. `type AllowedMessage = MsgA | MsgB | ...;`\n5. `contract Name { storage: Storage; incomingMessages: AllowedMessage; }`\n6. `Storage.load()` / `Storage.save(self)` helper methods\n7. `fun onInternalMessage(in: InMessage) { ... }` — main entry point with `lazy` + `match`\n8. `get fun ...` methods for off-chain reads (getters)\n9. Optional external message handler and other lifecycle functions\n\nThis declarative style is one of Tolk's biggest advantages over older FunC."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How do I create a brand new Tolk counter project using the modern Acton toolchain?"},
                {"role": "assistant", "content": "```bash\n# Install Acton (one-time)\n# See https://ton-blockchain.github.io/acton/\n\n# Create a new project from the built-in counter template\nacton new my_counter --template counter\ncd my_counter\n\n# Build + test locally\nacton build\nacton test\n\n# Deploy to testnet (after configuring wallet in Acton.toml)\nacton deploy my_counter --net testnet\n```"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Explain the role of `lazy` when loading storage or messages in Tolk and why it is used."},
                {"role": "assistant", "content": "`lazy` in Tolk creates a lazy-loaded view of data from a cell/slice. It delays full deserialization until fields are actually accessed. This is critical for gas efficiency on TON because you only pay for the data you touch.\n\nExample:\n```tolk\nval storage = lazy Storage.load();   // cheap until you read fields\nstorage.counter += 1;                // now the struct is materialized\nstorage.save();\n```\n\nUsing `lazy` + `match` on messages is the idiomatic, gas-efficient pattern for `onInternalMessage`."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Write a Tolk contract skeleton that can receive an `IncreaseCounter` message and also supports a simple ownership check (only the owner can reset)."},
                {"role": "assistant", "content": "```tolk\nstruct Storage {\n    counter: uint32;\n    owner: address;\n}\n\nstruct(0x7e8764ef) IncreaseCounter { increaseBy: uint32; }\nstruct(0x3a752f06) ResetCounter {}\n\ntype AllowedMessage = IncreaseCounter | ResetCounter;\n\ncontract Counter {\n    storage: Storage;\n    incomingMessages: AllowedMessage;\n}\n\nfun onInternalMessage(in: InMessage) {\n    val msg = lazy AllowedMessage.fromSlice(in.body);\n    val storage = lazy Storage.load();\n\n    match (msg) {\n        IncreaseCounter => {\n            storage.counter += msg.increaseBy;\n            storage.save();\n            cashback(sender());\n        }\n        ResetCounter => {\n            if (in.sender != storage.owner) {\n                throw(401); // not owner\n            }\n            storage.counter = 0;\n            storage.save();\n            cashback(sender());\n        }\n    }\n}\n\nget fun currentCounter(): uint32 {\n    return lazy Storage.load().counter;\n}\n```"}
            ]
        }
    ]

    cells.append(code_cell([
        "# Seed dataset — replace / augment this with your real corpus",
        "raw_data = " + json.dumps(seed_examples, indent=2),
        "",
        "print(f'Seed examples: {len(raw_data)}')",
        "for i, ex in enumerate(raw_data):",
        "    print(f'{i}: {ex[\"messages\"][0][\"content\"][:60]}...')"
    ]))

    # ========== 10. CONVERT TO HF DATASET ==========
    cells.append(code_cell([
        "# Convert to Hugging Face Dataset with the required 'messages' column",
        "dataset = Dataset.from_list(raw_data)",
        "",
        "print('Dataset columns:', dataset.column_names)",
        "print('Number of examples:', len(dataset))",
        "print()",
        "print('Example 0 (first user turn):')",
        "print(dataset[0]['messages'][0])",
        "print()",
        "print('Assistant response length (chars):', len(dataset[0]['messages'][-1]['content']))"
    ]))

    # ========== 11. OPTIONAL — LOAD YOUR OWN DATA ==========
    cells.append(md_cell("""
## 9. (Optional) Load Your Own Real Data

Uncomment and adapt the cells below once you have real `.tolk` files and documentation.

**Recommended data sources**:
- Official docs cloned or copied into Drive
- `acton-contracts`, `tolk-bench`, Blueprint examples
- Your private contracts + high-quality explanations you write or generate

A simple but effective approach:
1. For every `.tolk` file → create 1–2 examples: "Write a contract that does X" → full source, and "Explain this Tolk code" → explanation.
2. For documentation chunks → "How do I ... in Tolk?" → answer.
"""))

    cells.append(code_cell([
        "# === EXAMPLE: Load from Google Drive ===",
        "# from google.colab import drive",
        "# drive.mount('/content/drive')",
        "",
        "# import glob",
        "# from pathlib import Path",
        "",
        "# tolk_files = glob.glob('/content/drive/MyDrive/ton-data/**/*.tolk', recursive=True)",
        "# print('Found .tolk files:', len(tolk_files))",
        "",
        "# def build_examples_from_files(paths):",
        "#     examples = []",
        "#     for p in paths:",
        "#         code = Path(p).read_text()",
        "#         name = Path(p).stem",
        "#         # Very naive — improve this with better heuristics or an LLM judge",
        "#         examples.append({",
        "#             'messages': [",
        "#                 {'role': 'user', 'content': f'Write a complete Tolk smart contract called {name} that follows modern TON practices.'},",
        "#                 {'role': 'assistant', 'content': f'```tolk\\n{code}\\n```'}",
        "#             ]",
        "#         })",
        "#         # Add an 'explain' example too",
        "#         examples.append({",
        "#             'messages': [",
        "#                 {'role': 'user', 'content': f'Explain the storage layout and message handling in this Tolk contract:\\n\\n{code[:1500]}'},",
        "#                 {'role': 'assistant', 'content': 'This contract uses ...'}  # fill with real analysis or LLM-generated",
        "#             ]",
        "#         })",
        "#     return examples",
        "",
        "# your_examples = build_examples_from_files(tolk_files[:50])   # start small",
        "# dataset = Dataset.from_list(your_examples + raw_data)   # merge with seed"
    ]))

    # ========== 12. SFT CONFIG ==========
    cells.append(md_cell("""
## 10. SFTTrainer Configuration

Key settings for a Colab Hello World:
- Very small batch size + gradient accumulation
- Short context (1024–2048 is plenty for most contract files)
- 1–3 epochs on the tiny seed (you will see loss drop fast)
- `max_length` controls truncation of the full chat-formatted example
"""))

    cells.append(code_cell([
        "training_args = SFTConfig(",
        "    output_dir='./ton-tolk-sft-lora',",
        "    num_train_epochs=2,                 # 1-3 is typical for tiny seed sets",
        "    per_device_train_batch_size=1,",
        "    gradient_accumulation_steps=8,      # effective batch = 8",
        "    learning_rate=2e-4,",
        "    lr_scheduler_type='cosine',",
        "    warmup_ratio=0.1,",
        "    max_length=1536,                    # adjust down if OOM",
        "    logging_steps=2,",
        "    save_strategy='epoch',",
        "    save_total_limit=2,",
        "    report_to='none',                   # disable wandb etc. in Colab",
        "    # dataset_text_field is NOT needed when using the 'messages' format",
        "    # packing=True can be tried for higher throughput on longer sequences",
        ")",
        "",
        "print('SFTConfig ready. Output dir:', training_args.output_dir)"
    ]))

    # ========== 13. CREATE TRAINER ==========
    cells.append(code_cell([
        "trainer = SFTTrainer(",
        "    model=model,",
        "    args=training_args,",
        "    train_dataset=dataset,",
        "    # peft_config is already applied via get_peft_model above.",
        "    # If you want SFTTrainer to apply LoRA itself, pass peft_config=lora_config instead of calling get_peft_model.",
        "    tokenizer=tokenizer,   # some versions still benefit from explicit tokenizer",
        ")",
        "",
        "print('Trainer initialized. Starting training...')"
    ]))

    # ========== 14. TRAIN ==========
    cells.append(md_cell("## 11. Run Training"))

    cells.append(code_cell([
        "trainer.train()",
        "",
        "print('\\nTraining finished!')",
        "print('Best model checkpoint (if any):', trainer.state.best_model_checkpoint)"
    ]))

    # ========== 15. SAVE ADAPTER ==========
    cells.append(code_cell([
        "# Save the LoRA adapter (tiny — only a few MB)",
        "adapter_path = './ton-tolk-lora-adapter'",
        "trainer.save_model(adapter_path)",
        "tokenizer.save_pretrained(adapter_path)",
        "",
        "print('Adapter saved to:', adapter_path)",
        "print('Files:', os.listdir(adapter_path))"
    ]))

    # ========== 16. INFERENCE ==========
    cells.append(md_cell("""
## 12. Test the Fine-Tuned Model (Inference)

We load the base model again + the trained LoRA adapter for generation.

You can also merge the adapter for a standalone model (see next section).
"""))

    cells.append(code_cell([
        "# Reload base + adapter for clean inference",
        "print('Reloading base model for inference...')",
        "",
        "# Build clean kwargs for the inference load (4-bit or bf16/fp16)",
        "infer_kwargs = {",
        "    'trust_remote_code': True,",
        "    'device_map': 'auto',",
        "    'token': HF_TOKEN,",
        "}",
        "if precision['quantization_config'] is not None:",
        "    infer_kwargs['quantization_config'] = precision['quantization_config']",
        "else:",
        "    infer_kwargs['torch_dtype'] = precision['torch_dtype']",
        "",
        "base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **infer_kwargs)",
        "",
        "ft_model = PeftModel.from_pretrained(base_model, adapter_path)",
        "ft_model.eval()",
        "",
        "print('Fine-tuned (LoRA) model ready for inference.')"
    ]))

    cells.append(code_cell([
        "def ask_ton(model, tokenizer, prompt: str, max_new_tokens: int = 512):",
        "    messages = [",
        "        {'role': 'system', 'content': 'You are an expert TON smart contract engineer specializing in Tolk and the Acton toolchain.'},",
        "        {'role': 'user', 'content': prompt}",
        "    ]",
        "    input_ids = tokenizer.apply_chat_template(",
        "        messages,",
        "        add_generation_prompt=True,",
        "        return_tensors='pt'",
        "    ).to(model.device)",
        "",
        "    with torch.no_grad():",
        "        output = model.generate(",
        "            input_ids,",
        "            max_new_tokens=max_new_tokens,",
        "            temperature=0.2,",
        "            top_p=0.9,",
        "            do_sample=True,",
        "            pad_token_id=tokenizer.eos_token_id,",
        "            eos_token_id=tokenizer.eos_token_id,",
        "        )",
        "    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)",
        "    return response.strip()",
        "",
        "# === Test prompts ===",
        "prompts = [",
        "    'Write a Tolk contract for a simple counter that only the owner can reset.',",
        "    'Explain how to use lazy loading and match in a Tolk onInternalMessage handler.',",
        "    'Show the storage struct and a getter for a minimal TON NFT collection contract.',",
        "]",
        "",
        "for p in prompts:",
        "    print('='*70)",
        "    print('USER:', p)",
        "    print('-'*70)",
        "    ans = ask_ton(ft_model, tokenizer, p)",
        "    print('ASSISTANT:')",
        "    print(ans)",
        "    print()"
    ]))

    # ========== 17. MERGE & FULL MODEL ==========
    cells.append(md_cell("""
## 13. (Optional) Merge LoRA into Base Weights

Merging produces a standalone model (no PEFT dependency at inference time). Useful if you want to:
- Push a single model to the Hub
- Use with vLLM / standard `transformers` pipelines without `peft`
- Distribute a ready-to-use checkpoint

**Warning**: Merged 3B–7B bf16/16bit models are large (several GB). In Colab you may want to save only the adapter and merge later on a machine with more disk.
"""))

    cells.append(code_cell([
        "# Merge adapter into base (creates full weights)",
        "print('Merging LoRA weights into base model...')",
        "merged_model = ft_model.merge_and_unload()",
        "",
        "merge_path = './ton-tolk-merged'",
        "merged_model.save_pretrained(merge_path, safe_serialization=True)",
        "tokenizer.save_pretrained(merge_path)",
        "",
        "print('Merged model saved to:', merge_path)",
        "print('Size on disk will be much larger than the adapter.')"
    ]))

    # ========== 18. PUSH TO HUB (OPTIONAL) ==========
    cells.append(md_cell("## 14. Push to Hugging Face Hub (Optional)"))

    cells.append(code_cell([
        "# from huggingface_hub import login",
        "# login()   # or use HF_TOKEN",
        "",
        "# adapter only (recommended for most people)",
        "# ft_model.push_to_hub('your-username/ton-tolk-llama3.2-3b-lora')",
        "# tokenizer.push_to_hub('your-username/ton-tolk-llama3.2-3b-lora')",
        "",
        "# full merged model (heavier)",
        "# merged_model.push_to_hub('your-username/ton-tolk-llama3.2-3b-merged')",
        "# tokenizer.push_to_hub('your-username/ton-tolk-llama3.2-3b-merged')"
    ]))

    # ========== 15. ACTON TOOLCHAIN ==========
    cells.append(md_cell("""
## 15. Acton Toolchain — Full Modern TON/Tolk Dev Kit (Audit Complete)

**Acton** (https://ton-blockchain.github.io/acton/docs/welcome) is the official all-in-one, Rust-based, **Tolk-first** toolchain.

It covers the entire lifecycle:
- Project scaffolding (`acton new --template counter|jetton|nft|empty [--app]`, `acton init`)
- `Acton.toml` manifest (rich config for contracts, build, test, lint, wrappers, scripts, import mappings, networks, etc.)
- `acton build` + single-file `compile`
- Wrapper generation (Tolk + TypeScript via `acton wrapper --ts`)
- World-class testing (`acton test`): emulator, matchers (`@acton/testing/expect`), coverage, gas snapshots/baselines, mutation testing, fuzz, fork-net (real chain state), Test UI (browser traces), `--debug` (DAP source-level debugger)
- `acton script` (Tolk files as first-class deployment/automation scripts — emulate locally then `--net testnet`)
- Wallets with testnet airdrop (`acton wallet new --local --airdrop`)
- On-chain verification, library publishing, RPC inspection, `disasm`, `retrace`
- Linting (`acton check`), formatting (`acton fmt`), Git hooks, `doctor`
- `acton func2tolk` for FunC → Tolk migration
- Docker image + GitHub Action (`ton-blockchain/setup-acton`) + `acton up` for updates

**Reference material goldmines** (use for dataset curation / RAG):
- Official docs + `https://ton-blockchain.github.io/acton/llms-full.txt`
- TON Docs `https://docs.ton.org/llms.txt` + Tolk sections (overview, idioms-conventions, features/*, examples)
- Canonical contracts: https://github.com/ton-blockchain/acton-contracts (counter, jettons, NFT, w5, highload, multisig, dns, elector, ...)
- **Agent Skills** (exactly aligned with our SFT goal): https://github.com/ton-blockchain/skills (`acton/`, `tolk/`, `func2tolk/`, `ton-blockchain/` — install with `npx skills add -g ...`; each has `SKILL.md` + `references/` with checklists, command maps, idiomatic patterns, porting guides). Mention `$acton`, `$tolk` etc. in agent prompts.
- IDE: Official TON VS Code extension (full Tolk LSP + Acton project integration). Also JetBrains/Cursor/Zed via LSP (`acton ls`).

**Why this is critical for the SFT**:
After inference, paste the generated ```tolk block into a real project and run the *actual* compiler + test suite. Contracts that `build` cleanly + pass `test` (ideally with coverage/gas checks) are the only ones that count. The skills + llms-*.txt + acton-contracts are the best sources for high-quality training data and system prompts.

**Colab notes**:
- Ephemeral → re-run installer every session.
- We handle PATH for the kernel.
- Full build + test works great. Browser Test UI and full DAP stepping are limited (you can still use `--backtrace full`, gas snapshots, coverage). Deployment/verification needs testnet interaction (Acton airdrop + scripts make it feasible but use with care and never on mainnet without review).
- Alternative: Docker (`ghcr.io/ton-blockchain/acton:1.1.0 build ...`) for stronger reproducibility.
"""))

    cells.append(code_cell([
        "# === Install the Acton basic kit (execute in every new Colab session) ===",
        "# Official one-liner from https://ton-blockchain.github.io/acton/ and the GitHub releases (v1.1.0+ as of 2026)",
        "!curl -LsSf https://github.com/ton-blockchain/acton/releases/latest/download/acton-installer.sh | sh",
        "",
        "# Make the binary visible to this notebook kernel (Colab shells can be picky with PATH)",
        "import os, shutil, subprocess",
        "home = os.path.expanduser('~')",
        "for p in [f'{home}/.local/bin', '/usr/local/bin', f'{home}/.cargo/bin']:",
        "    if os.path.isdir(p):",
        "        os.environ['PATH'] = p + ':' + os.environ.get('PATH', '')",
        "",
        "acton_bin = shutil.which('acton')",
        "print('acton found at:', acton_bin or 'NOT ON PATH')",
        "!acton --version || echo '>>> Re-run the curl installer or adjust PATH manually'",
        "",
        "# Quick environment health check (very useful)",
        "!acton doctor || echo 'doctor not available or warnings above'"
    ]))

    cells.append(code_cell([
        "# === Quality gates on any project (fmt, lint, build, test) — run these on model output ===",
        "import os, textwrap",
        "os.chdir('/tmp/sft_acton_demo')  # from previous cell or re-create",
        "",
        "print('=== Formatter ===')",
        "!acton fmt --check || echo 'Formatting issues (run acton fmt to fix)'",
        "",
        "print('\\n=== Linter ===')",
        "!acton check --output-format plain | head -30 || echo 'Lint issues found'",
        "",
        "print('\\n=== Build ===')",
        "!acton build",
        "",
        "print('\\n=== Test (core validation) ===')",
        "!acton test 2>&1 | tail -15",
        "",
        "print('\\n=== Optional advanced signals ===')",
        "!acton test --coverage --coverage-format text 2>&1 | tail -5 || true",
        "# Gas snapshot example (compare against baseline):",
        "# acton test ... --baseline-snapshot path/to/baseline.json --fail-on-diff",
        "",
        "os.chdir('/content') if os.path.isdir('/content') else None"
    ]))

    cells.append(code_cell([
        "# === Realistic end-to-end: take model-generated Tolk and validate with full Acton kit ===",
        "import os, textwrap, pathlib",
        "",
        "# 1. (Re)create a clean demo project",
        "!acton new /tmp/sft_full_demo --template counter --force 2>/dev/null || true",
        "demo_root = pathlib.Path('/tmp/sft_full_demo')",
        "os.chdir(demo_root)",
        "",
        "# 2. Simulate output from the fine-tuned model (in real use, copy the ```tolk ... ``` block from an inference cell)",
        "generated_counter = textwrap.dedent('''",
        "    import \"@stdlib/gas-payments\" tolk 1.1",
        "",
        "    struct Storage { counter: uint32; owner: address; }",
        "",
        "    struct(0x7e8764ef) IncreaseCounter { increaseBy: uint32; }",
        "    struct(0x3a752f06) ResetCounter {}",
        "    type AllowedMessage = IncreaseCounter | ResetCounter;",
        "",
        "    contract Counter {",
        "        storage: Storage;",
        "        incomingMessages: AllowedMessage;",
        "    }",
        "",
        "    fun Storage.load() { return Storage.fromCell(contract.getData()); }",
        "    fun Storage.save(self) { contract.setData(self.toCell()); }",
        "",
        "    fun onInternalMessage(in: InMessage) {",
        "        val msg = lazy AllowedMessage.fromSlice(in.body);",
        "        val storage = lazy Storage.load();",
        "        match (msg) {",
        "            IncreaseCounter => { storage.counter += msg.increaseBy; storage.save(); cashback(sender()); }",
        "            ResetCounter => {",
        "                if (in.sender != storage.owner) { throw(401); }",
        "                storage.counter = 0; storage.save(); cashback(sender());",
        "            }",
        "        }",
        "    }",
        "",
        "    get fun currentCounter(): uint32 { return lazy Storage.load().counter; }",
        "''')",
        "",
        "# Write it into the project's contracts location (adjust filename to match your Acton.toml)",
        "contract_path = demo_root / 'contracts' / 'counter.tolk'",
        "contract_path.parent.mkdir(parents=True, exist_ok=True)",
        "contract_path.write_text(generated_counter)",
        "print('Wrote model output to', contract_path)",
        "",
        "# 3. Run the full modern validation pipeline",
        "print('\\n--- acton build (real Tolk compiler + ABI + wrappers) ---')",
        "!acton build",
        "",
        "print('\\n--- acton wrapper (regenerate if ABI changed) ---')",
        "!acton wrapper Counter --test 2>/dev/null || true",
        "",
        "print('\\n--- acton check (linter) + fmt --check ---')",
        "!acton check --output-format plain | cat",
        "!acton fmt --check || echo '(fmt would fix some things)'",
        "",
        "print('\\n--- acton test (emulator + matchers + any coverage/gas) ---')",
        "!acton test 2>&1 | tail -25",
        "",
        "# 4. (Optional) Emulate a deployment script locally before any on-chain spend",
        "# !acton script scripts/deploy.tolk   # (the template usually has one)",
        "",
        "os.chdir('/content') if os.path.isdir('/content') else os.chdir(os.path.expanduser('~'))",
        "print('\\n✅ Full dev loop complete. Generated contracts that survive build + check + test are high quality.')"
    ]))

    cells.append(md_cell("""
## 15b. Key Documentation Sources (for Training Data, RAG, and Reference)

These are the authoritative sources we audited. Use them to expand your dataset, build RAG, or as "ground truth" for the fine-tuned model.

**Acton (primary toolchain docs & source)**
- Landing: https://ton-blockchain.github.io/acton/
- Welcome & navigation: https://ton-blockchain.github.io/acton/docs/welcome
- Installation: https://ton-blockchain.github.io/acton/docs/installation
- Full source tree (raw .mdx): https://github.com/ton-blockchain/acton/tree/master/docs/content/docs
  - Key files: welcome.mdx, quickstart.mdx, walkthrough.mdx, tutorial/*, acton-toml.mdx (full schema), commands/overview.mdx + per-command, testing/*, debug.mdx, deploy.mdx, verify.mdx, lint.mdx, scripting/*, agent-skills/*, projects.mdx, ci-setup.mdx, dapps.mdx, etc.
- llms-full.txt (huge concatenated docs for LLMs): https://ton-blockchain.github.io/acton/llms-full.txt
- Examples repo (gold for contracts/tests/scripts): https://github.com/ton-blockchain/acton-contracts
- Agent Skills (for agentic workflows & high-quality instruction data): https://github.com/ton-blockchain/skills (install with `npx skills add -g ...`; use $acton / $tolk etc.)

**TON Official Docs (concepts, standards, Tolk language deep dive)**
- Site: https://docs.ton.org/
- Source repo: https://github.com/ton-org/docs (content/ is all the MDX)
- LLM-friendly index: https://docs.ton.org/llms.txt (and per-section llms.mdx paths)
- Critical Tolk sections (from llms structure):
  - Overview, Basic syntax, Idioms and conventions (typed structs, opcode messages with `struct(0x...)`, unions + `lazy AllowedMessage.fromSlice` + `match`, Storage.load/save, lazy loading, auto-serialization, onInternalMessage/onBouncedMessage, getters, createMessage + send modes, maps, etc.)
  - Type system (detailed), Syntax details (pattern matching, mutability, imports), Language features (message handling, contract storage/getters/ABI, sending, lazy, jetton payload, stdlib, asm, optimizations)
  - Migrating from FunC (comparison + converter = `acton func2tolk`)
  - Examples
- Broader relevant sections: Blockchain basics → Smart contract development (Acton + Blueprint), Tolk, techniques (security, gas estimation, on-chain libraries, etc.), Primitives (cells/BoC, addresses, messages, actions, fees, traces), TVM, Standard contracts (wallets V5/highload, Jettons, NFTs), APIs (TON Center), TON Connect, etc.

**Cross-references**
- Acton docs repeatedly point to TON Tolk docs for language details and to acton-contracts for patterns.
- TON docs list Acton as the modern toolchain and still document Blueprint as a popular alternative (JS/TS based, also supports Tolk).
- Both provide llms.txt/llms-full.txt for exactly LLM/agent consumption — ideal for synthetic data or RAG in your pipeline.

**In this notebook**
- The Acton cells above let you validate model outputs against the real tools described in these docs.
- For dataset work: `curl` the llms.txt files, chunk the Tolk idioms/features pages, pair with acton-contracts sources, and generate (instruction, strict Tolk code + explanation) examples.
- Recommended reading order for a specialist model: Acton quickstart/walkthrough + acton-toml + testing + commands, then TON Tolk overview + idioms-conventions + features/* + migration, then standards (Jetton/NFT/wallet) + security/gas techniques.

Always cross-check generated code by running `acton build`, `acton check`, `acton fmt --check`, and `acton test` (as demonstrated).
"""))

    cells.append(md_cell("""
## 15c. Scaling Your Dataset — From Hello World to Real Specialization

The current seed (8 examples) is only for the "Hello World" pipeline demo. For a useful specialized model you need **hundreds to a few thousand high-quality conversational examples**.

**Best sources** (all audited above):
- `acton-contracts` — real, production-grade Tolk + tests + scripts.
- TON `llms.txt` + Tolk pages (idioms, features, examples).
- Acton `llms-full.txt` + docs.
- Skills repo `SKILL.md` + `references/` (excellent structured instructions + checklists).
- Your own .tolk files + docs.

**Recommended strategy**:
1. Scrape / clone the sources.
2. For each .tolk file: create 2–3 pairs ("Write a Tolk contract that does X following modern Acton/Tolk idioms", full code) + ("Explain / review / improve this Tolk", analysis).
3. For doc chunks: "How do I ... in Tolk/Acton?" → accurate answer from the page.
4. Include "fix this" and "migrate from FunC" examples.
5. Always validate candidates with `acton build + check + test` before including.
6. Hold out 10-20% for manual spot-checking + Acton validation rate.

Below is a starter helper you can run/extend in Colab (mount Drive or clone repos).
"""))

    cells.append(code_cell([
        "# === Starter: Data ingestion helper (expand this for real corpus) ===",
        "import os, glob, textwrap",
        "from datasets import Dataset",
        "",
        "# Example: simplistic converter. Improve with better chunking, teacher model for explanations,",
        "# or LLM-as-judge filtering.",
        "",
        "def tolk_file_to_examples(filepath, max_chars=2000):",
        "    code = open(filepath).read()[:max_chars]",
        "    name = os.path.basename(filepath).replace('.tolk', '')",
        "    examples = []",
        "    # Code generation example",
        "    examples.append({",
        "        'messages': [",
        "            {'role': 'system', 'content': 'You are an expert TON smart contract engineer specializing in Tolk and the Acton toolchain.'},",
        "            {'role': 'user', 'content': f'Write a clean, modern Tolk smart contract called {name} that follows current Acton best practices and idioms.'},",
        "            {'role': 'assistant', 'content': f'```tolk\\n{code}\\n```'}",
        "        ]",
        "    })",
        "    # Explanation / review example (in real use, generate a good explanation or use doc context)",
        "    examples.append({",
        "        'messages': [",
        "            {'role': 'user', 'content': f'Explain the storage layout, message handling, and key idioms in this Tolk contract:\\n\\n{code[:800]}...'},",
        "            {'role': 'assistant', 'content': 'This contract uses a typed Storage struct with ... The main entrypoint uses lazy + match for safe message dispatch. ... (expand with real analysis)'}",
        "        ]",
        "    })",
        "    return examples",
        "",
        "# Usage example (uncomment and adapt after cloning repos into /content or Drive)",
        "# tolk_files = glob.glob('/content/acton-contracts/**/contracts/*.tolk', recursive=True)",
        "# print('Found .tolk files:', len(tolk_files))",
        "# new_examples = []",
        "# for f in tolk_files[:50]:  # start small",
        "#     new_examples.extend(tolk_file_to_examples(f))",
        "",
        "# Then merge with seed and push to HF or save as JSONL:",
        "# full_dataset = Dataset.from_list(raw_data + new_examples)",
        "# full_dataset.to_json('ton_tolk_sft.jsonl')",
        "# full_dataset.push_to_hub('your-username/ton-tolk-sft-v1')",
        "",
        "print('Data prep skeleton ready. Clone acton-contracts + curl llms.txt, then scale this logic.')"
    ]))

    # ========== 16. NEXT STEPS (renumbered) ==========
    cells.append(md_cell("""
## 16. Next Steps & Best Practices

### Improving Data Quality (most important)
- Extract real contracts + surrounding comments/docs from official sources and real repos (acton-contracts, tolk-bench, etc.)
- Create diverse instructions: "Implement X using Tolk best practices", "Audit this contract for ...", "Migrate this FunC pattern to Tolk"
- Use a stronger teacher model (GPT-4o, Claude 3.5, Qwen2.5-72B, etc.) to generate high-quality explanations and variations
- Keep a held-out validation set of prompts (manual review or exact match on key tokens)

### Scaling Training
- Increase dataset to hundreds/thousands of examples
- Use `packing=True` in SFTConfig for better throughput
- Try higher rank (32–64) or more targeted modules once you have real data volume
- Consider Unsloth (community favorite for Colab speed) as a drop-in accelerator while keeping the same `trl`/`peft` concepts

### Evaluation Ideas
- **Best signal**: Take every generated contract from the inference cells, write it to disk, and run `acton build` + `acton test` using the cells in the "Acton Toolchain" section above. A contract that compiles and whose tests pass is dramatically more valuable than one that only looks plausible.
- Unit tests on generated code (if you have a test harness)
- "Needle" tests: specific TON idioms, gas costs, message opcodes, etc. that only a properly specialized model should know

### Production Tips
- Always review & test generated smart contracts with the real toolchain before any deployment
- Fine-tuned models can still hallucinate dangerous patterns (re-entrancy, missing owner checks, wrong opcodes, unsafe lazy usage, etc.)
- Combine the fine-tuned model with RAG over the official Acton + Tolk docs + verified contracts for higher reliability

**You now have a complete, minimal, reproducible SFT pipeline targeting the TON/Tolk/Acton domain — including the ability to compile and test outputs with the actual developer toolchain inside Colab.**

Happy building on The Open Network!
"""))

    # Final metadata
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook


def main():
    nb = build_notebook()
    out_path = Path("notebooks/TON_Tolk_SFT_HelloWorld.ipynb")
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Notebook written to {out_path.resolve()}")
    print(f"Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
