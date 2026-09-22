# Fixed checkpoint and tokenizer revisions

Model and tokenizer revisions are identical for these checkpoints. Values below are read from the executed campaign D YAMLs and agree with their A/B/C model configurations. Layer indices are absolute and zero-based; routed-site counts do not include dense-only decoder blocks.

| Alias | Upstream model | Model/tokenizer revision | Local D layer | Window layer | Routed sites |
| --- | --- | --- | ---: | ---: | ---: |
| `qwen36_moe` | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0) | `995ad96eacd98c81ed38be0c5b274b04031597b0` | 20 | 20 | 40 |
| `qwen3_moe` | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39) | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | 36 | 24 | 48 |
| `glm4_moe_lite` | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash/tree/7dd20894a642a0aa287e9827cb1a1f7f91386b67) | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | 24 | 24 | 46 |
| `nemotron35_lightning` | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/tree/a9904d24bcc1d289a1950fa9d2b978c47cf903b9) | `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` | 27 | 27 | 23 |
| `nemotron3_nano` | [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/tree/2d59de1cbd51c0adf384eb906b766d1aee0e0517) | `2d59de1cbd51c0adf384eb906b766d1aee0e0517` | 27 | 27 | 23 |
| `gemma4_moe` | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52) | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | 15 | 15 | 30 |
| `olmoe` | [allenai/OLMoE-1B-7B-0924-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct/tree/7f1c97f440f06ce36705e4f2b843edb5925f4498) | `7f1c97f440f06ce36705e4f2b843edb5925f4498` | 8 | 8 | 16 |
| `gpt_oss_20b` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) | `6cee5e81ee83917806bbde320786a8fb61efebee` | 12 | 12 | 24 |
| `gpt_oss_120b` | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a) | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | 18 | 18 | 36 |

GPT-OSS native MXFP4 kernels: `kernels-community/gpt-oss-triton-kernels`, revision `642d2a71c6a9b32fc18acb8ec53505fb324bc394`; package `kernels==0.12.3`. No model weights, tokenizer payloads or downloaded third-party kernel code are redistributed here. Follow the upstream terms for each fixed model and runtime.
