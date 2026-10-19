# Third-party license and attribution index

This index accompanies an **inactive licensing proposal**; it preserves evidence
of existing upstream terms and does not impose new restrictions. It is not a
claim that the entire repository has one license or that this initial audit has
resolved every upstream notice obligation. Original source headers are unchanged.

| Material / existing notice | License text included here | Evidence and limits |
| --- | --- | --- |
| 135 of the 137 tracked files under `peft/` have Hugging Face Apache-2.0 headers, including HypLoRA's modified LoRA files. The headers identify the HuggingFace Inc. team, with 2023-, 2024-, or 2025-present dates. | [Apache-2.0](LICENSES/Apache-2.0.txt) | Copied verbatim from the [Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt). Its license wording matches the [PEFT v0.15.2 license](https://github.com/huggingface/peft/blob/3c7b6e7f0252cb18386d86b056a9b100a8160792/LICENSE), ignoring whitespace. The actual vendored snapshot is not established by the version label. |
| Microsoft LoRA-derived material in `peft/tuners/lora/layer.py`, with a Microsoft Corporation copyright and MIT notice; `peft/utils/save_and_load.py` also credits LoRA utilities. | [Microsoft LoRA MIT](LICENSES/MIT-LoRA.txt) | Exact text from [LoRA revision a0a92e0](https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/LICENSE.md); identical to the license at the other revision, [4c03338](https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/LICENSE.md), referenced by `layer.py`. Retain the source's `Copyright (c) Microsoft Corporation. All rights reserved.` notice as well. Exact copied-section boundaries remain to be traced. |
| `peft/tuners/_buffer_dict.py`: `Copyright (c) Meta Platforms, Inc. and affiliates.`; explicitly MIT, adapted from BoTorch. | [BoTorch MIT](LICENSES/MIT-BoTorch.txt) | Exact text from [BoTorch revision b0dd01b](https://github.com/pytorch/botorch/blob/b0dd01be61d517c387e234903a6e2cc38c057ea2/LICENSE). This supplies the named MIT text and matching attribution; the original copied revision is unresolved. |
| `peft/tuners/boft/fbd/__init__.py` | Empty file; no header | Matches the empty file in the inspected PEFT tag. No independent ownership or license conclusion is inferred from an empty file. |

The Apache and MIT texts above reproduce upstream licenses for their respective
material; their presence is not a new repository-wide grant by marlin-codes.

Additional embedded provenance needs review. Existing references include
Transformers, Accelerate, Diffusers, LyCORIS, scikit-learn, CorDA, LoftQ, NeMo,
P-tuning-v2, LLaMA-Adapter, Microsoft MTTL, LoRA-Torch, and kohya/sd-scripts. In
particular, `peft/utils/incremental_pca.py` says it adapts scikit-learn code, while
`peft/tuners/loha/layer.py` and `peft/tuners/lokr/layer.py` describe direct LyCORIS
copies. A PEFT Apache header alone does not settle all embedded attribution duties.
The source references remain intact; verify the copied revisions and any further
required texts/notices before a licensing release.

Training/evaluation scripts outside `peft/` also require provenance review: matching
passages were found in Apache-licensed Alpaca-LoRA and LLM-Adapters sources. Their
presence is evidence against assuming headerless scripts are wholly original.
See [PROVENANCE.md](docs/licensing/PROVENANCE.md) for the pinned comparison links,
complete path inventory, and review limits.

## License-text integrity

SHA-256 of the exact bytes supplied in this proposal:

| File | SHA-256 |
| --- | --- |
| `LICENSES/Apache-2.0.txt` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSES/MIT-LoRA.txt` | `c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383` |
| `LICENSES/MIT-BoTorch.txt` | `da6d3703ed11cbe42bd212c725957c98da23cbff1998c05fa4b3d976d1a58e93` |
