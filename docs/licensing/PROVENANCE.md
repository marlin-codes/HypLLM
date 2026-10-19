# Provenance and proposed scope inventory

**DRAFT FOR REVIEW — NO MATERIAL APPROVED FOR A NEW GRANT.**

This records what can be established from the inspected source and history. It is
not a certification of ownership, a complete historical distribution audit, or a
replacement for applicable upstream licenses. Read [LICENSING.md](../../LICENSING.md)
and the [notice index](../../THIRD_PARTY_NOTICES.md) alongside it.

## Inspected baseline and complete path boundary

- Repository: [marlin-codes/HypLoRA](https://github.com/marlin-codes/HypLoRA).
- Audited `main`: [4e919482a8bda623a12d8251957a78547f42b919](https://github.com/marlin-codes/HypLoRA/commit/4e919482a8bda623a12d8251957a78547f42b919).
- Complete reachable `main` history in the non-shallow checkout contains two
  commits: initial [66df99a412c6f1166073127f3f9da535dbc084c0](https://github.com/marlin-codes/HypLoRA/commit/66df99a412c6f1166073127f3f9da535dbc084c0)
  and the baseline above, both dated 2026-03-29. The second adds two delta-analysis
  scripts and two figure artifacts; it does not change the PEFT files.
- Neither commit contains a root `LICENSE`, `NOTICE`, or `COPYING`. No repository
  instructions, license manifest, or contribution agreement were found in that
  tree. This does not establish that no permission was granted elsewhere.
- [FILE_INVENTORY.csv](FILE_INVENTORY.csv) enumerates **all 271 tracked baseline
  paths**, each with its Git blob ID, observed notice/category, draft scope status,
  and PEFT comparison result where applicable. Its records refer only to the
  pinned baseline, not to later edits or these newly drafted documents.

| Exact baseline group | Count | Known evidence | Proposed treatment |
| --- | ---: | --- | --- |
| All tracked `peft/**` paths in the CSV | 137 | 135 Apache-header files, one Meta MIT file, one empty initializer; Microsoft MIT material is additionally credited inside LoRA files. | Entire group excluded from new research-only restrictions, including current project modifications. Preserve existing rights; review embedded provenance. |
| `dataset/**/*.json` and `ft-training_set/*.json` | 46 | Benchmark and training-data files; not code. No dataset-specific licensing manifest found in this tree. | Excluded; determine each source, revision, transformations, and redistribution permissions separately. |
| `figs/**`, `utils/figs/**`, and `utils/results/**` | 23 | 22 PNGs and one PDF. | Excluded; determine figure/output ownership and underlying data/model conditions separately. |
| Remaining paths, individually enumerated as `unresolved-code-docs-config` in the CSV | 65 | Eight root Python training/evaluation files, `dataset/mawps/data_process.py`, five `utils/*.py` files, 48 `scripts/**/*.sh` launchers, `setup.sh`, `requirements.txt`, and baseline `README.md`. No express license header found in these files. | Unresolved and excluded pending review; absence of a header does not prove originality or freedom to relicense. |

No tracked model weights or adapter/checkpoint files were found among those 271
paths. Referenced or downloaded base models, tokenizers, weights and later
checkpoints are outside this code proposal. Generated text, metrics and other
outputs receive no blanket grant or restriction from it; assess any protected code
or third-party content they contain and the applicable input/model terms.

## Core modifications were already distributed with permissive notices

The initial commit includes
[`peft/tuners/lora/layer.py`](https://github.com/marlin-codes/HypLoRA/blob/66df99a412c6f1166073127f3f9da535dbc084c0/peft/tuners/lora/layer.py),
including the `hyplora` and `hyplora_simplified` branches, curvature/scale state,
and associated helper functions, under its existing Apache-2.0 header. The file
also carries Microsoft LoRA's MIT attribution. HypLoRA's `lora_type` configuration
and dispatch additions in
[`config.py`](https://github.com/marlin-codes/HypLoRA/blob/66df99a412c6f1166073127f3f9da535dbc084c0/peft/tuners/lora/config.py)
and [`model.py`](https://github.com/marlin-codes/HypLoRA/blob/66df99a412c6f1166073127f3f9da535dbc084c0/peft/tuners/lora/model.py)
likewise appeared with Apache headers. No separate restrictive notice for those
modifications was found in the inspected history.

At the baseline, `layer.py` lines 1–13 contain the Apache header, 592–598 the
Microsoft attribution, 761–787 the simplified mapping helpers, and 822–875 the
HypLoRA forward branches. Its SHA-256 in both commits is
`2e8db82c3baa4f0ea83f1c4ab588c41aabad717c67eee62ec7f44110faca500f`.
These line ranges locate evidence, not an exhaustive ownership boundary.

This is evidence that project modifications may already have been granted under
the permissive notices accompanying publication. The exact legal effect and any
earlier/off-repository grants require review. It would be misleading to promise
that commercial users of this existing core must purchase a new license. Prior
valid grants cannot be retroactively revoked by this proposal. Author/ownership
authorization does not by itself undo prior grants. Future separately identified
contributions may be considered for new terms only after their rights and existing
grant obligations are checked; changing a path/header does not reset those rights.

## PEFT version and upstream comparison limits

The vendored `peft/__init__.py` reports `0.15.2.dev0`; `setup.sh` installs
`peft==0.15.2`. Neither establishes the actual source revision. A byte comparison
against PEFT's [`v0.15.2` commit
3c7b6e7f0252cb18386d86b056a9b100a8160792](https://github.com/huggingface/peft/tree/3c7b6e7f0252cb18386d86b056a9b100a8160792/src/peft)
found **75 identical files, 54 differing files, and 8 paths absent from that
tag**. The CSV records the result for every PEFT file, using the same relative
path beneath upstream `src/peft/`. Differences from this tag are **not** a list
of owner-authored changes; many may be changes from other upstream revisions.

The inspected PEFT tag contains an Apache `LICENSE` and no `NOTICE` file. That
observation does not prove that the actual copied snapshot or its embedded sources
had no additional notice obligations. The Meta MIT `_buffer_dict.py` and empty
`boft/fbd/__init__.py` match this tag byte for byte. The notice index supplies the
Apache and named MIT texts without replacing existing source headers.

## Headerless scripts: positive evidence of shared source

The baseline [`finetune_qwen.py`](https://github.com/marlin-codes/HypLoRA/blob/4e919482a8bda623a12d8251957a78547f42b919/finetune_qwen.py)
has a contiguous 33-line exact match at lines 180–212 with:

- [Alpaca-LoRA `finetune.py`, lines 212–244, revision 8bb8579](https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L212),
  whose root [license is Apache-2.0](https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/LICENSE).
- [LLM-Adapters `finetune.py`, lines 253–285, revision 8166572](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/816657208af4db747803f87ba40a4c71383fed7a/finetune.py#L253),
  whose root [license is Apache-2.0](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/816657208af4db747803f87ba40a4c71383fed7a/LICENSE).

There are further matching blocks. These comparisons establish shared text, not
the actual import path, date, author, or full scope of derivatives. Check all
related fine-tuning/evaluation files and their original source distributions
before selecting any of their portions for new terms or asserting notice
completeness. Other headerless utilities, scripts and documentation remain
unverified rather than presumed original.

## Approved scope schedule

**Empty. No current path, blob, or contribution is approved for the proposed new
grant.** The two draft agreements and this inventory are review documents, not
scope designations. Newly written proposal text is not a license to the existing
repository assets. Do not replace this section with a broad "all files except"
rule; any later approved schedule must identify exact material and rights.

## Questions to resolve before adoption

1. What precise upstream snapshot(s) produced the vendored PEFT tree, and which
   modifications, if any, can be separately identified without narrowing earlier
   grants? Review the public permissive headers and any other distributions.
2. Which headerless files/portions are original, copied, or adapted? Record their
   source revisions, authors/authorizations and required notices. The matching
   training scripts need particular attention.
3. Do embedded sources require additional texts or notices? Trace the references
   listed in the notice index, including scikit-learn and direct LyCORIS copies.
4. For each dataset/training file and figure, what are the source, revision,
   transformations, attribution and redistribution conditions? Do not infer them
   from a dataset name. Review models and checkpoints separately if later included.
5. Which exact prospective material is eligible for the research/commercial
   approach? Approve the scope, full legal text and activation notice, including
   the unresolved fields in the draft terms. No activation is part of this PR.
