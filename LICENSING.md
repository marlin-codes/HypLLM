# Licensing proposal — draft, not effective

This branch proposes research-use source-available terms and a separate commercial
agreement for material that the individual using the public alias
**marlin-codes** has authority to license. It is for full-text review. Publishing
this branch or pull request, or merging these draft documents unchanged, does
**not** activate a new license or commercial agreement.

There is **no approved covered material** in this proposal yet. No file is brought
under new terms merely by being in this repository. The current licensing evidence
and unresolved boundaries are recorded in the
[provenance and scope inventory](docs/licensing/PROVENANCE.md).

## Proposed use boundary

For material expressly included in a later approved scope:

| Use | Proposed treatment |
| --- | --- |
| Academic research, evaluation, and reproducibility | Free under the proposed research terms. |
| Company-internal research and evaluation, without providing external services or customer operational delivery | Free under the same terms, including evaluation of potential future commercial use. |
| Sharing code and modifications for the permitted research and reproducibility purposes | Included in the proposed free permission. |
| Commercial production deployment, including internal operational deployment | Requires a separately agreed commercial license, to the extent permission is not already available under an existing license. |
| Delivery for customer operational use, or provision of commercial services | Requires a separately agreed commercial license on the same limited basis. |

The activity determines the boundary; a company affiliation, salary, or research
funding alone does not make research commercial production. Internal evaluation
does not include running the software as an operational business system. Sharing
research code is distinct from delivering it for customer operational use.

Read the [proposed research terms](docs/licensing/RESEARCH-LICENSE-DRAFT.md) and
[commercial agreement outline](docs/licensing/COMMERCIAL-AGREEMENT-DRAFT.md).
This is a source-available proposal with use limits, not an open-source license.

## Existing rights and third-party material

Existing valid grants remain available under their own terms, including commercial
permissions where those licenses allow them. Nothing here revokes, narrows, or
adds a commercial payment requirement to those grants. The individual licensor
does not speak for upstream copyright holders.

In particular, **all of `peft/`, including the already published HypLoRA changes,
is excluded from the proposed research-only scope**. Apache-2.0 and MIT notices
were already present when the core implementation was published. The inventory
explains why ownership confirmation alone does not resolve the effect of those
earlier grants. This proposal does not claim that the entire existing HypLoRA core
can become noncommercial.

The [third-party notice index](THIRD_PARTY_NOTICES.md) points to preserved license
texts and attributions. Data, figures, model weights, adapters/checkpoints, and
outputs are separate assets; the proposed code terms do not grant rights to them
or impose a blanket license on them. See the inventory for each category.

## Before any activation

1. Resolve the provenance and previous-distribution questions in the inventory.
2. Identify each approved file or separable contribution, exact revision, rights
   holder, and applicable existing grants. Exclude upstream and previously granted
   rights from any new restriction.
3. Obtain the licensor's approval of the full text and exact scope, settle the open
   legal terms, and publish an explicit adoption notice identifying the approved
   terms, covered revision/material, and effective date. None is supplied here.

A draft PR approval or merge by itself is not that adoption notice. Until explicit
adoption, these proposed terms supply no new permission; existing licenses and
permissions continue to govern. Commercial inquiries may be directed through the
public [marlin-codes profile](https://github.com/marlin-codes); the contracting
individual's legal identity and contact details must be established in any actual
agreement, without treating the public alias as an invented legal name.
