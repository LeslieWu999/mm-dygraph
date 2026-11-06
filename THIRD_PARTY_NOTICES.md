# Third-party notices

The files below come from other projects and remain under their original licenses, reproduced in
[third_party_licenses/](third_party_licenses/).

| Files | Source | License |
|---|---|---|
| `models/FreeDyG.py` | [Tianxzzz/FreeDyG](https://github.com/Tianxzzz/FreeDyG), unmodified | [MIT](third_party_licenses/FreeDyG.txt) |
| `models/TPNet.py` | [lxd99/TPNet](https://github.com/lxd99/TPNet), unmodified | [MIT](third_party_licenses/TPNet.txt) |
| `models/Liquid.py`, `models/ccffcc.py`, `models/moe.py` | [wave5418/SALoM](https://github.com/wave5418/SALoM), unmodified | [MIT](third_party_licenses/SALoM.txt) |
| `models/CAWN.py`, `models/DyGFormer.py`, `models/GraphMixer.py`, `models/MemoryModel.py`, `models/modules.py`, `models/TCL.py`, `models/TGAT.py`, `utils/utils.py` | [yule-BUAA/DyGLib](https://github.com/yule-BUAA/DyGLib), with modifications | [MIT](third_party_licenses/DyGLib.txt) |

`models/moe.py` states its own origin in its header: it is by David Rau, based on the TensorFlow
implementation in tensor2tensor.

Changes to the unmodified files' behaviour are made from the outside, in `models/adapters.py`:

- **FreeDyG**: the aggregation weights of the final pooling are normalised with a softmax. Without
  it the model cannot be trained on high-dimensional features.
- **SALoM**: fixes a return-value mismatch between `MoE.forward` and `MemoryUpdater`, and only
  computes the memories that a batch reads, so that it scales to the graphs used here.

The training pipeline and the task definitions follow
[DTGB](https://github.com/zjs123/DTGB) and [DyGLib](https://github.com/yule-BUAA/DyGLib).

`UniGraph2Fuser` in `models/fusion_baselines.py` is a reimplementation of the modality alignment of
UniGraph2 (He et al., WWW 2025).
