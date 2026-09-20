# Reproducibility and artifact audit

## Independence from the original release hosts

StructCT model weights, data, model cards, dataset cards, and inference code are
versioned in this repository. The loader resolves all of them through paths below
the repository root. It contains no remote identifiers for the original model or
dataset repositories.

The MedEmbed text encoder is an explicit third-party dependency. It is downloaded
by `sentence-transformers` on first use and can then be reused from its local
cache. This exception does not provide access to, or fetch from, any StructCT
release repository.

## Checkpoint sanitization

The distributed `.pt` files contain only tensors from each model's `state_dict`.
The following training-only material was deliberately excluded:

- optimizer and learning-rate scheduler states;
- callback, epoch-loop, and logger state;
- training run names and experiment-tracking configuration;
- source-machine paths and cache locations;
- serialized Python configuration objects.

The adjacent JSON sidecar records the minimum architecture and preprocessing
configuration required for inference. The loader uses `torch.load(...,
weights_only=True)` and checks every key against the instantiated architecture.

## Integrity checks

- Full SHA-256 digests are stored beside every checkpoint and in
  `artifact-manifest.json`.
- Converted tensors were loaded again with `weights_only=True` and compared
  element-for-element with the source state dictionary.
- Tests instantiate representative 2-class and 3-class models and validate all
  dataset row counts.
- A repository-wide privacy scan checks text, Git configuration, and tracked file
  names before publication.
