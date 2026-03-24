# Motion-Conditioned AnyTop

## What This Is
Extension of AnyTop (arbitrary-skeleton motion diffusion) to support **motion-conditioned generation**: given a source motion on skeleton S, encode it into a skeleton-agnostic latent z, then decode with a target skeleton T to produce retargeted motion. Self-supervised training: encode X on S → z → decode X on S (same skeleton). Inference: encode X on S → z → decode on T (different skeleton).

## Architecture Overview
- **Base framework**: AnyTop (transformer-based DDPM for arbitrary skeletons)
- **New component — Source Motion Encoder**: takes motion X ∈ R^{N×J×D} + RestPE (rest pose encoding, NO topology/TreePE), uses learned attention pooling (K queries) over joints per frame, temporal CNN, then FSQ bottleneck → z ∈ {1,...,L}^{N'×K×K_fsq}
- **New component — Cross-attention in decoder**: each STT block gets a cross-attention layer between skeletal attention and temporal attention, where decoder tokens attend to z
- **Decoder**: AnyTop architecture (Enrichment Block + STT blocks with R,D topological conditioning), plus the new cross-attention pathway

## Key Design Decisions
- Encoder sees RestPE but NOT TreePE (prevents skeleton fingerprinting under FSQ)
- Spatial pooling via Perceiver-style learned queries (K=4), not mean pooling
- z has shape N'×K×K_fsq (preserves functional spatial decomposition from queries)
- Temporal resolution of z is coarser than decoder (N' = N/4)
- FSQ starting point: L=5 levels, K_fsq=4 dimensions per spatial slot
- Classifier-free guidance on z: randomly drop z with probability p during training
- Train from scratch (not from pretrained AnyTop)

## Codebase Structure
- Based on cloned AnyTop repo: https://github.com/Anytop2025/Anytop
- AnyTop's core files to understand before modifying:
  - Model architecture (transformer, STT blocks, enrichment block, topological conditioning)
  - Motion representation: X ∈ R^{N×J×D}, D=13 per joint (position, 6D rotation, velocity, foot contact)
  - Skeleton representation: S = {P_S, R_S, D_S, N_S}
  - Training loop and diffusion logic
- New files we create go in a clear subdirectory or alongside existing model code

## Motion Representation (from AnyTop)
Per joint: root-relative position (3) + 6D rotation (6) + velocity (3) + foot contact (1) = D=13
Motion tensor: R^{N×J×D}, padded to max N and max J across dataset
Skeleton: rest-pose P_S, joint relations R_S, graph distances D_S, joint names N_S

## Tech Stack
- Python, PyTorch
- Truebones Zoo dataset (70 skeletons, ~1219 motions)
- Single GPU training (RTX A6000 or similar)

## Commands
- Read existing AnyTop code before writing new code
- When adding new modules, follow AnyTop's coding style and conventions
- Run `find . -name "*.py" | head -30` to explore the repo structure before making changes

## Important Rules
- NEVER modify AnyTop's existing modules in-place without explicit approval — create new files or subclass
- The encoder must NOT receive TreePE or topology (R_S, D_S) — only RestPE from rest pose
- Each spatial query in the attention pooling processes joints independently per frame
- Temporal processing in the encoder operates on each spatial slot independently (no cross-slot mixing in encoder)
- Cross-attention in decoder: z has no joint dimension from the decoder's perspective — decoder joints route to z's spatial slots via learned attention