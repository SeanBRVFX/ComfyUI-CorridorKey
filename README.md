# ComfyUI-CorridorKey

## 1) Project name + one line summary

`ComfyUI-CorridorKey` is a ComfyUI custom node package that exposes a native `CorridorKey` node for CorridorKey-style coarse-mask refinement inside ComfyUI.

## 2) Problem statement and goals (what it does, what it does NOT do)

`ComfyUI-CorridorKey` adapts the intent of the upstream `CorridorKey` project for ComfyUI users who already build masks inside ComfyUI. Instead of running a separate command-line application and instead of bundling extra mask-generation tools such as `VideoMaMa` or `GVM`, this node accepts:

- an `IMAGE` input
- a coarse `MASK` input produced by other ComfyUI nodes

It then performs an edge-aware matte refinement pass and returns:

- a refined alpha `MASK`
- a selected preview `IMAGE` based on the requested output mode
- dedicated `IMAGE` outputs for foreground, processed, and comp previews

Goals:

- Keep the integration native to ComfyUI's standard custom-node format.
- Reuse existing ComfyUI nodes for coarse mask creation.
- Keep runtime dependencies small and compatible with a typical ComfyUI Python environment.
- Provide a deterministic, testable baseline refinement pipeline that is safe to run locally.
- Track upstream `CorridorKey` changes safely and surface newer verified commits without blindly overwriting local custom-node code.

Non-goals:

- This project does not reimplement the full upstream research model or ship upstream CorridorKey checkpoints.
- This project does not recreate `VideoMaMa`, `GVM`, or any separate mask-proposal pipeline.
- This project does not download models, call external services, or require network access at runtime.
- This project does not write output files to disk; it only returns ComfyUI node outputs.
- This project does not auto-overwrite its own code from the upstream `CorridorKey` repository, because that repository is not the same codebase as this ComfyUI custom node.

## 3) Features (current + clearly marked planned)

Current features:

- One ComfyUI node: `CorridorKey`
- Accepts batched `IMAGE` tensors and `MASK` tensors
- Broadcasts a single mask across an image batch when needed
- Validates shapes, value ranges, and numeric settings
- Exposes upstream-inspired controls for:
  - `Gamma Space`
  - `Despill Strength`
  - `Auto-Despeckle`
  - `Despeckle Size`
  - `Refiner Strength`
  - output mode selection (`FG`, `Matte`, `Processed`, `Comp`)
- Performs deterministic edge-aware mask smoothing, spill suppression, and comp preview generation
- Returns refined alpha mask plus selected output, foreground, processed, and comp preview images
- On module load, performs a best-effort background check for newer upstream commits that have no failing GitHub check-runs
- Includes pytest coverage for the core processor

Planned features:

- Planned: optional backend that can call a true upstream CorridorKey-compatible runtime if the user provides one
- Planned: trimap output for downstream compositing workflows
- Planned: preset profiles tuned for hair, glass, and semi-transparent edges
- Planned: optional preview compositing against solid-color backgrounds

## 4) Requirements (Python version, OS, dependencies)

- Python: 3.10 or newer
- OS: Windows, Linux, or macOS where ComfyUI runs
- Runtime dependencies:
  - `torch` (normally already present in the ComfyUI Python environment)
  - `numpy`
  - `Pillow`
- Development dependencies:
  - `pytest`
  - `ruff`
  - `black`
  - `mypy`

## 5) Quick start (venv, install, run)

This repository is meant to live under ComfyUI's `custom_nodes` directory.

Recommended ComfyUI installation:

1. Clone this repository into `ComfyUI/custom_nodes/`.
2. Install the runtime dependencies into the same Python environment ComfyUI uses.
3. Restart ComfyUI.

Typical install flow:

```powershell
cd ComfyUI\custom_nodes
git clone <repo-url> ComfyUI-CorridorKey
cd ComfyUI-CorridorKey
python -m pip install -r requirements.txt
```

After that, restart ComfyUI and the `CorridorKey` node should appear.

If you are using ComfyUI Portable on Windows and want to be explicit about the embedded Python:

```powershell
cd ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone <repo-url> ComfyUI-CorridorKey
..\..\python_embeded\python.exe -m pip install -r .\ComfyUI-CorridorKey\requirements.txt
```

Manual install also works:

1. Copy the `ComfyUI-CorridorKey` folder into `ComfyUI/custom_nodes/`.
2. Run `python -m pip install -r requirements.txt` from inside that folder.
3. Restart ComfyUI.

Development install:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
```

Restart ComfyUI after installation. The node will appear under the `CorridorKey` category.

For a normal ComfyUI install, `requirements.txt` is the primary runtime dependency file. `pyproject.toml` is kept for editable installs and development tooling.

## 6) Configuration (env vars, config files, defaults)

The node does not require config files.

Runtime configuration is passed through node inputs:

- `gamma_space`: `sRGB` or `Linear`, matching the upstream input interpretation concept
- `feather_radius`: integer blur radius used to soften the coarse mask before refinement
- `edge_focus`: float multiplier that strengthens image-edge preservation
- `threshold`: float cutoff used to center the matte transition
- `preserve_core`: float band that locks obvious foreground/background values
- `despill_strength`: normalized float from `0.0` to `1.0` that reduces green spill on semi-transparent foreground edges
- `refiner_strength`: float multiplier that strengthens the edge-aware matte correction pass
- `auto_despeckle`: enables a small morphology-based cleanup pass
- `despeckle_size`: integer control for the minimum speckle scale to suppress
- `output_mode`: selects the primary preview output as `FG`, `Matte`, `Processed`, or `Comp`

Defaults:

- `gamma_space = sRGB`
- `feather_radius = 4`
- `edge_focus = 1.5`
- `threshold = 0.5`
- `preserve_core = 0.2`
- `despill_strength = 1.0`
- `refiner_strength = 1.0`
- `auto_despeckle = enabled`
- `despeckle_size = 400`
- `output_mode = Processed`

Implementation note:

- The upstream app uses model-driven processing. This ComfyUI node mirrors the same user-facing control surface where practical, but the internal math is a deterministic local approximation built around the provided image and coarse mask.
- The upstream CLI uses a numeric prompt for `Auto-Despeckle Size` with a default of `400` pixels. This node exposes the same value as an integer input rather than forcing a fixed dropdown.

Environment variables for upstream tracking:

- `CORRIDORKEY_AUTO_CHECK_UPSTREAM`
  - Default: `1`
  - Set to `0` to disable the background upstream check
- `CORRIDORKEY_UPSTREAM_TIMEOUT_SECONDS`
  - Default: `3.0`
  - Per-request timeout for GitHub API calls
- `CORRIDORKEY_UPSTREAM_CHECK_DEPTH`
  - Default: `15`
  - How many recent upstream commits to scan when looking for the newest verified commit

The currently reviewed upstream head SHA and its last observed check state are pinned in `corridor_key/upstream_sync.py`.

## 7) Usage (examples, CLI/API examples if relevant)

Recommended ComfyUI flow:

1. Load or generate an image.
2. Create a coarse mask using existing ComfyUI nodes such as RMBG, SAM, Segment Anything, Florence-based tools, or manual masking nodes.
3. Feed the `IMAGE` and coarse `MASK` into `CorridorKey`.
4. Choose `FG`, `Matte`, `Processed`, or `Comp` as the node's selected preview output.
5. Use the refined `MASK` for compositing, inpainting, EXR assembly, or export.

Node inputs:

- `image`: `IMAGE`
- `mask`: `MASK`
- `gamma_space`: `COMBO`
- `feather_radius`: `INT`
- `edge_focus`: `FLOAT`
- `threshold`: `FLOAT`
- `preserve_core`: `FLOAT`
- `despill_strength`: `FLOAT`
- `refiner_strength`: `FLOAT`
- `auto_despeckle`: `COMBO`
- `despeckle_size`: `INT`
- `output_mode`: `COMBO`

Node outputs:

- `refined_mask`: `MASK`
- `selected_output`: `IMAGE`
- `foreground`: `IMAGE`
- `processed`: `IMAGE`
- `comp`: `IMAGE`

Example workflow idea:

- `Load Image` -> `RMBG` (or another segmentation node) -> `CorridorKey` -> `Preview Image`
- For EXR-oriented workflows, use the returned `foreground` and `refined_mask` separately and combine or export them with other ComfyUI nodes such as `cocotools_io`, instead of expecting this node to write EXR files directly.

There is no standalone CLI in this custom node package by design.

Upstream tracking behavior:

- On import, the package starts a small background check.
- It queries the upstream `nikopueringer/CorridorKey` GitHub repo.
- It looks for the newest recent commit with at least one successful check-run and no failing check-runs.
- If such a commit is newer than the currently pinned verified upstream baseline, it logs that an upstream update is available for manual review.
- It does not automatically rewrite local files from upstream.

## 8) Architecture overview

### High-level diagram in ASCII

```text
IMAGE + coarse MASK
        |
        v
  input validation
        |
        v
 mask normalization
        |
        v
 optional gamma handling
        |
        v
 image luminance + edge map
        |
        v
 refiner + despeckle
        |
        v
   refined alpha MASK
        |
        v
 despill + output assembly
   |       |        |
   v       v        v
  FG   Processed   Comp
```

### Key modules and responsibilities

- `nodes.py`
  - ComfyUI-facing node class
  - Defines `INPUT_TYPES`, return types, and node metadata
  - Converts node inputs into the internal processor call
- `corridor_key/config.py`
  - Dataclass for validated runtime settings
- `corridor_key/tensor_ops.py`
  - Shared tensor validation, mask normalization, color helpers, and small image-processing helpers
- `corridor_key/processor.py`
  - Pure processing pipeline for edge-aware matte refinement, despill, despeckle, and output assembly
- `corridor_key/upstream_sync.py`
  - Best-effort GitHub API integration for verified-upstream commit checks
- `tests/test_processor.py`
  - Unit tests for shape handling, validation, and output behavior

### Data flow / request flow

1. ComfyUI invokes `CorridorKey.run`.
2. The node builds a `CorridorKeySettings` object from numeric inputs.
3. The processor validates the image and mask tensors.
4. A normalized mask batch is created and aligned to the image batch.
5. The processor optionally adapts handling for `Gamma Space`.
6. The processor derives luminance, computes an edge map, and applies the refiner pass.
7. Optional despeckle cleanup is applied when enabled.
8. The processor builds foreground, processed, and comp previews, including optional despill.
9. The processor returns:
   - refined mask
   - selected output
   - foreground
   - processed
   - comp
10. ComfyUI receives the outputs without any file-system side effects.
11. In parallel, an optional background task may query GitHub and log if a newer verified upstream commit is available.

## 9) Repository structure (tree)

```text
ComfyUI-CorridorKey/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- pyproject.toml
|-- __init__.py
|-- nodes.py
|-- corridor_key/
|   |-- __init__.py
|   |-- config.py
|   |-- tensor_ops.py
|   |-- processor.py
|   `-- upstream_sync.py
`-- tests/
    `-- test_processor.py
```

## 10) Development

### local dev loop

```powershell
python -m pip install -e .[dev]
pytest -q
ruff check .
black --check .
mypy corridor_key
```

When changing behavior:

1. Update `README.md` first if the behavior, architecture, setup, or workflow changes.
2. Keep the implementation aligned with the documented repository structure.
3. Add or update pytest coverage for the changed logic.
4. Restart ComfyUI and re-test the node in a simple image-plus-mask workflow.

### testing strategy (pytest)

- Test pure processing code in `corridor_key/processor.py`
- Avoid ComfyUI runtime dependencies in the test suite where possible
- Cover:
  - mask broadcasting
  - parameter validation
  - output shapes
  - value range clamping
  - output mode selection
  - despeckle and despill edge cases
  - verified-upstream selection policy for mocked GitHub metadata

### lint/format rules (ruff + black or ruff format)

- `ruff check .` for linting
- `black .` for formatting
- Keep imports explicit and functions small
- Prefer readable tensor code over compact but opaque expressions

### type checking (mypy or pyright if useful)

- `mypy corridor_key`
- Use standard type hints for public helpers and processor methods
- Keep the ComfyUI adapter lightweight and typed where practical

## 11) Error handling and logging (logging module, structured messages)

- Internal validation raises `ValueError` with specific, user-readable messages
- The ComfyUI node catches nothing silently; invalid inputs should fail clearly
- Logging is intentionally minimal:
  - a module-level logger may emit debug-level messages in the processor
  - a background upstream checker may emit info-level status lines
  - no print-based progress logging
  - GitHub API checks use explicit short timeouts and fail closed

## 12) Security notes (inputs, secrets, safe defaults)

- No `eval`, `exec`, or unsafe deserialization
- No subprocess execution
- No model downloads or arbitrary network access during image processing
- No secrets are required
- All user-provided numeric inputs are validated and clamped to safe ranges
- File paths are not accepted as node inputs, so the node does not read or write arbitrary files
- The processor only operates on in-memory tensors returned by ComfyUI
- The optional upstream checker uses only GitHub's API with a short timeout, does not require tokens, and never executes remote code

## 13) Deployment (if applicable)

Deployment is simply installation as a ComfyUI custom node:

1. Place this folder under `ComfyUI/custom_nodes/`.
2. Install the package into the same Python environment used by ComfyUI.
3. Restart ComfyUI.

There is no separate server, container, or packaged service in this repository.

## 14) Troubleshooting and FAQ

Q: Why does this node require a mask input?

A: This package intentionally reuses existing ComfyUI segmentation and masking nodes instead of duplicating them.

Q: Why does it not match the full upstream CorridorKey model exactly?

A: This implementation provides a native, deterministic refinement pass inspired by the same workflow goal, without bundling upstream research weights or optional external pipelines.

Q: My mask looks too soft.

A: Lower `feather_radius`, lower `preserve_core`, raise `edge_focus`, or raise `refiner_strength`.

Q: My mask looks too harsh.

A: Raise `feather_radius` slightly or lower `threshold`.

Q: Why is `Processed` not writing an EXR with alpha by itself?

A: ComfyUI nodes exchange in-memory tensors. This node returns separate `foreground` and `refined_mask` outputs so you can assemble or export EXR data elsewhere in the workflow.

Q: Why is the default `despeckle_size` so large?

A: The upstream `CorridorKey` CLI defaults `Auto-Despeckle Size` to `400` pixels. On small images, you may need to lower that value or disable auto-despeckle.

Q: Does this node auto-update itself from GitHub?

A: It automatically checks for newer upstream commits that appear verified, but it does not self-modify local code. The upstream repo is a different project, so updates should still be reviewed and ported intentionally.

## 15) Roadmap

- Add an optional pluggable backend for a true upstream-compatible inference engine
- Add trimap and confidence outputs
- Add more regression tests around semi-transparent regions
- Add a true RGBA or EXR-friendly packed output path if the surrounding ComfyUI ecosystem provides a stable tensor contract for it
- Add an example ComfyUI workflow JSON once the node behavior stabilizes

## 16) Contributing (branching, commit style, PR checklist)

Branching:

- Use short-lived feature branches from `main`
- Keep one focused change per branch

Commit style:

- Prefer imperative commit messages
- Example: `Add edge-aware CorridorKey refinement node`

PR checklist:

- Update `README.md` first when behavior or structure changes
- Run `pytest -q`
- Run `ruff check .`
- Run `black --check .`
- Run `mypy corridor_key`
- Confirm the node loads in ComfyUI
- Keep the public node contract backward compatible unless documented

## 17) License placeholder

This package is based on the workflow and licensing expectations stated by the upstream `nikopueringer/CorridorKey` repository.

Upstream license status:

- The upstream repository does not present a standard standalone SPDX license file in the root.
- Its README includes a custom license statement described by the author as effectively a variation of `CC BY-NC-SA 4.0`.
- The upstream README also states additional restrictions, including:
  - no repackaging and resale of the tool
  - no paid API inference service using the model without a separate agreement
  - released variations or improvements should remain free and open source
  - future forks or releases should keep the `Corridor Key` name

For this custom node, treat the upstream repository's licensing terms as the governing reference unless the repository owner explicitly relicenses this work.

This repository also includes a local `LICENSE` file that summarizes the same upstream-derived terms for convenience.

Source reference:

- Upstream repository: `https://github.com/nikopueringer/CorridorKey`
- Upstream licensing section: `README.md`, section `CorridorKey Licensing and Permissions`
