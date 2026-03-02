# ComfyUI-CorridorKey

## 1) Project name + one line summary

`ComfyUI-CorridorKey` is a ComfyUI custom node package that exposes a native `CorridorKey` inference node for four-pass CorridorKey processing inside ComfyUI.

## 2) Problem statement and goals (what it does, what it does NOT do)

`ComfyUI-CorridorKey` adapts the upstream `CorridorKey` inference flow for ComfyUI users who already build coarse alpha hints inside ComfyUI. Instead of running the upstream command-line wizard and instead of bundling extra mask-generation tools such as `VideoMaMa` or `GVM`, the main node accepts:

- an `IMAGE` input
- a coarse `MASK` input produced by other ComfyUI nodes

It then runs an upstream-style 4-channel model pass (RGB plus coarse alpha hint), resizes through the expected `2048x2048` inference resolution, and returns:

- `fg`: straight foreground color
- `matte`: processed linear alpha
- `processed`: linear premultiplied RGB
- `QC`: sRGB checkerboard composite preview

Goals:

- Keep the integration native to ComfyUI's standard custom-node format.
- Reuse existing ComfyUI nodes for coarse mask creation.
- Keep the real CorridorKey inference path available inside a normal ComfyUI graph.
- Preserve the upstream color math rules for sRGB and linear conversions.
- Track upstream `CorridorKey` changes safely and surface newer verified commits without blindly overwriting local custom-node code.

Non-goals:

- This project does not ship the CorridorKey model checkpoint.
- This project does not recreate `VideoMaMa`, `GVM`, or any separate mask-proposal pipeline.
- This project does not download models automatically.
- This project does not require network access for normal inference once dependencies and the checkpoint are installed.
- This project does not auto-overwrite its own code from the upstream `CorridorKey` repository, because that repository is not the same codebase as this ComfyUI custom node.

## 3) Features (current + clearly marked planned)

Current features:

- One ComfyUI node:
  - `CorridorKey`
- Uses vendored upstream-style runtime pieces:
  - `GreenFormer`
  - `CNNRefinerModule`
  - upstream color transfer math
- Accepts batched `IMAGE` tensors and `MASK` tensors
- Requires a matching coarse alpha hint for each frame in a batch instead of silently reusing one mask across multiple frames
- Exposes upstream settings for:
  - `Gamma Space`
  - `Despill Strength`
  - `Auto-Despeckle`
  - `Despeckle Size`
  - `Refiner Strength`
- Returns the four practical passes directly:
  - `fg`
  - `matte`
  - `processed`
  - `QC`
- Includes a bundled example workflow:
  - `example_workflows/corridorkey_example_workflow.json`
  - `example_workflows/PF0044-01_Clip-1_REC709_2K.mp4`
  - `example_workflows/PF0044-01_Clip-1_REC709_2K_mask.mp4`
- On module load, performs a best-effort background check for newer upstream commits that have no failing GitHub check-runs

Planned features:

- Planned: a loader node that can expose a reusable cached engine object explicitly
- Planned: optional raw-alpha and raw-fg outputs in addition to the practical four-pass outputs
- Planned: optional direct support for a dedicated ComfyUI models search path

## 4) Requirements (Python version, OS, dependencies)

- Python: 3.10 or newer
- OS: Windows, Linux, or macOS where ComfyUI runs
- Runtime dependencies:
  - `torch` (normally already present in the ComfyUI Python environment)
  - `torchvision`
  - `timm`
  - `numpy`
  - `opencv-python`
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
3. Download the CorridorKey model checkpoint and place it in `ComfyUI-CorridorKey/models/`.
4. Restart ComfyUI.

Typical install flow:

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/SeanBRVFX/ComfyUI-CorridorKey.git
cd ComfyUI-CorridorKey
python -m pip install -r requirements.txt
```

Before restarting ComfyUI, download the model checkpoint from:

```text
https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth
```

Then place that `.pth` file in:

```text
ComfyUI-CorridorKey/models/
```

Recommended filename:

```text
ComfyUI-CorridorKey/models/CorridorKey.pth
```

The loader will also accept another `.pth` filename in `models/`, so the original `CorridorKey_v1.0.pth` filename works too if it is the only checkpoint in that folder.

After that, restart ComfyUI and the `CorridorKey` node should appear.

If you are using ComfyUI Portable on Windows and want to be explicit about the embedded Python:

```powershell
cd ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/SeanBRVFX/ComfyUI-CorridorKey.git
..\..\python_embeded\python.exe -m pip install -r .\ComfyUI-CorridorKey\requirements.txt
```

Manual install also works:

1. Copy the `ComfyUI-CorridorKey` folder into `ComfyUI/custom_nodes/`.
2. Run `python -m pip install -r requirements.txt` from inside that folder.
3. Add the checkpoint file under `models/`.
4. Restart ComfyUI.

Development install:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
```

For a normal ComfyUI install, `requirements.txt` is the primary runtime dependency file. `pyproject.toml` is kept for editable installs and development tooling.

Repository publishing notes:

- Repository URL: `https://github.com/SeanBRVFX/ComfyUI-CorridorKey`

- This repo is already in the standard shape for both manual `git clone` installs and ComfyUI-Manager `git-clone` installs:
  - the node files live at the repository root
  - `requirements.txt` is in the repository root
  - `README.md` is in the repository root
  - `__init__.py` is in the repository root
- There is no special extra metadata file required inside this repository for basic ComfyUI-Manager compatibility.
- For ComfyUI-Manager discovery, the repository URL typically needs to be added to an external node list maintained by ComfyUI-Manager or another registry source.

## 6) Configuration (env vars, config files, defaults)

The node does not require config files.

Runtime configuration is passed through node inputs:

- `gamma_space`: `sRGB` or `Linear`
- `despill_strength`: normalized float from `0.0` to `1.0`
- `refiner_strength`: float multiplier for the upstream refiner deltas
- `auto_despeckle`: enables the upstream-style connected-component cleanup pass
- `despeckle_size`: minimum island size in pixels for auto-despeckle

Defaults:

- `gamma_space = sRGB`
- `despill_strength = 1.0`
- `refiner_strength = 1.0`
- `auto_despeckle = enabled`
- `despeckle_size = 400`

Implementation notes:

- This node uses the real upstream-style model path and color math, not the earlier heuristic approximation.
- The upstream CLI uses a numeric prompt for `Auto-Despeckle Size` with a default of `400` pixels. This node exposes the same value as an integer input rather than forcing a fixed dropdown.
- The `mask` input is the upstream-style coarse alpha hint: rough, blurry, and usually slightly eroded works better than an expanded mask.
- For batched images, the mask batch must match the image batch. The node does not auto-repeat one mask across multiple frames.

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

Environment variables for performance and CUDA behavior:

- `CORRIDORKEY_PREFER_CHANNELS_LAST`
  - Default: `1`
  - Enables channels-last memory format for better CUDA throughput when supported
- `CORRIDORKEY_ENABLE_TF32`
  - Default: `1`
  - Allows TF32 on supported NVIDIA GPUs for better throughput with small quality tradeoffs typical for inference

## 7) Usage (examples, CLI/API examples if relevant)

Recommended ComfyUI flow:

1. Load or generate an image.
2. Create a coarse alpha hint using existing ComfyUI nodes such as RMBG, SAM, Segment Anything, Florence-based tools, or manual masking nodes.
3. Keep that alpha hint rough rather than over-expanded. Slight erosion, softness, and blur are usually closer to what the upstream model was trained on.
4. Feed the `IMAGE` and coarse `MASK` into `CorridorKey`.
5. For batched images, provide one mask per frame. Do not rely on one static mask for a moving sequence.
6. Send the four pass outputs into downstream preview or save nodes.
7. Send the four pass outputs into your preferred existing saver or export nodes.
8. While the node runs, it emits brief progress text in ComfyUI and prints matching status lines in the ComfyUI console so long operations are visible.

Node inputs:

- `image`: `IMAGE`
- `mask`: `MASK`
- `gamma_space`: `COMBO`
- `despill_strength`: `FLOAT`
- `refiner_strength`: `FLOAT`
- `auto_despeckle`: `COMBO`
- `despeckle_size`: `INT`

Node outputs:

- `fg`: `IMAGE`
- `matte`: `MASK`
- `processed`: `IMAGE`
- `QC`: `IMAGE`

Example workflow idea:

- `Load Image` -> `RMBG` (or another segmentation node) -> `CorridorKey` -> your existing save/export nodes
- Branch the outputs into standard preview nodes and whichever saver nodes you already use for EXR, PNG, or compositing workflows.
- A bundled video example is included at `example_workflows/corridorkey_example_workflow.json`. It uses core `LoadVideo`, `GetVideoComponents`, `ImageToMask`, `CreateVideo`, and `SaveVideo` nodes plus `CorridorKey`.
- That bundled workflow is only a simple video-based test to confirm the node runs end-to-end inside ComfyUI.
- For best quality and behavior closer to the original CorridorKey workflow, prefer EXR image-sequence input and EXR image-sequence output instead of compressed video files.
- For EXR-focused ComfyUI workflows, `ComfyUI-CoCoTools_IO` is a recommended companion node pack:
  - `https://github.com/Conor-Collins/ComfyUI-CoCoTools_IO`
- The bundled example references these included sample files by filename:
  - `PF0028-05_Clip-5_REC709_2K_input.mp4`
  - `PF0028-05_Clip-5_REC709_2K_mask.mp4`
- The bundled sample footage is sourced from ActionVFX practice footage:
  - `https://www.actionvfx.com/practice-footage/mechanic-in-electrical-explosion/15806`
  - Scene: `PF0028`
  - Title: `Mechanic in Electrical Explosion`
  - ActionVFX describes this footage as covered by its license agreement for commercial use, learning, teaching, demo reels, and similar use cases. Review the ActionVFX license terms directly if you plan to redistribute or publish derivative examples.
- The bundled mask example is intentionally simple: it was made in DaVinci Resolve as a basic rough mask with a very slight outward expand/erode adjustment plus blur, to keep it in coarse alpha-hint territory rather than final roto.
- An observed end-to-end runtime for the bundled video test was `484` seconds on this reported machine:
  - `128 GB` RAM
  - `RTX 4070 Ti Super`
  - `16 GB` VRAM
  - This is only a real-world reference point for the bundled sample, not a formal benchmark. Actual runtime will vary with driver state, other loaded ComfyUI models, codec behavior, and source resolution or duration.
- If your local `LoadVideo` node does not automatically resolve files from `example_workflows`, use the same filenames and reselect the bundled files once in the node UI.

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
 batch normalization
        |
        v
 resize to 2048x2048
        |
        v
 4-channel model inference
        |
        v
 upstream refiner delta scaling
        |
        v
 linear alpha + straight FG
        |
        v
 auto-despeckle + despill
        |
        v
 output assembly
  |     |       |      |
  v     v       v      v
 FG   Matte  Processed  QC
```

### Key modules and responsibilities

- `nodes.py`
  - ComfyUI-facing `CorridorKey` node
- `corridor_key/config.py`
  - Dataclass for validated runtime settings
- `corridor_key/color_utils.py`
  - Vendored upstream sRGB/linear conversions, despill, composite, and matte cleanup helpers
- `corridor_key/model_transformer.py`
  - Vendored upstream `GreenFormer` and `CNNRefinerModule`
- `corridor_key/engine.py`
  - Vendored and adapted upstream inference engine with 2048 resize handling
- `corridor_key/tensor_ops.py`
  - Tensor validation and tensor/NumPy batch conversion helpers
- `corridor_key/processor.py`
  - Batch bridge between ComfyUI tensors and the upstream-style engine
- `corridor_key/upstream_sync.py`
  - Best-effort GitHub API integration for verified-upstream commit checks
- `tests/test_processor.py`
  - Unit tests for configuration, color math, and sync policy helpers

### Data flow / request flow

1. ComfyUI invokes `CorridorKey.run`.
2. The node builds a `CorridorKeySettings` object from numeric inputs.
3. The processor validates the image and mask tensors and aligns the batch.
4. The engine resizes each frame to the fixed `2048x2048` inference size, converts gamma as needed, and runs the 4-channel model.
5. The engine applies upstream-style auto-despeckle, despill, linear premultiplication, and checkerboard compositing.
6. The processor returns:
   - `fg`
   - `matte`
   - `processed`
   - `QC`
7. The four outputs are then available to downstream preview, save, or export nodes already present in your ComfyUI setup.
8. In parallel, an optional background task may query GitHub and log if a newer verified upstream commit is available.

## 9) Repository structure (tree)

```text
ComfyUI-CorridorKey/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- pyproject.toml
|-- __init__.py
|-- nodes.py
|-- example_workflows/
|   `-- corridorkey_example_workflow.json
|   |-- PF0044-01_Clip-1_REC709_2K.mp4
|   `-- PF0044-01_Clip-1_REC709_2K_mask.mp4
|-- models/
|   `-- CorridorKey.pth  (user-supplied, not committed)
|-- corridor_key/
|   |-- __init__.py
|   |-- config.py
|   |-- color_utils.py
|   |-- engine.py
|   |-- model_transformer.py
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

- Test configuration and utility code without requiring the checkpoint where possible
- Avoid relying on a full model load in unit tests unless an explicit integration test is added later
- Cover:
  - parameter validation
  - color transfer function behavior
  - batch shape handling
  - four-pass output shapes
  - despill and auto-despeckle edge cases
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
- Runtime status output is intentionally concise:
  - model-loading errors are raised with explicit setup guidance
  - the node emits short progress text while loading the model and processing frames
  - the same progress is printed to the ComfyUI console for visibility during long runs
  - a module-level logger may still emit debug-level messages in the processor
  - a background upstream checker may emit info-level status lines
  - GitHub API checks use explicit short timeouts and fail closed

## 12) Security notes (inputs, secrets, safe defaults)

- No `eval`, `exec`, or unsafe deserialization
- No subprocess execution
- No model downloads or arbitrary network access during image processing
- No secrets are required
- All user-provided numeric inputs are validated and clamped to safe ranges
- Inference reads only the user-supplied checkpoint under the project directory
- The processor only operates on in-memory tensors returned by ComfyUI
- The optional upstream checker uses only GitHub's API with a short timeout, does not require tokens, and never executes remote code

## 13) Deployment (if applicable)

Deployment is simply installation as a ComfyUI custom node:

1. Place this folder under `ComfyUI/custom_nodes/`.
2. Install the package into the same Python environment used by ComfyUI.
3. Add the checkpoint file under `models/`.
4. Restart ComfyUI.

There is no separate server, container, or packaged service in this repository.

## 14) Troubleshooting and FAQ

Q: Why does this node require a mask input?

A: The input is the upstream-style coarse alpha hint. This package intentionally reuses existing ComfyUI segmentation and masking nodes instead of duplicating them.

Q: Can I use one mask for a whole image batch or video batch?

A: No. For batched inputs, provide a matching mask for each frame. CorridorKey expects the coarse alpha hint to describe the current frame, even if that hint is rough and slightly eroded.

Q: The node says the checkpoint is missing.

A: Place the `CorridorKey.pth` model file in `ComfyUI-CorridorKey/models/` and restart ComfyUI.

Checkpoint download URL:

`https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth`

Q: Why is `Processed` not writing an EXR with alpha by itself?

A: `Processed` is returned as premultiplied RGB. Use your existing ComfyUI saver/export nodes to pack or write it the way you want, including workflows that combine `processed` with `matte` downstream.

Q: Why does the bundled example workflow use video if EXR is better?

A: The bundled workflow is only a simple native-node test. For production work and behavior closer to the original CorridorKey workflow, use EXR image sequences for input and output. If you want EXR-focused ComfyUI IO, `ComfyUI-CoCoTools_IO` is a strong companion option: `https://github.com/Conor-Collins/ComfyUI-CoCoTools_IO`

Q: Why is the default `despeckle_size` so large?

A: The upstream `CorridorKey` CLI defaults `Auto-Despeckle Size` to `400` pixels. On small images, you may need to lower that value or disable auto-despeckle.

Q: Why does strong despill sometimes look purple?

A: Higher despill values remove excess green and rebalance color into red and blue, which can push heavy spill areas toward magenta or purple. If that happens, lower `despill_strength` and do the remaining color cleanup later in comp if needed.

Q: Does this node auto-update itself from GitHub?

A: It automatically checks for newer upstream commits that appear verified, but it does not self-modify local code. The upstream repo is a different project, so updates should still be reviewed and ported intentionally.

Q: Can this work on 16GB consumer GPUs?

A: That is a design goal, but the inference path stays fixed at `2048x2048` because that is what the model is trained for. Practical support on 16GB-class GPUs should come from memory and throughput optimization at the fixed size, not from silently changing the inference resolution.

## 15) Roadmap

- Prioritize runtime performance first: reduce per-frame overhead, minimize transfers, and keep throughput strong on production-sized sequences
- Improve practical support for consumer GPUs, especially 16GB-class cards, while keeping the fixed `2048x2048` inference path intact
- Add a loader node that can expose a reusable cached engine object explicitly
- Add optional raw-alpha and raw-fg outputs in addition to the practical four-pass outputs
- Add optional direct support for a dedicated ComfyUI models search path
- Expand the bundled example workflows and keep their real-world benchmark notes current as the node evolves

## 16) Contributing (branching, commit style, PR checklist)

Branching:

- Use short-lived feature branches from `main`
- Keep one focused change per branch

Commit style:

- Prefer imperative commit messages
- Example: `Port upstream CorridorKey runtime into ComfyUI`

PR checklist:

- Update `README.md` first when behavior or structure changes
- Run `pytest -q`
- Run `ruff check .`
- Run `black --check .`
- Run `mypy corridor_key`
- Confirm the node loads in ComfyUI
- Keep the public node contract backward compatible unless documented
- Performance and cleanup contributions are explicitly welcome. This wrapper was prototyped quickly, and contributors with deeper ComfyUI, PyTorch, or optimization experience can improve it further.

Release checklist:

- Confirm `.gitignore` excludes caches and local editor files before the first commit
- Ensure `requirements.txt` matches runtime imports
- Ensure `pyproject.toml` version is correct for the release
- Ensure `README.md` installation instructions match the actual repo URL and current node name
- Make sure `__pycache__` folders are not staged
- Create a clean commit on `main`
- Tag the release, for example `v0.1.0`
- Push both `main` and the tag to GitHub
- Verify a fresh `git clone` into `ComfyUI/custom_nodes` works with `python -m pip install -r requirements.txt`

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
