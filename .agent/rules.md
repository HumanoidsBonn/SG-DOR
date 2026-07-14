# Workspace Rules


## 1. Research Transparency

- **Documentation First**: This is a machine learning research project. Working code is not enough; intent and logic must be documented.
- **Config Transparency**: All configurable parameters MUST be exposed in the config system. Do not hardcode magic numbers.
- **Reproducibility**: Ensure `config.json` and `experiments` artifacts capture the exact state of a run.
- **No Graceful Fallbacks**: As this is for scientific research, do not implement "graceful fallbacks" for missing dependencies or failed computations. If a dependency is required for a specific research metric (e.g., Shapely for Union Occlusion), let the script fail loudly so issues are not hidden.

## 2. Testing & Stability

- **Regression Testing**: When fixing a bug, verify that related functionality has not broken.
- **Feature Isolation**: New features must not break existing workflows. Test with both new and legacy configurations (e.g., check both GraphSAGE and GINE paths).

## 3. Commit Logging & Memory

- **RECENT_COMMITS.md**: After every significant fix or feature implementation, append a summary of changes to `RECENT_COMMITS.md` (root).
- **SESSION_SUMMARY.md**: Maintain a detailed, date-wise log of all activities in `docs/SESSION_SUMMARY.md`. This serves as the agent's primary memory.
- **Startup Protocol**: Every time this workspace is opened, the agent MUST first read both `docs/SESSION_SUMMARY.md` and `RECENT_COMMITS.md` in the root directory to establish context.
- **Format**: `[Date] [Type] Description of change (Files touched)`

## 4. File Hygiene & Terminal

- **Docs Directory**: All documentation and markdown files (except `README.md` and `RECENT_COMMITS.md`) should be placed in `docs/`.
- **Terminal Output**: Do NOT use `grep` to pipe command outputs directly if you need to inspect the result. Instead, pipe the output to a temporary log file (e.g., `command > out.log 2>&1`) and then read that file. This ensures reliable capture of all streams.
- **No Root Pollution**: Do not dump logs, temporary files, or experimental scripts in the root directory. Keep the workspace clean.

## 5. Code Quality

- **Type Hints**: All function signatures must have Python type hints.
- **Docstrings**: All modules, classes, and functions must have descriptive docstrings explaining their purpose, arguments, and return values.

## 6. Versioning

- **Changelog**: meaningful changes must be logged in `changelog.rst` under a new version heading or `Unreleased`.
- **Tags**: Use semantic versioning git tags (e.g., `v0.1.0`) for stable checkpoints.

## 7. Commenting


- **Long Functions**: Any function longer than ~20 lines or containing complex logic chunks MUST have inline comments explaining the steps. Short, self-explanatory functions can remain comment-free.
