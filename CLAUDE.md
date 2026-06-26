# Soccer xG — working notes for Claude Code

## README / Hugging Face deploy (important)

- **`README.md` on `main` must have NO YAML frontmatter.** GitHub renders
  frontmatter as an ugly table, so it was deliberately removed. Do not re-add a
  `---` config block to the top of `README.md`.
- The Hugging Face Spaces config (sdk, app_port, title, emoji, colors) lives in
  **`deploy/hf-header.md`**. Edit that file if Space config needs to change.
- Deploy to the Space with **`./scripts/deploy-hf.sh`** (never plain
  `git push space main`). The script prepends `deploy/hf-header.md` to the
  README, force-pushes to the `space` remote, and restores the clean local
  README. It requires a clean working tree.
- Remotes: `origin` = GitHub, `space` = Hugging Face Space.

## Commits

- **Never add Claude / Co-Authored-By attribution** to commits or PRs.
- Use concise one-line commit messages; split unrelated work into separate commits.

## Project shape

- The trained model (`models/baseline.joblib`) is committed, so the app runs
  with no data download or training.
- Run locally: `uvicorn xg.serve.app:app --reload` (serves model + UI).
- Retrain: `python -m xg.data.load` then `python -m xg.models.baseline`.
- `src/xg/features/build.py` is the single train-and-serve feature source
  (9 geometric features); keep train and serve paths identical.
