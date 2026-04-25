Generate a self-study MD and HTML guide for a Jupyter notebook, then optionally add it to the webapp.

## Arguments

Parse from `$ARGUMENTS`:
- **notebook_path** (required) — path to the `.ipynb` file
- **`--add-to-webapp`** (optional flag) — copy HTML to `webapp/` and add a card to `webapp/index.html`
- **`--submission vN $XX,XXX`** (optional) — e.g. `--submission v8 $21,805` — marks this as a Kaggle submission with version and score

## Step 1 — Read and parse the notebook

Read the `.ipynb` file. It is JSON with a `cells` array. Each cell has:
- `cell_type`: `"markdown"` or `"code"`
- `source`: array of strings (the cell content)
- `outputs`: array of output objects (for code cells that have been run)

Extract the following automatically:

**From markdown cells:**
- `notebook_title`: text of the first `# Heading` (strip the `#`)
- `notebook_topic`: the subtitle or first paragraph after the title
- Major sections: each `## Heading` becomes an h2 section in the study guide

**From code cell outputs** (look in `outputs[*].text` or `outputs[*].data["text/plain"]`):
- `cv_rmse`: find lines matching patterns like `RMSE`, `rmse`, `val RMSE`, `OOF RMSE`, `CV RMSE` followed by a dollar amount or number — extract the numeric value
- `best_params`: find lines containing `best_params_` or `Best params` — capture the dict/values
- Any printed score tables or comparison tables

**From `--submission` argument (if provided):**
- `kaggle_version`: e.g. `v8`
- `kaggle_rmse`: e.g. `$21,805`

## Step 2 — Derive output filenames

From `notebook_path`:
- Get the filename without extension: e.g. `07_extra_trees_stack` from `notebooks/experiments/07_extra_trees_stack.ipynb`
- `md_path` = `outputs/<basename>_explained.md`
- `html_path` = `outputs/<basename>_explained.html`
- `html_filename` = `<basename>_explained.html`
- `notebook_basename` = `<basename>.ipynb`

## Step 3 — Write `outputs/<basename>_explained.md`

Structure:
```
# <notebook_title>

> <1-sentence summary of what this notebook does and what it found>

## Overview
[2-3 paragraphs: what problem this notebook tackles, what approach was used, key result]

## [Section for each major ## heading found in the notebook]
[Plain-English explanation of the concept + relevant code snippet + what it means]

## Results
[Table of key metrics extracted from cell outputs]

## Key Takeaways
[3-5 bullet points: what we learned, what worked, what didn't]
```

## Step 4 — Write `outputs/<basename>_explained.html`

Use this exact CSS template (copy it verbatim into the `<style>` block):

```css
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 24px 80px; color: #1a1a1a; line-height: 1.7; background: #fafafa; }
h1 { font-size: 2rem; color: #1a1a1a; border-bottom: 3px solid #0369a1; padding-bottom: 12px; }
h2 { font-size: 1.4rem; color: #0369a1; margin-top: 48px; border-left: 4px solid #0369a1; padding-left: 12px; }
h3 { font-size: 1.1rem; color: #374151; margin-top: 28px; }
h4 { font-size: 1rem; color: #6b7280; margin-top: 20px; }
code { background: #1e293b; color: #e2e8f0; padding: 2px 7px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 0.88em; }
pre { background: #1e293b; color: #e2e8f0; padding: 20px 24px; border-radius: 8px; overflow-x: auto; font-size: 0.88em; line-height: 1.6; margin: 16px 0; }
pre code { background: none; padding: 0; color: inherit; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.93em; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
th { background: #0369a1; color: white; padding: 10px 14px; text-align: left; font-weight: 600; }
td { padding: 9px 14px; border-bottom: 1px solid #e5e7eb; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f8fafc; }
blockquote { background: #eff6ff; border-left: 4px solid #0369a1; margin: 20px 0; padding: 14px 20px; border-radius: 0 8px 8px 0; color: #1e3a5f; }
blockquote p { margin: 0; }
.highlight { background: #ecfdf5; border-left: 4px solid #059669; margin: 20px 0; padding: 14px 20px; border-radius: 0 8px 8px 0; color: #065f46; }
.highlight p { margin: 0; }
.warning { background: #fffbeb; border-left: 4px solid #f59e0b; margin: 20px 0; padding: 14px 20px; border-radius: 0 8px 8px 0; color: #78350f; }
.warning p { margin: 0; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 40px 0; }
p { margin: 10px 0; }
ul, ol { padding-left: 24px; }
li { margin: 6px 0; }
.badge { display: inline-block; background: #0369a1; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.82em; font-weight: 600; margin-right: 6px; vertical-align: middle; font-family: 'Courier New', monospace; }
.badge-green { background: #059669; }
.badge-orange { background: #d97706; }
.score-box { background: white; border: 2px solid #0369a1; border-radius: 12px; padding: 20px 28px; margin: 24px 0; display: flex; gap: 40px; flex-wrap: wrap; }
.score-item { text-align: center; }
.score-label { font-size: 0.82em; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.score-value { font-size: 1.6rem; font-weight: 700; color: #0369a1; }
.score-delta { font-size: 0.88em; color: #059669; font-weight: 600; }
```

HTML structure:
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>[notebook_title]</title>
<style>[CSS above]</style>
<script src="nav.js"></script>
</head>
<body>

<h1>[notebook_title]<br><small style="font-size:0.55em;color:#6b7280;">[1-line subtitle]</small></h1>

<!-- Score box: always include if any RMSE found -->
<div class="score-box">
  <!-- If --submission provided: -->
  <div class="score-item">
    <div class="score-label">[kaggle_version] Kaggle RMSE</div>
    <div class="score-value" style="color:#d97706">[kaggle_rmse]</div>
  </div>
  <!-- Always if cv_rmse found: -->
  <div class="score-item">
    <div class="score-label">CV / OOF RMSE</div>
    <div class="score-value" style="color:#059669">[cv_rmse]</div>
  </div>
  <!-- If finding only (no --submission): label it "Experiment Result" -->
</div>

<!-- Highlight box: 1-sentence key insight -->
<div class="highlight"><p><strong>Key insight:</strong> [one sentence]</p></div>

<!-- Sections: one <h2> per major topic from the notebook -->
...

<hr>
<p style="color:#9ca3af;font-size:0.85em;text-align:center;">
  [badge: submission version OR "Experiment"]
  <span class="badge">[notebook_basename]</span>
  Team 3 · NTU SCTP Module 3 · [today's date]
</p>
</body>
</html>
```

Write both files now.

## Step 5 — If `--add-to-webapp` flag is present

**5a.** Copy the HTML file to `webapp/<html_filename>` (use the Bash tool: `cp outputs/<basename>_explained.html webapp/<basename>_explained.html`)

**5b.** Add a new card to `webapp/index.html` inside the `.card-grid` div, before the closing `</div>` of the grid.

- If `--submission` provided → use gold badge: `<span class="badge b-gold">[kaggle_version] · [kaggle_rmse]</span>`
- If finding only → use gray badge: `<span class="badge b-gray">Experiment</span>`
- Always add: `<span class="badge b-blue">[notebook_basename]</span>`
- Icon: use 🚀 for submissions, 🔬 for findings

Card template:
```html
    <a class="card" href="#" data-doc="[html_filename]">
      <div class="card-icon">[icon]</div>
      <h3>[notebook_title]</h3>
      <p>[2-sentence plain description of what the notebook does and what it found]</p>
      <div class="card-meta">[badges]</div>
    </a>
```

Insert the new card at the END of the `.card-grid` div (before its closing `</div>`).

**5c.** Run `python3 webapp/build_index.py` to rebuild the search index.

## Step 6 — Report to user

Print a summary:
```
✅ Study guide created:
   MD:   outputs/<basename>_explained.md
   HTML: outputs/<basename>_explained.html

[If --add-to-webapp:]
✅ Added to webapp:
   File: webapp/<basename>_explained.html
   Card added to webapp/index.html
   Search index rebuilt (webapp/search_index.json)

[If --submission:]
   Version: [kaggle_version] · Kaggle RMSE: [kaggle_rmse]

Auto-extracted from notebook:
   CV/OOF RMSE: [cv_rmse or "not found"]
   Best params:  [summary or "not found"]
```
