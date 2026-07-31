Adjust clustering parameters and refit the Phase 1 pipeline without re-running UMAP from scratch.

The user wants to tune clustering quality. The key levers in anther_ml/cluster.py's fit_clusters():

| Parameter | Default | Effect |
|---|---|---|
| `min_cluster_size` | 100 | Larger = fewer, bigger clusters. ~1% of N is a good starting point. For 8k tracks: 80-150. |
| `cluster_selection_method` | "eom" | `"eom"` = hierarchical (can be lopsided). `"leaf"` = more equal-sized clusters. |
| `n_neighbors` | 30 | Controls UMAP global vs local structure. Higher = more global. |
| `n_pca_components` | 100 | Pre-PCA. None to disable. Check variance retained in output. |

Ask the user what they want to change and why (too many noise points? too few clusters? lopsided sizes?), then:

1. Read anther_ml/cluster.py to confirm current defaults
2. Suggest specific values based on the symptom
3. Update the fit_clusters() call in notebook 02 cell-5 (NotebookEdit on cell-5)
4. Remind the user they need to re-run cells 5 → 13 in notebook 02 to regenerate the pipeline and index

Do NOT change the default parameter values in cluster.py itself — keep the notebook cell as the configuration point.
