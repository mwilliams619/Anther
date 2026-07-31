Check which pipeline artifacts exist in models/ and summarize what has been computed vs. what still needs to be run.

Run this bash command and report what's present and what's missing:

```bash
ls -lh models/ 2>/dev/null || echo "models/ directory not found"
```

Then tell the user:
- Which phase(s) are ready (pipeline + index files present)
- Which notebooks need to be run to get to a working state
- The approximate size of each artifact on disk

Notebook dependencies:
- Phase 1 ready = pipeline_phase1.pkl + index_phase1.npy + index_phase1.json all exist
- Phase 2 ready = pipeline_phase2.pkl + index_phase2.npy + index_phase2.json all exist
- fma_small_features.pkl must exist before notebook 02 can run
