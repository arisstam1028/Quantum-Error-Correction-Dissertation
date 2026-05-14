# Plotting

This section contains Matplotlib plotting helpers for simulation output.

## Files

### `plotter.py`

Functions:

- `plot_fer(results, title="FER vs p")`: plots frame error rate against depolarizing probability for one result list.
- `plot_iterations(results, title="Average iterations vs p")`: plots average decoder iterations against depolarizing probability for one result list.
- `plot_fer_compare(all_results, title="FER Comparison")`: plots FER curves for multiple decoders or matrix-family members.
- `plot_iterations_compare(all_results, title="Average Iterations Comparison")`: plots iteration curves for multiple decoders or matrix-family members.

## Expected Input

Each result row is a dictionary with:

```text
p
frames
failures
fer
avg_iterations
```

