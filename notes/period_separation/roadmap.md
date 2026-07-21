Overwrite report.md with your findings from the following objectives. Make sure to have very consistent naming conventions for the run: addsub-L18-08-<what is being tested><value>. In the report, be explicit about which run is which, and what are the most important hyperparameters for each run.

# Objective 1 - Find the optimal impmin coeff for the initial training
First, let’s determine the optimal peak value for the importance-minimality coefficient.
Run 4000-step decomposition runs for layer 18, using the same hyperparameters as the canonical addsub-L18-04-hidden run, except:
- Do not enable the HiddenActsReconLoss yet
- Use the SmoothL0 importance minimality
- Use constant hyperparameters: the gamma for SmoothL0 should always be 1, and the importance-minimality should be constant within a run (the value depends on the experiment -- this is what is being assessed here).

Commit the changes before running the sweep.

After the sweep is done, make a plot of the relevant metrics as a function of the impmin coeff. The most important ones are:
- PGDReconLoss
- ImportanceMinimality loss (not counting the coefficients, so different runs can be compared)
- Metrics that measure how many components are periodic, at each period, and how many components are periodic at multiple periods.

All taken after 4000 steps of training. By 4000 steps, the best runs would have the following properties:
- They have periodic subcomponents at each period (2, 5, 10, 20, 50) in each matrix (optimal: one per matrix, but multiple are fine)
- For each matrix/period, there are as few subcomponents as possible
- Each subcomponent responds to exactly one period -- the periods are not mixed
Please make sure that the metrics you log and plot allow to assess how well each run does according to these criteria.

When you are done with this, pick the run that gave the most promising results, and move on to the next objective, using the optimal impmin coeff value you just determined. The importance-minimality does not matter at that point, however I expect the PGD recon loss at 4000 steps to be below 0.015. Runs with a recon error above probably have an excessive impmin coeff.

# Objective 2 - Test initialization
There are three initialization schemes: kaiming, coupled, and within_span. within_span is "coupled with the coupling broken": each side is the W-image of its own independent Gaussian (V_c ∝ W^T g_c, U_c ∝ W h_c), so both sides lie within the matrix's row/column spaces with S²-weighted directions — the same per-side marginals as coupled — but the two sides are statistically independent. Do a 4000-step run with each initialization method, using the optimal parameters from above, to find whether initialization affects period mixing. When that’s done, pick the best initialization for the rest of the series.

# Objective 3 - Implement a more efficient hidden acts recon loss, and test it.
Following the way it is implemented in the `experiment/8B_targeted_jax`, re-implement the hidden acts recon loss so that it does not require its own forward passes: instead, it should be an extra attribute to the StochasticRecon loss. It should use the same stochastically-ablated forward pass, and compute the deviation in hidden activations from them.

When that’s done, commit and push the changes.

Now, let’s see if we can get better-separated components by adding the new auxilliary hidden-activation loss. Here again, since we are only looking at the first "stage" of training (first 4000 steps), there’s no need to try annealing. Just try different values (0, 0.001, 0.01, 0.1). Add a plot to the report showing the effect of the hidden-activation loss.

# Objective 4 - Test the "beta" parameter of importance minimality (the frequency-minimality loss)
Similarly to above, using the best hyperparameters found so far, test several values for the beta coefficient: 0, 0.5, 0.75.
