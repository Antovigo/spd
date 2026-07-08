Common setup: Run addsub-L18-04-2x-beta0.75-LR/model_24000.pth; operations add & sub over the 1..100 grids. Every intervention and the KL readout are at the last (=) position only; reference = raw target model.

"Selected" subcomponent set S: two sets of experiments.
A: "alive subcomponents": subcomponents identified as alive by the dedicated script. The list is in the alive_subcomponents.tsv file. Always the same across prompts.
B: "active subcomponents": subcomponents that have a per-prompt CI > 0.01 at the last token position. Different across prompts.

Run the experiments for alive subcomponents first, then for active subcomponents.

For each site we compute the per-op grid mean μ at = and split x = μ + δ; every projection P runs in three flavors:
- raw P(μ+δ)
- centered μ + Pδ (information-routing test)
- bias-only Pμ + δ (offset-routing test).
Raw is the strictest faithfulness claim (the mechanism takes both its signal and its offsets from the same activation components); centered−raw gap attributes failures to offset vs information routing.
As a baseline, we also run the model with only selected subcomponents, and no projection. The delta component is off in all experiments.
Note that P is constant in the case of "alive subcomponents", but variable in the case of "active subcomponents". The order in which data is collected should be optimized accordingly.

Run the following set of experiments separately for the L18 attention block and for the L18 MLP block, for each of the 3 flavors of projection. Here is a breakdown for MLPs:

1. Original weights, with activations projected on the subspace of selected subcomponents
   1A. Project each matrix’s input activations on the subspace of selected subcomponents V vectors; run the original matrices. Gate's input → span(V_S^gate) and up's → span(V_S^up) separately (hooks on the Linears). Post-SwiGLU activations h → span(V_S^down), feeding the original down_proj.
   1B. Run the original matrices; project each matrix’s output activations on the subspace of selected subcomponents U vectors. Gate and up preactivations → span(U_S^gate), span(U_S^up) (before SwiGLU). MLP output → span(U_S^down).
   
2. Selected subcomponents, with activations projected on the subspace of neurons
   The exact converse experiment. This is only meaningful when the matrix has fewer rows than the dimension of the input activations, so we only apply the filtering to these places.
   2A. Project each matrix’s input activations on the subspace of the matrix’s rows; run the selected subcomponents.
   Post-SwiGLU activations h → row(W_down) (4096-dim of 14336); 
   2B. Run the selected subcomponents, project the output activations on the subspace of the matrix’s columns.
   circuit gate/up writes → col(W_gate), col(W_up).
   
Then, do the same thing for attention (following the same logic). All the intermediate data should be saved as either TSV or JSON files.

# Plots
The plots should be generated using a Marimo notebook, using data from the data files generated above.

1. Boxplots: grouped box plots, one group per intervention, showing the distribution of KL across the different prompts, for each conditions (including the circuit baseline).
2. Heatmaps: for each intervention, KL(a,b) grids raw | centered | bias-only | baseline on a shared log scale (one PNG per intervention per op), to see whether residual failures are structured (diagonals, operand bands) or diffuse.
3. Attribution scatter: per prompt, raw KL vs centered KL (colored by n_ci) — points on the diagonal mean the failure is informational; points far above it mean it's offset damage.

All the experimental scripts should go in @param_decomp_lab/scripts/validation/subspace_filtering. GPU jobs should be submitted through SLURM.
