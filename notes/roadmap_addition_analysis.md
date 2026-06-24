For the following objectives, the scripts should follow the general design of the @param_decomp_lab/scripts/validation folder. They should be documented in spec.md and sample commands should be stored in commands.md.
For this series of scripts, however, I want to filter subcomponents based on their mean causal importance across the 100x100 examples. So, by "alive components", all the following scripts refer to the subcomponents that were already flagged as alive by the existing script, and also that have a mean CI above the threshold. The default mean CI threshold should be 0.1.
Note that this depends on the operation of interest: for example some scripts may refer only to addition tasks. The subcomponents that are alive on addition might not be the same that are alive on multiplication etc.

Addition is referred to as "add", subtraction as "sub" and multiplication as "mult".

The data files names are suffixed with the operation of interest: for example, "inner_activations_add.tsb". As usual, the files should be stored in the run's folder.

For now, let's only run the scripts on addition, but all the scripts should support subtraction and multiplication (using the × symbol) as well. Run these experiments on the reference run: `addmult-L18-03`.

# Objective 1
First, let's run forward passes for the 100x100 prompts of the operation of interest (add, sub or mult) and store the hidden activations at the following hook points, all at the last token position:
- The residual stream just before layer 18's MLP
- The input vector to layer 18's MLP (after the RMSnorm)
- The neurons inside layer 18's MLP: store both the result of up projection and gate projection (before the nonlinearity is computed)
- The layer 18 MLP's output (after down proj, what gets added to the residual stream)
Store this as a npz file. 

This objective will require a GPU. Use a slurm job with only one GPU.

# Objective 2
Compute the inner activation of each alivesubcomponent, for each of the 100x100: the dot product between the input vector and the V vector of the subcomponent normalized to norm 1. 

Run this for each alive subcomponent in layer 18 MLP's up, gate and down matrix.
Make sure to use the direct input to each component (accounting for RMSnorm for the gate and up projections, and the nonlinearity for the down projection).

Save this as a TSV file with columns a, operation, b, matrix, subcomponent, inner_act.

# Objective 3
Most subcomponents activate periodically on either the first operand (a), the second one (b), or both. 
Write a script that takes each alive subcomponent and determines its activation period.
Use two different metrics: autocorrelation and and Fourier transform, performed on the inner activations determined above.
For each subcomponent, try 
For each metric, store the best-dcoring frequency and it's score according to the metric.
This should be done along both directions (first operand a an second operand b) and the results from both should be stored.
Produce a TSV file called "subcomp_periods.tsv".

# Objective 4
For the gate and up_proj subcomponents, compute the cosine similarities between either the u or v vectors of each pair of alive components.
The subcomponents should be sorted according to their activation period (obtained from the previous script). A thicker line should separate subcomponents of different periods on the heatmap.
For each direction (input vectors V or output vectors U) make a square heatmap plot showing the similarities between each pair of subcomponents.
The two heatmaps (one for v vectors and one for u) are displayed side by side. Use the RdBu color map.

Also make a plot (on a different file) for the down_proj subcomponents (they can’t be compared with the gate/up_proj subcomponents because the u/v vectors have different dimensionality).

# Objective 5
Build a HTML interactive visualizer to see which subcomponents are connected to which neurons.
To quantify the strength of a connection between a subcomponent and a neuron, normalize the subcomponents V and U vectors so V has norm 1 (and U is normalized accordingly to keep the outer product the same).
- For the pre-swiGLU matrices (up_proj and gate), the connection strength is the value of the U (output) vector for that neuron after normalization -- in other words, it’s the value that would be written to that neuron if the dot product of the previous activations and the V vector was 1.
- For the post-swiGLU matrix (down_proj), the connection strength is the value of V (input) for that neuron after V is normalized to norm 1.

The user picks operands a and b. Then, from left to right, the visualizer shows:
- (left) the subcomponents from the gate and up proj that are causally important on that prompt. The up subcomponents are shown on top, then the gate ones below. Then they are sorted by period.
- (center) the neurons that the active subcomponents write to (or read from). Show only the neurons that reach a connection strength above a threshold (that can be set with an input field) for at least one subcomponent (in any of the matrices). Sort the neurons vertically according to which input subcomponent (in gate or up_proj) has the highest connection strength, such that neurons appear roughly next to the subcomponents that write to them. Use connection strength as a secondary sorting key.
- (right) the subcomponents from the down proj that are causally important on that prompt, sorted by their periods.

When hovering a subcomponent , the applet shows its causal importance pattern (heatmap as a function of a and b).
When hovering a neuron, show its up, gate and output values for the current prompt.

Draw lines between subcomponents and neuron to show the connection strengths. Negative values are blue, positive values are red.
Use a white background and keep it simple and minimalistic. 

Use the playwright plugin to make sure the applet runs smoothly.
