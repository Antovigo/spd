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

# Objective 6
Estimate and reduce the real dimensionality of the subcomponent representation, separately for the **input space** (what the up and gate subcomponents read: the post-RMSNorm MLP input — not the raw residual stream — projected onto the span of their unit V directions) and the **output space** (what the down subcomponents write: the MLP output projected onto the span of their unit U directions). Both activations are taken at the last token, directly from the grids stored in Objective 1 (`mlp_input` and `mlp_output`).

The subcomponent directions are non-orthogonal and redundant (several may read/write the same plane), so build an orthonormal basis Q of their span from the Gram matrix G of the unit directions (G = E Λ Eᵀ, keeping the non-negligible eigenvalues; Q = Dᵀ E Λ^(-1/2), where D stacks the unit directions). The reduced representation is the stored activation projected onto that basis, z = Qᵀ x — a plain projection of the real activation (no reconstruction or synthesis), so the input and output sides are handled identically.

As a completeness check, report the fraction of each activation's variance that lands in the subspace (‖z‖² / ‖x‖²): how much of what the MLP actually reads / writes lives in the subcomponents' directions.

Run TwoNN (the scikit-dimension package) on z to estimate the intrinsic dimensionality of the input and output representations. The geometric rank (non-negligible G eigenvalues) and the linear effective dimension (participation ratio of the cov(z) / PCA spectrum) come for free from the construction and the scree plot below.

For verification, build an interactive HTML applet (generated with Plotly as a single self-contained, offline file) to visualize the structure of z:
- Show the scree plot (PCA eigenvalues of cov(z)), with a dialog to set the eigenvalue / rank threshold.
- Show interactive, rotatable 3D scatter plots, each with the points' shadow projected on the "floor". Show two orderings, both in consecutive groups of three axes: the raw orthonormal z-axes (z0–z2, z3–z5, …) and the PCA-ordered axes (PC1–3, PC4–6, …).
- Have a selector to colour the points by the first operand (a), the second operand (b), or the result (a+b for addition).
- Do all of the above for both the input and the output activations.
- Show the TwoNN result at the bottom of the page.

# Objective 7
Run Independent Subspace Analysis on z to find mutually independent blocks, without assuming what they encode. Whiten z, run ICA (scikit-learn FastICA), then group the components into subspaces by their magnitude (energy) correlation, so a circular feature — whose components are linearly uncorrelated but jointly dependent — is recovered as a single subspace. Output the subspace decomposition.

Interpret each block afterwards with the (a, b) grid (e.g. colour a 2D projection by a, b, a+b) and check the blocks are near-orthogonal via principal angles; the discovery itself stays unsupervised.

Again, build an interactive HTML applet (Plotly, self-contained offline) for visualization of the results, reusing the Objective 6 3D machinery (per-subspace 3D projection coloured by a / b / a+b) plus the energy-correlation and principal-angle heatmaps.

# Objective 8
Build a new HTML applet to visualize activations (either input space post-RMSnorm or the MLP output space) as a 3D scatter plot, using the subspace of 3 user-selected subcomponents.
On the right, there is a list of subcomponents to pick from, each with a thumbnail showing their inner-activation pattern as a function of the two operands. The user can pick up to 3. The plot is updated live.
For the input space, the available bases are the V vectors from the up_proj and gate subcomponents. For the output space, they are the U vectors from the down_proj subcomponents.
The scatterplot is interactive and can be rotated. At the bottom there is a dark grey shadow to help visualization. The shadow should always be at the bottom, and not rotate with the rest of the points. Let me know if this is complicated to implement.

# Objective 9
To work with multiplication, we need a way to assign periods to subcomponents that activate with a logarithmic pattern: that is, the inner activations should roughly periodic with respect to log(a) or log(b) (or both). This is made trickier by the fact that they are only periodic over a limited range: usually, for very small values of a or b, the resolution is too low to see any periodic pattern. For this objective, come up with a scheme to detect log periods — such that it works even if the periodicity is only apparent for values over some threshold, not the whole range. Of course, make sure you are not overfitting a small set of high value that happen to look periodic by coincidence. Try to find the period for which there is the most evidence that the subcomponent activates periodically, at this period, for x > some threshold value (which should be as low as possible).
For testing, here are subcomponents from the addmult-L18-03 decomposition that activate logarithmically, at different periods and only for high enough values. These are periodic in both a and b:
* mlp.up_proj#39
* mlp.down_proj#373
* mlp.up_proj#122
* mlp.up_proj#291

And these are periodic only in a:
* mlp.down_proj#112
* mlp.gate_proj#104

Once the detection scheme works well, incorporate it into the script that detects periods. 

# Objective 10
Make an interactive HTML app to investigate which neurons are involved in the arithmetic tasks. For now, let’s just support addition.

Required data:
- for each subcomponent x each neuron, we need their coefficient of interaction. For subcomponents in up_proj or gate (that write to the neurons), it’s the absolute value of u/||u|| at that neuron. For subcomponents in down_proj (that read from the neurons), it’s the absolute value of v/||v||.

The applet should have two panels, each taking on horizontal half of the screen.

The left panel should be a 2D heatmap:
- horizontal axis: alive subcomponents, sorted by period, then by mean CI over the 100x100 addition target data
- vertical axis: neurons, sorted by their total coefficient of interaction across subcomponents (across all matrices)
- color: the coefficient of interaction. Subcomponents that write to the neuron should be shown on a white-to-blue color scale, and subcomponents that read from the neuron should be show on a white-to-red scale. A simple way to do that is to use a RdBu diverging scale, and flip the sign of output subcomponents (since coefficients of interaction are always positive).
  
  The heatmap should only show as many neurons as fit on the screen (the number of alive subcomponents is a good starting point). Then, there should be buttons to scroll to the next/previous page.
  By clicking on the heatmap, the user selects a neuron/subcomponent pair, which gets highlighted with a thin black border. This displays information in the right panel.

  The right panel shows information about the selected neuron/subcomponent pair:
  - the inner activation heatmap of that subcomponent as a function of the two operands a and b
  - heatmaps of the up_proj, gate and output values for the selected neuron as a function of the two operands a and b

# Objective 11
Let’s redefine the coefficient of interaction, so it better accounts for the actual variation of activations over the target data.
Let x be the subcomponent vector that interacts with the neurons: u for input matrices and v for the output matrix. Currently the coefficient of interaction is abs(x/||x||). This means that a subcomponent with an overall low ||x|| will get high coeffs of interaction, even if this subcomponent has little effect on the neuron.
So let’s adopt a new definition, based on the standard deviation of activations across the target dataset. This will be called "interaction score" and replace the previous coefficient of interaction.
- for input subcomponents, a neuron’s interaction score is the standard deviation of what the subcomponent writes to that neuron over the target data. It can be calculated from the subcomponent’s inner activations and the subcomponent’s u and v vectors (remember that the inner activation is defined as the dot product between the input activations and the subcomponent’s normalized input vector  v/||v||).
- for output subcomponents, a neuron’s interaction score is simply the standard deviation of the neuron’s final activation (after swiGLU, i.e. what’s fed to the down_proj for this neuron) multiplied by the the subcomponent’s normalized input vector v/||v|| for that neuron.
This means we have two slightly different metrics for input and output subcomponents, but it should be fine for now.
Update the neuron investigator so it uses the new metrics instead of the previous ones.

Add a selector to filter neurons based on their max interaction score across subcomponents (so I can type in a threshold).

# Objective 12
Let’s upgrade the Subspace Scatter applet, to inspect what happens inside the MLP. Currently, there’s a dropdown called "side" that can be either input or output. Let’s add two more option: "pre-nonlinearity" and "post-nonlinearity". Pre-nonlinearity shows the hidden activations before the nonlinearity is computed (i.e., the values directly out of the up_proj or gate), in the subspace of the selected subcomponent’s out vector (u). As before in this applet, the actual directions of the subcomponents out vectors should be shown as red arrows. The subcomponents can be selected from the up or gate matrices.
Post-nonlinearity shows the hidden activations after the nonlinearity (the actual outputs of the neurons) in the subspace of the selectod subcomponent’s input vectors (v). The subcomponents can be selected from the down matrix.

# Objective 13: Fourier features
In Feucht et al, 2026 (https://arxiv.org/pdf/2605.01148v1), they show that Llama-3.1-8B encodes both the operands as circular features in the output space to L18’s MLP, and write the result of the addition to the residual stream after L18’s MLP. The objective here is to find the planes in which the circular features for each period T (in 2, 5, 10, 20, 50 and 100) lie. For inputs, use the activations after the RMSnorm. For outputs, use the MLP’s output (what it writes to the residual stream).
Replicate Feucht et al probing strategy for finding the Fourier features, by writing a new script in the validation scripts folder. The activations should be collected on both addition, subtraction and multiplication tasks. Fourier features should be fit separately for each task (and written to a different JSON file, with the appropriate suffix).
Write the resulting vectors for each canonical period, as well as their offset (the center of the circle) to a JSON file in @~/out/runs/fourier_features/coordinates.json. 

# Objective 14: compare subcomponents to Fourier features
Next, we write an HTML applet that displays the activations projected onto the Fourier basis of the each period. There should be dropdowns to pick the activation task (add, sub or mult), the task used to find the basis (add, sub or mult), and the operand (first input operand, second input operand, or result in outputs). In other word, it’s possible to plot the activations for subtraction projected on the Fourier basis for addition, etc. There is one plot for each period -- the different plots for the different frequencies are shown side by side.
The points can be colored by a, b, or result. It’s possible to zoom on the plot by scrolling or dragging.
The (unit-normalized) subcomponents are projected on the Fourier basis and displayed as arrows starting at the zero point -- similar to the subspace scatter plots. The goal is to clearly see whether the subcomponents accurately capture the Fourier features. Only subcomponents that have a norm on the projected basis above threshold are displayed. There’s a form to set the threshold.
When the user clicks the arrow head for a component, the inner activation heatmaps for this subcomponent are shown in a panel at the bottom.

In addition, instead of showing the directions of subcomponents’ vectors, there should be an option to show individual neurons’ input or output directions (for gate, up or down proj). This way, we can tell whether there are directions that are captured well by neurons but not subcomponents, or the opposite.
