I have several single-block targeted decomposition of llama-8B on arithmetic tasks (addition and subtraction, "addsub"). Now, I want to combine them, making sure that the network where all the eligible layers are replaced with their decomposed versions, and the objectives of PD are still satisfied.

Please work on the following objectives. Throughout, update the following files in @notes/combine_layers:
- spec.md: the specification of the different scripts -- what they are supposed to do.
- lab_notebook.md: your observations and findings, the problem you meet and how you solved them
- commands.md: for each script you create, the commands that need to be run (on the reference runs)
- report.md: a report with the key conclusions. Put plot files in a companion report_figure folder. Make sure the plots are easy to read and the data is easy to compare (for example, plots that show commensurable data should be shown against the same axis. Make sure the raw data is visible -- summary statistics are useful, but it’s better to have them as a thin overlay, and keep the focus on raw data. Don’t plot too many things -- only the most informative metrics. But also, don’t hide anything important.)
  
For testing purposes, let’s make this work for the following runs:
- addsub-L16-04-init-proj
- addsub-L17-04-init-proj
- addsub-L18-05-coupled
- addsub-L19-05

We want to complete all the objectives as fast as possible, so it’s often a good idea to run a well-chosen quick test, or start runs and then interrupt them once the wanted information has emerged.

# Objective 1: find whether the decompositions can readily be combined
If we simply take the existing decompositions, put them together and run all the add/sub tasks while enabling only the components whose CI value is above --ci-thr (default 0.1), how good is the reconstruction? Is it much higher than the rounded recon loss obtained at the end of training for the different decompositions?

# Objective 2: assemble by fine-tuning, with separate CI fn
Evaluate whether it is feasible to fine-tune the assembled network to optimize the collective decomposition, without changing the architecture -- each block still has its own separate CI-function network. You may use up to 4 GPUs: for example, the run add-6L-04 trained 6 layers at the same time, so the hardware presumably supports it.
Things worth trying:
- freeze the CI functions and train subcomponents only
- train both CI functions and subcomponents
You’ll need to combine the hyperparameters, and these are often different between runs. When chosing the combined hyperparameters, the rule of thumb should be that we would rather have an excellent PGD recon loss and a high L0 than the opposite. The priority is to get an accurate and faithful description of what the model is doing -- and, conditional on that, we want to achieve it with as little subcomponents as possible. We want the description to be a sparse as possible, but we don’t want to force sparsity beyond the natural sparsity of the model.

# Objective 3: re-train CI fn
Similarly to objective 2 (or if objective 2 is deemed unfeasible), another way to assemble the decompositions is to put the subcomponents in the same model, then re-train a single CI-function from scratch (using the same overall transform architecture, but possibly quantitatively bigger to accomodate the multiple layers). The aim is to obtain a combined model, like in objective 2, but using a much lighter combined CI-function instead of a separate one for each layer.

If this fails, an alternative would be to distill the multiple CI function into a single combined CI function, but that’s much more complicated -- only explore this direction if you deem it necessary.

# Objective 4: anti-redundancy training
[TBA]

# Objective 5: resurrect subcomponents
[TBA]
