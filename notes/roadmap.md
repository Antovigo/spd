Here is a roadmap for research. I want you to try to get as far as possible down this roadmap over the weekend. Once you have completed an objective, move on to the next objective. Take notes about your progress and your findings in a notebook md file in @notes. You can start as many decomposition runs as you want using SLURM jobs. Please never use more than 6 GPUs from the cluster at any point. Always use SLURM jobs for GPU tasks.

# Objective 1: implement new useful features
First, let’s add some features that will be useful in the future. For these two features, switch to the `feature/targeted` branch (checked out in the "targeted" folder of the worktree), and create a new branch ("feature/n_alive" for n_alive, "feature/ci_scaled_wd" for CI-scaled weight decay). Then, commit the new branch, switch back to `experiment/8B_targeted` and merge the features into that branch so you can use them. Here is a description of the features:
1) *New eval: n_alive*. Tracks the number of components that are active (above ci-thr, default to 0.1) at least once over the eval target batch. Should be calculated on the same eval batches as the other short eval on target data, then logged to wandb and metrics.jsonl. There should be one value of n_alive for each decomposed matrix.
2) *CI-scaled component weight decay*. After running a batch of target data, compute the maximal CI within the batch (at any position) for each subcomponent. Then, apply weight decay to the components according to 1 - max(CI), such that subcomponents that have a CI of 1 anywhere in the batch don’t decay at all, while subcomponents that never active at all (max(CI) = 0) decay the fastest. This should allow for very high weight-decay coefficents (typically 0.2).
   
# Objective 2: optimize the llama8b MLP 18 addition decomposition
The run "llama8b-add-02" is the standard decomposition for the targeted decomposition of addition tasks on layer 18’s MLP. It’s pretty good. Our goal is to replicate and optimize this decomposition so we can reach the same (or better) results in less time, using less compute.

The metrics for success are the final values for the RoundedReconLoss and total L0. A run will be considered successful if it is better than llama8b-add-02 on both of these metrics. In addition, the n_alive metric at the end of the run should be below 100 per matrix. 

Your goal is then to get this decomposition to work:
- On only 2 GPUs (instead of 4 currently)
- In less than 8h
Without degrading the end metrics. The final PGD reconstruction loss should also be below 0.1, and as low as possible.

Some things you can try:
- Decrease the number of available components per matrix ("C"). For this decomposition, we expect fewer than 100 alive subcomponents per matrix, so it’s probably possible to decrease C from 512 to lower values. In exchange, this should allow to increase the batch size (to fill up the 2 GPUs).
- Use impmin-coefficient scheduling. The idea is that high values at the beginning are helpful to reach sparse solutions quickly, and then once the decomposition is good enough the coefficient can be released gradually — at that point, subcomponents that already reconstruct the output well have no reason to change, while subcomponents that are a low-rank approximation of the true computation will be re-arranged to get better reconstruction. So, after a short warmup, the impmin-coefficient can be elevated to a high value (5x is a good starting point), then gradually released over training.
- Play with the learning rate
- Decrease the number of steps. Because of p-annealing and impmin-coeff scheduling, the training is not uniform, so be careful when optimizing this.
- Enable the unmasked reconstruction loss (where all subcomponents are enabled, but not the delta component)
- Enable CI-scaled component weight decay (a high value of 0.2 is probably helpful to reduce the number of alive components)
  
Don’t hesitate to start short runs, just to test hyperparameters and optimize them, before you start actual serious multi-hour runs. Once you manage to make a good run on addition tasks under 8h on 2 GPUs, commit the changes. Run the AB causal importance plotting function. Then, move on to the next objective.

# Objective 3: reconstruct only the last sequence position
Currently, even in targeted decomposition, we are trying to reconstruct the model’s output on every position in the sequence. But here, I just want to understand how the model performs addition. Therefore, it might be enough to find a decomposition that successfully reconstruct the model’s output on the last position (in this case, the "=" token).

Using the optimized parameters from objective #2, run a decomposition on addition tasks, only reconstructing the output distribution on the last position. I will check myself whether it worked or not when I’m back. You just have to run it. Then, run the AB causal importance plotting function, but only on position 4.

# Objective 4: subtraction
Starting from the optimized hyperparameters for the addition task, optimize the decomposition for the subtraction task (using subtraction only). The goal it to get reconstruction losses under 0.05 with a L0 under 10. The number of alive components should also be at least 20% below the total number of available components "C". You may need to increase C to make this work.

# Objective 5: multiplication
Same as objective 4, but for the multiplication prompts. Use "×" as the multiplication symbol.

# Objective 6: refine the best runs
For each of the previous cases (addition, addition only on the last token, subtraction and multiplication), make a new run, with the hope of getting more accurate components. Make the CI function slightly more powerful, and train for twice as many steps with a cosine LR that decays to zero. Release the importance minimality to a lower final value (while keeping the peak value the same). When these runs are done, run the AB matrices plots with a ci-thr of 0.8. 

# Objective 6-bis: refine the addition only on the last token
The current best run for this case has a higher final L0 than the all-position version. This is not normal: you should need fewer components if you drop the requirement to reconstruct the outputs of the first 3 tokens. Carefully optimize this run so it reaches a better L0 than the all-position version, while still having an equal or better reconstruction. If, after a few attempts, this turns out not to work, you are allowed to give up and move on to the next objective.

# Objective 7: addition + subtraction
Now, make a run where the target data combines both addition and subtraction. This may require increasing the number of available components and some extra optimization (don’t hesitate to do test runs). Intuitively, the number of alive components should be below the sum of the number of alive components for each of the tasks (addition or subtraction) taken separately, but it’s also possible that it becomes much higher. Try to make the best possible run, then make the AB plots for each of the two tasks. (use ci-thr=0.8).

# Objective 8: addition + multiplication
Same as objective 7, but combining the addition task and the multiplication task.

# Objective 9: addition on more layers.
Try to make the addition decomposition work on layer 18’s attention.
