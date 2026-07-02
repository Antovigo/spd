Let’s make a suite of toy model transformers with known ground truth that we can use to test and benchmark the parameter decomposition setup. There are two sources for toy models:
- The Tracr paper: these are compiled transformers made by assembling rank-1 operations. Presumably, they have a defined ground truth for parameter decomposition (the rank-1 operations used to assemble them). Since they were not trained using gradient descent, they are less realistic.
- The InterpBench paper: (ignoring the IOI transformer) these are more realistic versions of the Tracr models, that were trained by gradient descent so they are more realistic. They should implement the same computation as the Tracr circuits, however the rank-1 subcomponents will not be the same. They likely have a similar structure (so the number of subcomponents that activate on a given prompt/module should be the same, and they should encode the same information)
Our goal is thus to implement a parameter decomposition scheme to decompose each of these toy models, and also find what the ground truth should be like, so we can programmatically evaluate whether a given decomposition run gave the correct result.

All the work for this project must be done on the "feature/interpbench" branch (in the "interpbench" folder of the worktree). 
At the end, we should have decomposition scripts for select Interpbench models, as well as reference decomposition configs. All should go in the @param_decomp_lab/experiments/lm/interpbench folder.

Throughout this roadmap, write down any findings and observations in @notes/interpbench/notebook.md.

# Objective 0: find a subset of useful Interpbench models
Let’s first write a table (in TSV format) of the different InterpBench models with their name (if any is defined), id, task, input data format, expected output format, number of layers, number of parameters (total and non-embedding only), whether they use magnitude or only qualitative features, and comments. Comments can be anything that you think is important to flag -- especially regarding whether it will be easy to generate a ground truth for this model, whether it has architectural properties that differ from the rest, etc.
Also take note of whether the model uses one-hot encoded inputs and outputs a probability distribution over a vocabulary. If some toy models don’t do this, we shouldn’t use them as test cases for PD.’
The table should be stored in @notes/interpbench/models.tsv.
Then, suggest a model we could use for initial testing. It’s shouldn’t be trivially simple, but it should have only categorical variables (no magnitude-encoded features) and use both attention and MLPs. Ask me for confirmation before proceeding to the next objective.

# Objective 1: write a decomposition implementation
Now, we can try to decompose the selected toy model. Ideally, we want a decomposition script that would work for any of the  Interpbench models, as much as possible (even though extending it to other models would probably require making new data generators).

Then, implement the decomposition for the toy model selected above.
The InterpBench models use a HookedTransformer format, meaning they can’t currently be loaded directly from the lm script. Thus, they have to be converted: add a new target "kind" in lm/run.py. Importantly, if the model uses one-hot inputs and outputs a probability distribution over vocab, we don’t want to decompose the embedding and unembedding matrices. All the other matrices should be converted to nn.Linear modules. If the model uses fused Q, K, V matrices, they should be split into three separate matrices (but different heads should remain as one matrix). 

For data generation: For each target case, generate inputs by sampling random token sequences over the case's vocabulary (`get_vocab()`, prefixed with `TRACR_BOS`), encoding them to `input_ids` via the loaded HookedTransformer's own tokenizer so the ids match what the model was trained on — vendoring just `circuits_benchmark`'s vocab definitions and sampling loop rather than installing the library. 

Important: throughout the implementation, everything should be as minimal as possible. These are simply new toy models for testing -- implementing them should touch as few files as possible, and only change the rest of the code if absolutely necessary.

As a check for the model loading, build a simple test that runs the model in the converted format and checks it’s accuracy against 1) the original version of the model, 2) the output labels (to ensure that the data generations and tokenization all went right).

After doing this, run a code review (in high setting), fix the issues and commit the changes.

# Objective 2: find good hyperparameters for the decomposition of case 19
The goal here is to get high reconstruction accuracy and low final L0 on the case 19 InterpBench transformer. 
Take inspiration from the Pile-llama-4L transformer, except this model is vastly smaller, so everything can be scaled down -- in particular, it should be possible to obtain a good decomposition in less than 15,000 steps. For example, each matrix in case 19 is unlikely to have more than a dozen alive subcomponents, so a C value of 100 per matrix is probably plenty.
A good decomposition would have a final PGDReconLoss below 0.02, and then the L0 (on 10 eval batches) should be as low as possible under the PGD constraint.

# Objective 3: build an eval to monitor the interpbench decomposition
TBA.

