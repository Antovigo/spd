# Autointerp prompt draft — v1

> This is a baked example prompt for a component from Jose (pile_llama_simple_mlp-4L).
> Everything below the line is what the LLM would see.

Meta notes: in the team we call "inner activation" "component activation" but I've used "inner" here as I think it's easier to understand one-shot.

---

### Context

Below you will be presented with data about a component of a neural network as isolated by a brand new Mechanistic Interpretability technique "Stochastic Parameter Decomposition". You will be tasked with describing the component in terms of its activation patterns on various text examples from a pretraining dataset, and other supporting evidence.

In Stochastic Parameter Decomposition, each weight matrix of a network is decomposed into C rank-1 parts, called "subcomponents", where C is usually greater than the rank of the weight matrix. These are parameterised as U • V, where V is the "read direction" (dimension `d_in` — what input patterns the component responds to) and U is the "write direction" (dimension `d_out` — what the component contributes to the output). They multiply to a rank-1 matrix of the shape of the original matrix, and can be thought to represent a one-dimensional slice of the computation the weight matrix does.

These subcomponents are learned in an unsupervised manner under 3 main losses:
- Faithfulness: the C rank-1 subcomponent matrices must sum to the original weight matrix - this should be a direct factorisation of the original weight matrix.
- Minimality / Simplicity: For a given datapoint, as few subcomponents as possible should be necessary for the network. Alternatively - as many subcomponents as possible should be ablatable
- Reconstruction: The network should nonetheless reproduce the behaviour of the target network.

In order to facilitate this, we train a small auxiliary "Causal Importance Network" which produces a mask of Causal Importance values in (0, 1) for each datapoint/token. The minimality/simplicity loss above incentivizes this mask to be sparse.

At each token position, each component has 2 "activation"s, which are different in important ways:
1. A Causal Importance value: How causally important is it that this rank-1 subcomponent of this weight matrix is present for this token. 
2. An "Inner Activation" value: With no ablations, what is the inner activation of this component, i.e. what is the scale of `x @ V` where x is the input vector

While these 2 values - the causal importance and the inner activation - are correlated, they are meaningfully different. A large inner activation without high CI means the input happens to align with the component's read direction, but the component's contribution isn't needed for this token's output.

A component is said to "fire" when its causal importance exceeds a threshold. The data below uses a CI threshold of 0.0, so even very weakly important positions are included as firings.

**Sign convention:** An important thing to understand about **inner activations** is that there is no inherent meaning to their sign - negating both uᵢ and vᵢ produces the same subcomponent. This means that, across examples, sign is not an important piece of information. However, sign can be meaningful *within* a component's examples: positive and negative activations produce opposite contributions to the output, and may correspond to qualitatively distinct input patterns (e.g. negative activations on one token class, positive on another). Check whether examples cluster by activation sign, whether they seem to have 2 regimes, etc. but avoid making conclusions about a specific sign. A broad positive inner activation does not signify excitation, and negative does not signify suppression, for example. Treat sign roughly how you would in the context of an embedding vector - while a positive or negative value at index x may have a specific semantic meaning, there is no global meaning.

### Context
The component you will be labeling today comes from a decomposition of a 4-block transformer trained on The Pile. Specifically, it is part of the GLU up-projection matrix in the 2nd of 4 blocks. It has a firing rate of 3.41% (fires ~1 in 29 tokens). The target model has ~42M parameters — keep its expected capability in mind, it is not a smart model.

## Evidence:

### Output token statistics

At each position where the component fires, we look at the model's next-token prediction distribution. The following tokens have the highest PMI (pointwise mutual information) between the component firing and the model assigning high probability to that token as its next-token prediction. A positive PMI value means this token is predicted more often than its base rate when the component fires. The value is in nats: 0 = no association, 1 ≈ 3× base rate, 2 ≈ 7×, 3 ≈ 20×.

**Top output tokens by PMI:**
- `' the'`: 2.31
- `' a'`: 1.89
- `' an'`: 1.74
- `' The'`: 1.55
- `' his'`: 1.42
- `' her'`: 1.38
- `' their'`: 1.21
- `' my'`: 1.10
- `' its'`: 0.98
- `' this'`: 0.91


### Activating examples

The following **activating examples** are sampled uniformly at random from all positions in the dataset where the component fires (CI above threshold). For each sampled activation location, we extract both a leading and trailing window of tokens centered on the firing position, with up to 20 tokens of context on each side. Windows are truncated at sequence boundaries — so a firing at the beginning of a training sequence will have little or no left context. This truncation is itself evidence (e.g. a component that consistently fires near the start of sequences). we include annotations for **all** firing positions in the window - not just the firing which was sampled to produce the window, however we don't include inner activations for all tokens - this would be too noisy - all tokens have at least epsilon inner activation on almost all components.

The training data consists of variable-length documents concatenated with `<|endoftext|>` separator tokens between them, then sliced into fixed 512-token sequences. This means `<|endoftext|>` tokens can appear anywhere within a sequence (not just at the start), and a single sequence may contain parts of multiple documents. If you see `<|endoftext|>` in examples, it is a literal token the model processed, not a formatting artifact.

Each example is shown as an XML block with two views:
- `<raw>`: the literal token text of the window
- `<annotated>`: the same window with firing tokens wrapped as `[[[token (ci:X, act:Y)]]]`

**Annotation legend:**
- **ci** (causal importance): 0–1. How essential this component is at this position.
- **act** (inner activation): inner product with the component's read direction. See the sign convention note above — within one component, sign separates distinct patterns. These values are normalised so that typical magnitudes fall roughly in (-1, 1), but this is just a presentation decision. Inner activation magnitude has a similar invariance to sign — intra-component inner activation value differences are meaningful, but inter-component differences aren't.

1.
<example>
<raw><![CDATA[The quick brown fox jumped over the lazy dog. The]]></raw>
<annotated><![CDATA[The quick brown fox jumped over the lazy dog[[[. (ci:0.94, act:0.31)]]] The]]></annotated>
</example>

2.
<example>
<raw><![CDATA[from the data, we conclude that the hypothesis]]></raw>
<annotated><![CDATA[from the data[[,[++ (ci:0.88, act:0.27)]]] we conclude that the hypothesis]]></annotated>
</example>

3.
<example>
<raw><![CDATA[running down the hill; the wind was cold and]]></raw>
<annotated><![CDATA[running down the hill[[[; (ci:0.91, act:0.34)]]] the wind was cold and]]></annotated>
</example>

4.
<example>
<raw><![CDATA[He said nothing]]></raw>
<annotated><![CDATA[He said nothing[[[. (ci:0.82, act:-0.19)]]]]]></annotated>
</example>

5.
<example>
<raw><![CDATA["I don't know," she replied. "Maybe]]></raw>
<annotated><![CDATA["I don't know[[,[++ (ci:0.79, act:0.22)]]]" she replied[[[. (ci:0.86, act:0.29)]]] "Maybe]]></annotated>
</example>

6.
<example>
<raw><![CDATA[<|endoftext|>The first time I saw]]></raw>
<annotated><![CDATA[<|endoftext|>[[[The (ci:0.71, act:-0.41)]]] first time I saw]]></annotated>
</example>

7.
<example>
<raw><![CDATA[<|endoftext|>In this paper we present a novel approach to]]></raw>
<annotated><![CDATA[<|endoftext|>[[[In (ci:0.68, act:-0.38)]]] this paper we present a novel approach to]]></annotated>
</example>

8.
<example>
<raw><![CDATA[the value of x is 0.5, and the result]]></raw>
<annotated><![CDATA[the value of x is 0.5[[,[++ (ci:0.85, act:0.25)]]] and the result]]></annotated>
</example>

## Task

Based on all the above context and evidence, please give a label of 8 words or fewer for this component. The label should read like a short description of the job this component does in the network. Please also provide a short summary of your reasoning. Use all the evidence: activation examples, token statistics, and activation values. Be epistemically honest — express uncertainty when the evidence is weak, ambiguous, or mixed. Lowercase only.
