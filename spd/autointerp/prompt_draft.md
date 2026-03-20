# Autointerp prompt draft — v1

> This is a baked example prompt for a component from Jose (pile_llama_simple_mlp-4L).
> Everything below the line is what the LLM would see.

---

Describe what this neural network component does. 

### Decomposition method

Each component is a rank-1 parameter vector learned by Stochastic Parameter Decomposition (SPD). A weight matrix W is decomposed as a sum of outer products W ≈ Σ uᵢ vᵢᵀ. When the model processes a token, each component computes an activation: the inner product of the residual stream with its read direction vᵢ.

Each component also has a causal importance (CI) value predicted per token position: CI near 1 means the component is essential at that position, CI near 0 means it can be ablated without affecting the model's output. A component "fires" when its CI exceeds a threshold.

**Sign convention:** The decomposition has a global sign invariance — negating both uᵢ and vᵢ produces the same weight matrix. This means the absolute sign of a component's activation is arbitrary. However, sign is meaningful *within* a component's examples: positive and negative activations produce opposite contributions to the output, and may correspond to qualitatively distinct input patterns (e.g. negative activations on one token class, positive on another). Check whether examples cluster by activation sign.

### Context

- 4-block transformer trained on The Pile (diverse text: books, academic papers, code, web pages)
- Component location: GLU up-projection in the 2nd of 4 blocks
- Firing rate: 3.41% (~1 in 29 tokens)

### How to read the evidence

**Activation examples** are sampled uniformly at random from all positions in the dataset where the component fires (CI above threshold). Each example shows a window of tokens centered on a firing position, with up to 10 tokens of context on each side. Windows are truncated at sequence boundaries — so a firing near the start of a document will have little or no left context. This truncation is itself evidence (e.g. a component that consistently fires near the start of texts).

The model's tokenizer uses the following special tokens: `<|endoftext|>` (document boundary). If you see these in examples, they are literal tokens the model processed, not formatting artifacts.

Each example is shown as an XML block with two views:
- `<raw>`: the literal token text of the window
- `<highlighted>`: the same window with firing tokens wrapped as `[[[token (ci:X, act:Y)]]]`

**Annotation legend:**
- **ci** (causal importance): 0–1. How essential this component is at this position.
- **act** (component activation): inner product with the component's read direction. See the sign convention note above — within one component, sign separates distinct patterns.

**Token statistics** summarize correlations between this component's firings and specific tokens across the full dataset:
- **Precision**: P(component fires | token). Of all occurrences of token X, what fraction had this component firing?
- **PMI** (pointwise mutual information, in nats): How much more (or less) likely is co-occurrence than chance? 0 = no association, 1 ≈ 3×, 2 ≈ 7×, 3 ≈ 20×.

"Output" statistics concern the token the model predicts (at its final logits) at positions where the component fires — not the component's direct causal effect.

### Output token statistics

**Output PMI:**
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

### Activation examples

1. <example>
<raw><![CDATA[The quick brown fox jumped over the lazy dog. The]]></raw>
<highlighted><![CDATA[The quick brown fox jumped over the lazy dog[[[. (ci:0.94, act:0.31)]]] The]]></highlighted>
</example>
2. <example>
<raw><![CDATA[from the data, we conclude that the hypothesis]]></raw>
<highlighted><![CDATA[from the data[[,[++ (ci:0.88, act:0.27)]]] we conclude that the hypothesis]]></highlighted>
</example>
3. <example>
<raw><![CDATA[running down the hill; the wind was cold and]]></raw>
<highlighted><![CDATA[running down the hill[[[; (ci:0.91, act:0.34)]]] the wind was cold and]]></highlighted>
</example>
4. <example>
<raw><![CDATA[He said nothing]]></raw>
<highlighted><![CDATA[He said nothing[[[. (ci:0.82, act:-0.19)]]]]]></highlighted>
</example>
5. <example>
<raw><![CDATA["I don't know," she replied. "Maybe]]></raw>
<highlighted><![CDATA["I don't know[[,[++ (ci:0.79, act:0.22)]]]" she replied[[[. (ci:0.86, act:0.29)]]] "Maybe]]></highlighted>
</example>
6. <example>
<raw><![CDATA[<|endoftext|>The first time I saw]]></raw>
<highlighted><![CDATA[<|endoftext|>[[[The (ci:0.71, act:-0.41)]]] first time I saw]]></highlighted>
</example>
7. <example>
<raw><![CDATA[<|endoftext|>In this paper we present a novel approach to]]></raw>
<highlighted><![CDATA[<|endoftext|>[[[In (ci:0.68, act:-0.38)]]] this paper we present a novel approach to]]></highlighted>
</example>
8. <example>
<raw><![CDATA[the value of x is 0.5, and the result]]></raw>
<highlighted><![CDATA[the value of x is 0.5[[,[++ (ci:0.85, act:0.25)]]] and the result]]></highlighted>
</example>

### Task

Give a label of 8 words or fewer describing this component's function. The label should read like a short description of the job this component does in the network.

Use all the evidence: activation examples, token statistics, and activation values. Be epistemically honest — express uncertainty when the evidence is weak, ambiguous, or mixed. Lowercase only.
