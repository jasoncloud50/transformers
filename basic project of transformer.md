# GPT Language Model on WikiText: Training, Evaluation, and Failure Analysis — Project Report

## 1. Project Overview

This project implements and trains a Decoder-only GPT language model from scratch. The model is pretrained on the WikiText corpus with a next-token prediction objective, and greedy search is used for text generation to examine the model's capabilities. Training ran for 10 epochs, reaching a final training accuracy of about 26%; however, the validation loss kept rising, and the generated text degenerated into repetitions of high-frequency words and punctuation. This report fully documents the model configuration, the training process, and the generation results, provides an in-depth analysis of the root causes of the generation failure, and finally presents actionable improvement measures.

## 2. Model and Training Configuration

### 2.1 Model Architecture

| Configuration | Value | Description |
|---|---|---|
| mode | GPT | Decoder-only autoregressive architecture |
| embedding_dimension | 512 | Token embedding dimension |
| num_of_attention_head | 8 | Number of attention heads |
| hidden_units_dimension | 1024 | Feed-forward network hidden dimension |
| decoder_layers | 12 | Number of Transformer decoder layers |
| block_size | 64 | Context window length |
| max_len | 5000 | Maximum positional encoding length |
| dropout | 0.01 | Dropout ratio |
| use_flash_attention | true | Flash Attention acceleration enabled |

### 2.2 Training Hyperparameters

| Configuration | Value |
|---|---|
| learning_rate | 1e-5 |
| num_epochs | 10 |
| batch_size | 64 |
| num_workers | 4 |
| is_same_language | false |

## 3. Training Process and Results

### 3.1 Training Log Summary

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Grad Norm | Throughput (tokens/s) |
|---|---|---|---|---|---|---|
| 1 | 7.3945 | 0.0972 | 7.4989 | 0.1523 | 1.4124 | 40571.60 |
| 2 | 7.0508 | 0.1696 | 7.9653 | 0.1856 | 2.4777 | 39861.49 |
| 3 | 7.1445 | 0.1938 | 8.5314 | 0.1924 | 3.0945 | 33008.48 |
| 4 | 7.2930 | 0.2083 | 9.1356 | 0.1972 | 3.3838 | 33964.13 |
| 5 | 7.4844 | 0.2201 | 9.7641 | 0.1985 | 3.4945 | 39748.15 |
| 6 | 7.6953 | 0.2296 | 10.4019 | 0.1989 | 3.6563 | 39542.64 |
| 7 | 8.1016 | 0.2437 | 11.5953 | 0.1988 | 4.0730 | 40162.80 |
| 8 | 8.2891 | 0.2501 | 12.1908 | 0.1949 | 4.2769 | 39062.14 |
| 9 | 8.4688 | 0.2566 | 12.7865 | 0.1959 | 4.4947 | 39285.96 |
| 10 | 8.6484 | 0.2629 | 13.3840 | 0.1931 | 4.7506 | 39154.25 |

### 3.2 Key Observations

The training process exhibits an **abnormal curve**, which can be summarized in three phenomena:

1. **The training loss rises instead of falling.** Training loss climbs monotonically from its minimum of 7.05 at epoch 2 to 8.65 at epoch 10. Under normal circumstances the training loss should decrease (approximately) monotonically; its reversal here is an important signal that the optimization process is unhealthy.
2. **The validation loss deteriorates continuously and dramatically.** Validation loss rises monotonically from 7.50 to 13.38, an increase of 78%. Superficially this resembles typical overfitting, but combined with phenomenon 1 (the training loss is also rising), we can conclude: **the model is not "memorizing the training data" — it is learning a "shortcut strategy" that becomes progressively worse on both the training and validation sets**.
3. **Accuracy diverges from loss.** Training accuracy steadily improves from 9.7% to 26.3%, and validation accuracy also rises to about 20% before stalling. Rising accuracy together with rising loss indicates that the model's predicted distribution is becoming more and more "confidently concentrated" — but concentrated in the wrong direction. This is precisely the core of the analysis below.

In addition, the grad norm grows continuously from 1.41 to 4.75, indicating that parameter updates are becoming larger and larger: the optimizer is not converging, but gradually moving away from the low-lying regions of the loss surface.

## 4. Generation Results Evaluation

Using greedy search with the prompt "Let me tell you a story:", the model produces the following output (excerpt):

> , and the " the " the " " the " " " , and the " the " th e " " " , " The " the " " " " , " The " " " " , " The " " " , " The " " " , " " " " " " ' " " " ' " " " ". " @ " " @ "... @.S @.@ " @ " @ " @. @ " @-@@ = = = = = @-@ " @-@@@ = = = = = = ...

The generation result is a **complete failure**: the model does not produce any meaningful English sentences, and instead degenerates into loops of three kinds of high-frequency tokens:

- High-frequency function words: "the", "and", ","
- Quotation marks and whitespace: `"`, `'`
- Noise symbols: "@", "=", "-"

The model has clearly "learned" the **marginal frequency distribution** of tokens in the corpus (which tokens are most common), but has completely failed to learn the **conditional distribution** (which token is appropriate given the context). The next section analyzes the essential cause of this phenomenon.

## 5. Root-Cause Analysis

### 5.1 Essence: The "High-Frequency Token Shortcut" under Objective Mismatch

The essence of this phenomenon is **objective mismatch**: we expect the model to learn "predict the most plausible next word given the context", but the actual minimization path of cross-entropy loss contains a low-cost shortcut — **predicting high-frequency corpus tokens at every position**.

Specifically, function words such as "the", ",", ".", and "and" have extremely high marginal frequencies in English text (together accounting for more than 10% of all tokens). A model that completely ignores context and simply outputs the marginal frequency distribution at every position can obtain a "not-too-bad" loss and roughly 20%+ token accuracy — because about one fifth of the tokens in the corpus really are these high-frequency words. This is exactly what the logs show:

- **Accuracy rises to about 26% and stalls**: this is the frequency ceiling of high-frequency tokens; improving further requires genuinely understanding context, but the model never took that path;
- **Validation loss explodes continuously**: the shortcut strategy concentrates the probability mass ever more sharply on a few high-frequency tokens (the distribution becomes sharper); whenever the target token is not a high-frequency word — which is the case at most positions in the validation set — the model pays an enormous cross-entropy penalty. The more the model trains, the more confident it becomes, the smaller p(correct token) gets, and the larger −log p becomes, so the loss soars monotonically;
- **Greedy decoding amplifies the degeneration**: greedy search takes the argmax at every step, and the model's argmax at every position is "the", a quotation mark, or "@", so the output is necessarily an infinite repetition of these symbols.

It must be emphasized: **this is not overfitting — it is underfitting**. Overfitting is defined as the model fitting the training set too well (very low training loss) while generalizing poorly; but here the training loss itself is rising, which means the model has not even fitted the training data properly. The model is merely trapped in a local pitfall of "lowering the average loss with high-frequency words", and its effective capacity has not been genuinely utilized. The possible factors causing the underfitting include:

1. **Insufficient dataset size**: the corpus is too short; the statistical advantage of high-frequency words in the gradient signal is amplified, and the model is easily captured by the "shortcut solution" before it can learn genuine linguistic structure;
2. **Regularization too strong relative to effective capacity / limited model capacity**: although dropout is only 0.01, relative to the already modest data volume and the 512-dimensional embedding capacity, any additional perturbation further depresses the effective capacity;
3. **Context window too short (block_size = 64)**: long-range dependencies cannot be modeled, making it hard for the model to obtain predictive signals from context more useful than the "marginal frequency", which further encourages the high-frequency shortcut;
4. **Learning rate too low (1e-5), causing the optimization to stall in a poor local minimum**: see Section 6.3 for details.

### 5.2 Why Does Loss Rise While Accuracy Also Rises?

The two are not contradictory. Accuracy measures only whether the argmax hits the target, while loss measures the quality of the entire distribution. The process of the model sinking deeper into the "high-frequency trap" is: the argmax lands ever more stably on high-frequency words (accuracy slowly rises to the frequency ceiling), while the distribution becomes ever sharper (confidence rises), and −log p at the wrong positions grows sharply (loss soars). This pair of diverging curves is precisely the fingerprint of "shortcut learning".

## 6. Improvement Measures

Sections 6.1–6.4 present four measures targeting the model and its hyperparameters; Section 6.5 collects auxiliary suggestions. All measures are mutually independent yet composable.

### 6.1 Lower the Dropout Ratio

**Current setting**: dropout = 0.01.

**Mechanism**: Dropout randomly zeroes out activations during training to prevent feature co-adaptation; in essence it sacrifices part of the **effective capacity** in exchange for generalization. But this is a net gain only when the model has already fitted the data sufficiently and overfitting risk has emerged. The diagnosis of this model is **underfitting** — even the training loss is rising, meaning the model has not yet learned the basic statistical structure of the data. Applying dropout at this point is like cutting the model's feeding time while it is still underfed: the number of effectively available neurons in each forward pass is further reduced, making it harder to express the delicate conditional mapping "context → next token", so the model can only fall back on the coarse strategy of "outputting high-frequency words", which is inherently robust to perturbation (the high-frequency strategy depends on no particular neuron and is naturally immune to dropout — which is exactly why the shortcut solution becomes *more* competitive under regularization).

**Recommendation**: Reduce dropout to 0 (or keep a very small dropout only on attention weights). Watch whether the training loss resumes a monotonic decrease: if the training loss descends smoothly and the validation loss only rises later, the underfitting has been lifted, and dropout can then be gradually added back to handle genuine overfitting.

### 6.2 Increase the Embedding Dimension

**Current setting**: embedding_dimension = 512, hidden_units_dimension = 1024.

**Mechanism**: The embedding dimension determines the information bandwidth of each token's representation — it is the "master valve" of the model's capacity. No matter how deep the 12 decoder layers are, each layer can only process information within a 512-dimensional representation space. Language modeling requires simultaneously encoding word meaning, part of speech, syntactic role, coreference, topic, and many other dimensions of information; 512 dimensions is tight for a WikiText model with a non-trivial vocabulary. When capacity is insufficient, the model is unable to "distinguish different contexts": many different contexts are squeezed together in the representation space, and for them the model can only produce a compromise output biased toward high-frequency words. Increasing the embedding dimension (e.g., 512 → 768 or 1024, while proportionally raising hidden_units_dimension to 2048–4096, keeping the classic ratio of about 1:4) can:

1. Enlarge the representation space so that different contexts become geometrically separable, making it possible for the conditional distribution to be learned at all;
2. Increase the expressiveness of the attention heads (with the number of heads fixed, each head's dimension grows from 64 to 96/128, allowing a single head to capture richer dependency patterns);
3. From the perspective of loss-surface geometry, a larger parameter space makes the "barriers" between poor local minima easier to bypass (the loss surfaces of over-parameterized networks are smoother, with more saddle points than local minima), so optimization can more easily escape the high-frequency trap.

**Cost and trade-off**: Parameter count and memory footprint grow approximately quadratically with the dimension (embedding, QKV projections, and the output projection are all on the order of d²), and training throughput will decrease. Given that the current throughput of 40k tokens/s still has headroom, this cost is acceptable.

### 6.3 Increase the Learning Rate to Escape the Local Minimum

**Current setting**: learning_rate = 1e-5, constant, while the grad norm climbs monotonically from 1.41 to 4.75.

**Mechanism**: 1e-5 is **significantly too low** for training a 12-layer GPT from scratch (a common starting point at this scale is 3e-4 to 1e-3 with warmup). An overly low learning rate has two consequences:

1. **The step size is too small to escape the poor basin of attraction.** The high-frequency shortcut corresponds to the edge of a wide, shallow basin on the loss surface — the basin's entrance is wide, so optimization falls into it easily in the early stage; climbing out requires parameter updates with sufficient energy. A step size of 1e-5 leaves the model trapped at the basin's edge, "oscillating in place". This explains why the training loss rises instead of falling: the model repeatedly adjusts the sharpness of the distribution near the shortcut solution (which lets accuracy creep up), but never manages to cross the barrier into the region that genuinely exploits context.
2. **The continuously increasing gradient norm is a danger signal.** The more the parameters are updated, the larger the gradient becomes, indicating a severe mismatch between the optimization direction and the curvature of the loss surface: the model is being pushed toward higher-loss regions instead of converging. This is typical behavior of falling into an ill-conditioned curvature region under a small learning rate with no scheduling strategy.

**Recommendation**:

- Raise the peak learning rate to **3e-4** (conservatively, start with 1e-4), together with **linear warmup (first 500–2000 steps) + cosine decay**. Warmup prevents large early gradients from scattering the randomly initialized parameters, and cosine decay ensures the optimization can settle into a flat minimum late in training;
- Add **gradient clipping (e.g., max_norm = 1.0)**. The current grad norm has already reached 4.75 and is still growing; clipping suppresses destructive updates caused by occasional abnormal batches and keeps training stable under the larger learning rate;
- The update "noise" brought by a larger learning rate itself has an implicit regularization effect, helping the optimization jump out of sharp, poor basins. It complements the two measures in Sections 6.1 and 6.2: the capacity expansion provides a region "worth going to", and the learning-rate increase provides the momentum to "get there".

### 6.4 Enlarge the Context Window (block_size)

**Current setting**: block_size = 64.

**Rationale**: With only 64 tokens of context, long-range dependencies simply cannot be modeled — the model cannot see far enough to obtain predictive signals from context that are more informative than the marginal frequency. As a result, the information gain of the conditional distribution never exceeds that of the marginal distribution, and the high-frequency shortcut naturally remains a local optimum. Enlarging block_size (e.g., 64 → 256 or higher) gives the model enough context to make genuine conditional prediction worthwhile, so that the shortcut solution is no longer locally optimal. This should be done jointly with enlarging the dataset (Section 6.5.1): a longer window only pays off when the corpus contains enough long-range structure to exploit.

### 6.5 Auxiliary Suggestions

Beyond the four measures above, the following complementary steps address the problem and help verify the improvements:

1. **Enlarge the dataset**: the fundamental reason the shortcut solution is attractive is that the data is too scarce and the high-frequency statistical signal is too strong. A longer corpus naturally dilutes the relative advantage of high-frequency tokens, and is the most direct way to weaken the shortcut;
2. **Improve training monitoring**: beyond loss and accuracy, monitor the **perplexity**, the **entropy of the output distribution**, and the **frequency composition of the top-10 predicted tokens**. If the entropy keeps decreasing and the top tokens are fixed to high-frequency words, shortcut learning can be identified at an early stage;
3. **Generation-side remedies**: before the model quality reaches an acceptable level, replace greedy decoding with temperature sampling or top-k / top-p (nucleus) sampling, to prevent the argmax from amplifying the degeneration into a visibly repetitive stream of symbols.

## 7. Conclusion

This project successfully built and trained a 12-layer, 512-dimensional GPT language model, with training throughput stable at 33k–40k tokens/s; the engineering implementation (Flash Attention, per-epoch checkpoint saving, etc.) works correctly. However, the model quality fell short of expectations: after 10 epochs, the training loss rose instead of falling (7.05 → 8.65), the validation loss deteriorated by 78%, and greedy generation degenerated into repetitions of high-frequency words and symbols.

The analysis shows that the essence of the problem is not overfitting but a superposition of **underfitting and objective mismatch**: cross-entropy loss admits the low-cost shortcut of "predicting high-frequency words", and under the joint effect of insufficient data, limited effective capacity, and an overly low learning rate, the model became trapped in a poor local minimum corresponding to the shortcut solution. The diverging curves of slowly rising accuracy and soaring loss are the outward manifestation of this process.

The targeted improvement path is now clear: **lower the dropout ratio (remove the regularization constraint during the underfitting phase), increase the embedding dimension (expand the effective capacity), and increase the learning rate together with warmup and gradient clipping (provide the optimization momentum to escape the local minimum)**, supplemented by enlarging the dataset, lengthening the context window, and strengthening training monitoring. After these measures are implemented, the diagnostic indicators in Section 5.1 (monotonicity of the training loss, entropy of the output distribution, and the composition of top tokens) can be used to verify whether the model has escaped the shortcut trap and learned a genuine conditional language distribution that produces fluent, coherent text.
