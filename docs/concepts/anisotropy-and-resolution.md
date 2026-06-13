# Anisotropy, Resolution, and What ambit Measures

> Crowding in an embedding space is a loss of **resolution**: when items pack too
> tightly, cosine similarity can no longer tell them apart. This note explains the
> phenomenon, the academic vocabulary for it, how it is quantified, the failure
> modes it causes downstream, and how **ambit** makes it visible. It is the
> conceptual companion to the
> [embedding-space occupancy study](../design/embedding-occupancy-study/).
>
> Citations are linked inline; full entries with arXiv IDs / DOIs are collected in
> [§10 References](#10-references).

---

## 1. The problem: crowding = low resolution

An embedding model is only as useful as its ability to keep *distinct things
distinct*. If two unrelated items land at nearly the same direction, then any
metric built on that geometry — cosine similarity, nearest-neighbor retrieval,
clustering, dedup, a RAG retriever — inherits the ambiguity. The space has high
nominal **capacity** (say 768 or 1536 dimensions) but low **effective
resolution** in practice.

The tell is simple and measurable. Take many random pairs of items and look at
their cosine similarity. In a healthy space, unrelated pairs average ≈ 0 (nearly
orthogonal) and a genuinely *related* pair stands out sharply. In a crowded space,
*every* pair — related or not — scores high (e.g. 0.3–0.9), so the signal you care
about barely rises above the floor. The dynamic range cosine similarity can express
has been compressed. That compression is the subject of this document.

Two cautions frame everything below:

- **Cosine similarity is not automatically meaningful.** Steck, Ekanadham & Kallus
  ([2024](https://arxiv.org/abs/2403.05440)) show analytically that for *learned*
  embeddings the cosine similarity can be arbitrary — it depends on regularization
  and can even be non-unique — so a high or low number is only interpretable
  relative to the *distribution* of similarities in the space, not in absolute
  terms. This is why ambit always reads cosine against a reference distribution
  rather than against a fixed threshold.
- **A handful of dimensions can dominate the score.** Timkey & van Schijndel
  ([2021](https://arxiv.org/abs/2109.04404)) find that a few high-magnitude "rogue
  dimensions" drive most of the cosine similarity in transformer representations;
  standardizing them away often *restores* the resolution that raw cosine hides.

---

## 2. The vocabulary

### Isotropy ↔ anisotropy (the core axis)

- **Isotropy** (good): variance is spread evenly across directions; the cloud fills
  the space; random pairs are ~orthogonal.
- **Anisotropy** (bad / the "crowding" you observe): embeddings concentrate into a
  narrow **cone**, sharing a dominant direction and looking similar to each other
  regardless of meaning.

The origin of the observation traces to static word vectors: Mimno & Thompson
([2017](https://aclanthology.org/D17-1308/)) found skip-gram-with-negative-sampling
vectors occupy a narrow cone, and Ethayarajh
([2019](https://arxiv.org/abs/1909.00512)) operationalized **anisotropy** as the
*mean cosine similarity of random pairs*, showing contextual embeddings (BERT,
GPT-2, ELMo) are strikingly anisotropic in their upper layers.

A crucial nuance: anisotropy is often a *global* artifact masking *local*
structure. Cai, Huang, Bian & Church ([2021](https://openreview.net/forum?id=xYGNO86OWDH))
show the contextual space is approximately isotropic *within* clusters and along
low-dimensional manifolds even when it is globally a cone — so "crowded globally"
and "useful locally" can coexist, and good diagnostics must look at both scales.

### Alignment & uniformity (the contrastive-learning framing)

Wang & Isola ([2020](https://arxiv.org/abs/2005.10242)) decompose representation
quality into two properties on the unit hypersphere:

- **Alignment** — semantically similar items map close together.
- **Uniformity** — items spread evenly over the sphere, preserving maximal
  information.

Your "low resolution" is **low uniformity**. Good representations need *both*:
alignment without uniformity collapses everything together; uniformity without
alignment is noise. SimCSE (Gao, Yao & Chen,
[2021](https://arxiv.org/abs/2104.08821)) made these two the standard lens for
diagnosing sentence embeddings and showed contrastive training improves uniformity.

### The named pathologies

- **Representation degeneration / the "cone effect"** — Gao, He, Tan, Qin, Wang &
  Liu ([2019](https://arxiv.org/abs/1907.12009)). During likelihood training the
  (tied) output embeddings drift toward a shared direction and degenerate into a
  narrow cone. Biś, Podkorytov & Liu
  ([2021](https://aclanthology.org/2021.naacl-main.403/)) trace the mechanism to a
  *common gradient direction* that shifts all embeddings together, and show a
  simple mean-removal transform restores isotropy.
- **Inherent to attention** — Godey, de la Clergerie & Sagot
  ([2024](https://arxiv.org/abs/2401.12143)) argue anisotropy is not merely a
  cross-entropy-on-long-tails artifact but arises directly from the self-attention
  mechanism, appearing across objectives and modalities.
- **Dimensional collapse** — Jing, Vincent, LeCun & Tian
  ([2022](https://arxiv.org/abs/2110.09348)). The embeddings span only a
  *lower-dimensional subspace* than the model affords; the remaining axes carry
  almost no variance and are effectively wasted capacity.
- **Neural collapse** — Papyan, Han & Donoho
  ([2020](https://doi.org/10.1073/pnas.2015509117)). In the terminal phase of
  classification training, within-class variability collapses and class means
  arrange into a maximally-separated simplex (an equiangular tight frame). It is
  "good" collapse *between* classes but erases distinctions *within* a class.

### Hubness (the retrieval-side symptom)

Radovanović, Nanopoulos & Ivanović
([2010](https://www.jmlr.org/papers/v11/radovanovic10a.html)), *Hubs in Space*. A
curse-of-dimensionality effect: in crowded high-dimensional spaces a few points
become the nearest neighbor of disproportionately many queries ("hubs") while
others are nobody's neighbor ("anti-hubs"). Hubness directly degrades kNN retrieval
and is a practical fingerprint of poor resolution.

### A note on the word "crowding"

Your exact word is also a term of art — but a narrower one. Van der Maaten & Hinton
([2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html)) coined the **"crowding
problem"** specifically for **t-SNE**: the difficulty of placing moderate-distance
relationships when squeezing high-dimensional structure into 2-D. Same word,
different scope — it is about the *visualization*, not the native embedding space.
Keep them separate: anisotropy is a property of the embeddings; the t-SNE crowding
problem is an artifact of projecting them (and a reason ambit pairs every 2-D
projection with native-space diagnostics).

---

## 3. The geometry, in one picture

Write each embedding as a shared component plus an item-specific residual,
$x_i = \mu + r_i$. Cosine similarity between two items is

$$\cos(x_i, x_j) = \frac{(\mu + r_i)\cdot(\mu + r_j)}{\lVert \mu + r_i\rVert\,\lVert \mu + r_j\rVert}.$$

When the shared part $\mu$ (or a few dominant directions) is large relative to the
residuals, the numerator is dominated by $\mu\cdot\mu$ for *every* pair, so all
cosines inflate toward a common positive value — the cone. The information you want
lives in the residuals $r_i$, but it is swamped. This is exactly why three cheap
fixes work: **mean-centering** removes $\mu$; **all-but-the-top** (Mu, Bhat &
Viswanath, [2018](https://arxiv.org/abs/1702.01417)) removes the top few dominant
directions; and **standardizing rogue dimensions** (Timkey & van Schijndel, 2021)
rescales the few axes that dominate the dot product. Each one re-exposes the
residual structure that raw cosine had buried.

---

## 4. Why it happens

- **Likelihood / softmax pressure.** Training to maximize likelihood pushes token
  embeddings to grow along a shared direction to lower the partition function,
  producing the cone (Gao et al., 2019).
- **A common gradient shift.** The optimization moves all embeddings together along
  a dominant direction (Biś et al., 2021), which is what mean-removal undoes.
- **Frequency / rogue dimensions.** A few high-magnitude dimensions, correlated with
  token frequency, dominate variance and cosine (Timkey & van Schijndel, 2021).
- **The attention mechanism itself** contributes anisotropy independent of the loss
  (Godey et al., 2024).
- **Missing negative pressure.** Without an explicit uniformity / contrastive term,
  nothing forces embeddings to spread out (Wang & Isola, 2020).

---

## 5. How resolution is quantified

Different metrics capture different facets of the same thing. The ones below are
what ambit's *resolution / isotropy* facet draws on.

| Metric | What it captures | Healthy value | Source |
|---|---|---|---|
| **Mean random-pair cosine** | angular spread / anisotropy fingerprint | ≈ 0 | [Ethayarajh 2019](https://arxiv.org/abs/1909.00512) |
| **IsoScore** | "uniformity of embedding-space **utilization**" (your phrasing, literally) | → 1 | [Rudman et al. 2022](https://arxiv.org/abs/2108.07344) |
| **Partition-function isotropy** $I(\mathcal C)$ | directional uniformity | → 1 | [Mu, Bhat & Viswanath 2018](https://arxiv.org/abs/1702.01417); [Arora et al. 2016](https://aclanthology.org/Q16-1028/) |
| **Effective rank / participation ratio** | effective dimensionality (collapse) | → full dim | [Roy & Vetterli 2007](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf); [RankMe, Garrido et al. 2023](https://arxiv.org/abs/2210.02885) |
| **Spectral (eigenvalue) decay** | how variance distributes across axes | gentle, not steep | — |
| **Uniformity loss** | spread on the hypersphere | more negative | [Wang & Isola 2020](https://arxiv.org/abs/2005.10242) |
| **Coding rate / log-volume** | volume the features occupy | larger | [MCR², Yu et al. 2020](https://arxiv.org/abs/2006.08558) |
| **Silhouette / Davies–Bouldin / Fisher ratio** | separability *with labels* | higher / lower / higher | classical |
| **Hubness (k-occurrence skew)** | retrieval degradation | low skew | [Radovanović et al. 2010](https://www.jmlr.org/papers/v11/radovanovic10a.html) |

Useful formulas:

- **Anisotropy** (mean random-pair cosine):
  $\;A=\mathbb{E}_{i\neq j}\big[\cos(x_i,x_j)\big]$, isotropic ⇒ $A\approx 0$.
  For iid points on the unit $d$-sphere, random-pair cosine is ≈ $\mathcal N(0, 1/d)$
  — the reference ambit draws against.
- **Uniformity loss** (lower = more uniform):
  $\;\mathcal{L}_{\text{unif}}=\log\,\mathbb{E}_{x,y}\big[e^{-t\lVert x-y\rVert^2}\big]$.
- **Alignment loss**:
  $\;\mathcal{L}_{\text{align}}=\mathbb{E}_{(x,y)\sim\text{pos}}\big[\lVert f(x)-f(y)\rVert^2\big]$.
- **Effective rank** from singular values $\sigma_i$, with $p_i=\sigma_i/\sum_j\sigma_j$:
  $\;\operatorname{erank}=\exp\!\big(-\sum_i p_i\log p_i\big)$.
- **Participation ratio** from covariance eigenvalues $\lambda_i$:
  $\;\mathrm{PR}=\big(\sum_i\lambda_i\big)^2 / \sum_i\lambda_i^2$.
- **Partition-function isotropy**:
  $\;I(\mathcal C)=\dfrac{\min_{c\in\mathcal C}Z(c)}{\max_{c\in\mathcal C}Z(c)}$, where
  $Z(c)=\sum_i e^{c^\top x_i}$; isotropic ⇒ $I\approx 1$.

**Caveats.** (1) A single global anisotropy number can be misleading when the space
is locally isotropic (Cai et al., 2021) — pair it with cluster-conditioned
measures. (2) Standardize rogue dimensions *before* trusting cosine-based metrics
(Timkey & van Schijndel, 2021). (3) Treat absolute cosine values with suspicion;
compare against the isotropic reference distribution (Steck et al., 2024).

---

## 6. A worked reading

The numbers below are the diagnostic profile ambit's *resolution / isotropy* facet
reports for a representative, realistically-imperfect corpus (12,000 sentence
embeddings, 768-d) — useful as a template for how the metrics read *together*:

| Diagnostic | Value | Isotropic reference | Verdict |
|---|---|---|---|
| Mean random-pair cosine | **0.34** | ≈ 0.00 ± 0.036 | strongly anisotropic — a cone |
| Effective rank | **47 / 768** | → 768 | severe dimensional collapse |
| Top-1 PC variance | **37%** (90% in 31 dims) | evenly spread | a few axes dominate |
| IsoScore (utilization) | **0.27 / 1.0** | 1.0 | space badly under-used |
| NN cosine margin (median) | **0.041** | ≈ 0.11 | thin — top-1 barely beats top-2 |
| Hubness (top-1 k-occurrence) | **184** | ≈ 10 | a few hubs dominate retrieval |

Read as a whole: this space *looks* high-dimensional but lives on ~47 effective
axes, packs unrelated items at cosine ≈ 0.34, and resolves nearest neighbors by a
margin of only 0.04 — so retrieval is fragile and a handful of hubs absorb most
queries. The fix order suggested by §3 — mean-center, drop top directions / whiten,
re-measure — would move every one of these numbers toward its reference column. A
multilingual variant of this analysis (Rajaee & Pilehvar,
[2022](https://aclanthology.org/2022.findings-acl.103/)) finds the same pattern
across languages.

---

## 7. Why it matters

Low resolution silently caps the ceiling of everything built on the embeddings:

- **Retrieval / RAG** — a compressed similarity range means the right document is
  only marginally above the wrong ones; thresholds become unreliable and top-k gets
  noisy, while hubness over-retrieves a few documents.
- **Clustering / dedup** — crowded regions merge distinct entities; cluster
  boundaries blur.
- **Classification probes** — within-class collapse plus cross-class crowding
  reduces linear separability.
- **Drift / monitoring** — if the baseline is already anisotropic, there is little
  headroom to detect a real distributional shift.
- **Model selection** — effective rank is a *label-free* predictor of downstream
  quality (RankMe, Garrido et al., 2023), so resolution metrics can rank candidate
  encoders before you have task labels.

---

## 8. Remedies

- **All-but-the-top** (Mu, Bhat & Viswanath, 2018) — subtract the mean and remove
  the top few dominant directions.
- **BERT-flow** (Li et al., [2020](https://arxiv.org/abs/2011.05864)) and
  **whitening** (Su et al., [2021](https://arxiv.org/abs/2103.15316)) — map the
  embeddings to a more isotropic distribution as a post-process.
- **Cluster-based / local isotropy enhancement** (Rajaee & Pilehvar,
  [2021](https://aclanthology.org/2021.acl-short.73/)) — correct degeneration
  *within* clusters rather than globally.
- **Contrastive uniformity** (SimCSE, Gao et al., 2021) — train with a term that
  explicitly spreads embeddings out.
- **Spread-out regularization** (Zhang, Yu, Kumar & Chang,
  [2017](https://arxiv.org/abs/1708.06320)) and **maximal coding-rate** objectives
  (MCR², Yu et al., 2020) — encourage volume/diversity during training.

A pragmatic first step is often just **mean-centering + whitening** before computing
cosine similarity — it frequently recovers a large fraction of the lost resolution
at no training cost — but heed Steck et al. (2024): whether cosine is meaningful at
all is regularization-dependent, so always re-measure against a reference.

---

## 9. How this relates to ambit

ambit visualizes **how a dataset occupies an embedding space**. That occupancy *is*
isotropy / uniformity / utilization seen spatially — the same phenomenon the metrics
in §5 summarize as scalars. The mapping is direct:

| ambit facet | What it shows | The concept in this doc |
|---|---|---|
| **Density / hotspots** (VIZ 01–04) | where items pile up | crowded, **low-resolution** regions — the local face of anisotropy |
| **Coverage / voids** (VIZ 07–09) | what the data fills vs. leaves empty | **unused capacity** — the spatial face of dimensional collapse |
| **Topology / structure** (VIZ 05–06) | clusters, bridges, chokepoints | where distinctness is load-bearing vs. fragile; local-isotropy structure (Cai et al. 2021) |
| **Comparison / reference** (VIZ 10–11) | dataset vs. an ideal/reference distribution | the gap from **isotropic** |
| **Three dimensions** (3D · live, 3D 01–05) | the occupied *volume* and its anisotropy (e.g. thin-in-Z) | anisotropy you can rotate and see |
| **Resolution / isotropy** (ISO 01–05) | random-pair cosine histogram, eigenvalue scree & effective rank, IsoScore gauge, NN-margin, within-vs-between cosine | the scalar metrics of §5 and the worked reading of §6, made legible |

In short: the **density and coverage** views are the *picture* of crowding, the
**resolution / isotropy** diagnostics are its *measurement*, and the **3-D** views
let you see the anisotropy directly. A hotspot in the density view and a mass of
random-pair cosines piled up near 0.34 are the same fact told two ways — ambit's job
is to put both in front of you so "the embeddings look fine" can be replaced with
"here is exactly how much resolution this dataset has, and where it is spent."

---

## 10. References

**Foundational geometry & origin of the observation**
- Mimno, Thompson (2017). [*The strange geometry of skip-gram with negative sampling*](https://aclanthology.org/D17-1308/). EMNLP 2017.
- Arora, Li, Liang, Ma, Risteski (2016). [*A Latent Variable Model Approach to PMI-based Word Embeddings*](https://aclanthology.org/Q16-1028/). TACL 4 (arXiv:1502.03520).
- van der Maaten, Hinton (2008). [*Visualizing Data using t-SNE*](https://www.jmlr.org/papers/v9/vandermaaten08a.html). JMLR 9 — the t-SNE "crowding problem."
- Radovanović, Nanopoulos, Ivanović (2010). [*Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data*](https://www.jmlr.org/papers/v11/radovanovic10a.html). JMLR 11.

**Anisotropy in (contextual) embeddings**
- Ethayarajh (2019). [*How Contextual are Contextualized Word Representations?*](https://aclanthology.org/D19-1006/) EMNLP-IJCNLP 2019 (arXiv:1909.00512).
- Gao, He, Tan, Qin, Wang, Liu (2019). [*Representation Degeneration Problem in Training Natural Language Generation Models*](https://arxiv.org/abs/1907.12009). ICLR 2019.
- Biś, Podkorytov, Liu (2021). [*Too Much in Common: Shifting of Embeddings in Transformer Language Models and its Implications*](https://aclanthology.org/2021.naacl-main.403/). NAACL-HLT 2021.
- Cai, Huang, Bian, Church (2021). [*Isotropy in the Contextual Embedding Space: Clusters and Manifolds*](https://openreview.net/forum?id=xYGNO86OWDH). ICLR 2021.
- Timkey, van Schijndel (2021). [*All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality*](https://aclanthology.org/2021.emnlp-main.372/). EMNLP 2021 (arXiv:2109.04404).
- Godey, de la Clergerie, Sagot (2024). [*Anisotropy Is Inherent to Self-Attention in Transformers*](https://aclanthology.org/2024.eacl-long.3/). EACL 2024 (arXiv:2401.12143).
- Reif, Yuan, Wattenberg, Viégas, Coenen, Pearce, Kim (2019). [*Visualizing and Measuring the Geometry of BERT*](https://arxiv.org/abs/1906.02715). NeurIPS 2019.

**Collapse & the contrastive lens**
- Wang, Isola (2020). [*Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*](https://arxiv.org/abs/2005.10242). ICML 2020.
- Papyan, Han, Donoho (2020). [*Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training*](https://doi.org/10.1073/pnas.2015509117). PNAS 117(40).
- Jing, Vincent, LeCun, Tian (2022). [*Understanding Dimensional Collapse in Contrastive Self-Supervised Learning*](https://arxiv.org/abs/2110.09348). ICLR 2022.

**Metrics of resolution / utilization**
- Roy, Vetterli (2007). [*The Effective Rank: A Measure of Effective Dimensionality*](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf). EUSIPCO 2007.
- Rudman, Gillman, Rayne, Eickhoff (2022). [*IsoScore: Measuring the Uniformity of Embedding Space Utilization*](https://aclanthology.org/2022.findings-acl.262/). Findings of ACL 2022 (arXiv:2108.07344).
- Garrido, Balestriero, Najman, LeCun (2023). [*RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank*](https://arxiv.org/abs/2210.02885). ICML 2023.
- Yu, Chan, You, Song, Ma (2020). [*Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction (MCR²)*](https://arxiv.org/abs/2006.08558). NeurIPS 2020.

**Remedies**
- Mu, Bhat, Viswanath (2018). [*All-but-the-Top: Simple and Effective Postprocessing for Word Representations*](https://arxiv.org/abs/1702.01417). ICLR 2018.
- Li, Zhou, He, Wang, Yang, Li (2020). [*On the Sentence Embeddings from Pre-trained Language Models*](https://aclanthology.org/2020.emnlp-main.733/) (BERT-flow). EMNLP 2020 (arXiv:2011.05864).
- Su, Cao, Liu, Ou (2021). [*Whitening Sentence Representations for Better Semantics and Faster Retrieval*](https://arxiv.org/abs/2103.15316) (BERT-whitening).
- Gao, Yao, Chen (2021). [*SimCSE: Simple Contrastive Learning of Sentence Embeddings*](https://aclanthology.org/2021.emnlp-main.552/). EMNLP 2021 (arXiv:2104.08821).
- Rajaee, Pilehvar (2021). [*A Cluster-based Approach for Improving Isotropy in Contextual Embedding Space*](https://aclanthology.org/2021.acl-short.73/). ACL-IJCNLP 2021 (arXiv:2106.01183).
- Rajaee, Pilehvar (2022). [*An Isotropy Analysis in the Multilingual BERT Embedding Space*](https://aclanthology.org/2022.findings-acl.103/). Findings of ACL 2022 (arXiv:2110.04504).
- Zhang, Yu, Kumar, Chang (2017). [*Learning Spread-out Local Feature Descriptors*](https://arxiv.org/abs/1708.06320). ICCV 2017.

**On interpreting cosine itself**
- Steck, Ekanadham, Kallus (2024). [*Is Cosine-Similarity of Embeddings Really About Similarity?*](https://doi.org/10.1145/3589335.3651526) WWW '24 Companion (arXiv:2403.05440).
</content>
