---
layout: post
title: In the Perspective of Manifold Hypotheses - 2
date: 2025-12-15 00:00:00
tags: Tech-Blog
categories: English
---

## **Introduction: Architectural Evolution and the Path Traveled**

For a long time, we have anticipated a next-generation architecture sufficient to replace the Transformer or the existing Attention mechanism. Whether it is the Mamba series from the past two years, aiming to subvert the Transformer entirely, or the recently discussed Sparse/Gated Attention based on iterative optimization of existing mechanisms, research continues to advance, yet the goal remains elusive. Compared to the unpredictable future, perhaps the path we have traveled offers greater clarity.

## **I. From MLP to Transformer**

From the perspective of the Manifold Hypothesis, all classical architectures can be viewed as topological transformations from a curled manifold to a flat feature space, each carrying different Inductive Biases.

### **1. MLP: The Universal Manifold Operator with No Structural Priors**

The MLP treats the input space $$R^D$$ as a flat Euclidean space, utilizing hierarchical affine transformations and non-linear activation functions to indiscriminately fold, twist, and stretch the entire input space. According to the Universal Approximation Theorem, given sufficient neurons, an MLP can approximate any continuous function. Theoretically, this implies it can transform a manifold of any topological structure into a linearly separable form.

-   **The Worst Operator**: The MLP structure embeds almost no Inductive Bias. Its core assumption is "smoothness"—that similar inputs yield similar outputs. Mathematically, this requires the learned function to satisfy the Lipschitz continuity condition: $$∥f(x)-f(y)∥≤K∥x−y∥$$. However, in high-dimensional space $$R^D$$, without specific geometric priors, guaranteeing this smoothness requires a sample size $$N$$ that grows exponentially with dimension $$D$$. Furthermore, the metric based on Euclidean distance $$∥x−y∥_2$$ fails in high-dimensional space, as distances between all point pairs tend to converge. This **assumption of global spatial isotropy** ignores the complexity of high-dimensional manifold structures and is the root cause of its inefficiency. Consequently, massive use of MLPs often leads to suboptimal performance.

-   **The Best Operator**: Conversely, the near-absence of Inductive Bias means MLP has no requirements for input data distribution, possessing the capability to fold and project all manifolds. As a rigid transformation operator, MLP does not alter the data's topological structure. Therefore, in cross-modal feature fusion (direct alignment of embeddings) and most downstream heads (where the feature space is sufficiently flat), MLP becomes the optimal (and most universal) choice. This has led to minimalist approaches like LLaVA.

### **2. CNN: Introduction of Euclidean Group Symmetry**

The success of CNNs stems from explicitly exploiting the **Geometric Priors** of natural image manifolds:

-   **Local Connectivity**: Assumes the manifold possesses local topological structure; pixel correlation decays with distance.

-   **Weight Sharing**: Assumes the geometry of the tangent space is uniform across the manifold. The manifold $$M$$ is invariant under the translation group $$SE(2)$$, i.e., $$f(T(x))=f(x)$$ (invariance) or $$f(T(x))=T′(f(x))$$ (equivariance).

As a spatially local operator, CNN explicitly hard-codes the **symmetry** of the manifold into the network structure. It performs "low-pass filtering" in the local neighborhood of the input manifold, attenuating high-frequency noise and unstable directions, thereby making the representation closer to the low-dimensional structure. This can be approximated as CNN performing multiple equidistant foldings and distortions on the entire manifold simultaneously.

However, CNNs retain the Euclidean geometric metric. When the data manifold has non-zero curvature (e.g., spherical projection data, non-Euclidean graph data), the translation invariance assumption fails. CNNs cannot effectively capture long-range dependencies because, on a curved manifold, the "straight line" shortest path becomes a geodesic, which fixed-size convolution kernels cannot cover.

### **3. RNN: Dynamical System Trajectories on State Manifolds**

RNN models the data manifold as a Dynamical System evolving over time:

$$h_t=σ(W_h h_{t−1} + W_x x_t)$$

This is essentially the discretized Euler integration of the ordinary differential equation $$\frac{dh}{dt}=f(h,x)$$. The RNN is a temporal recursive operator attempting to learn the vector field of the manifold's tangent space. Each hidden state $$h_t$$ is a coordinate point on the manifold, and the weight matrix $$W$$ defines the Flow on the manifold. The learning objective of RNN is not to memorize context, but to learn a hidden state manifold $$S$$ and its evolutionary laws, such that task-relevant variables form low-dimensional, predictable trajectories on $$S$$, while irrelevant perturbations are compressed into the normal direction and gradually attenuated.

The primary issue with RNN is the use of the same weight matrix $$W$$ at every step. This enforces an assumption that the manifold is **flat**, i.e., the **tangent space is identical at every location**. However, real semantic manifolds often possess complex curvature. When sequences are long, the actual geometric transformation (the product of Jacobian matrices) leads to exponential explosion or decay of eigenvalues and gradients due to curvature accumulation. RNN attempts to approximate a continuously changing tangent space transformation with a fixed linear operator, which is mathematically ill-posed.

**Alternative Perspective**:

If time $$t$$is viewed as an independent dimension, RNN treats the manifold as **one or more static curves** in a high-dimensional Euclidean space $$R^{D+1}$$defined by parameter $$t$$. The hidden state $$h_t$$ effectively encodes the tangent space and historical trajectory information at a point $$(x_t,t)$$ on the manifold. The recursive formula above is geometrically equivalent to path integration along a curve on the manifold surface. RNN attempts to define a vector field that, by advancing via tangent vectors along the $$t$$ axis, progressively delineates the manifold's shape.

In this static space, RNN forcibly assumes the manifold is **simply connected** and **sequentially dependent**. It must traverse the manifold strictly along the gradient direction of $$t$$. If the manifold is curled in high-dimensional space such that $$t_i$$ and $$t_{i+k}$$ are extremely close in Euclidean distance but far apart in geodesic distance, RNN must traverse the entire lengthy geodesic to establish a connection. Moreover, relying on the continuous accumulation of local linear approximations of tangent vectors means that once a tangent space estimation deviates at any point (gradient vanishing/exploding), this geometric distortion is amplified exponentially along the path, leading to a collapse in the cognition of the manifold's global topology. In summary, RNN forcibly reduces a static geometric structure to a one-dimensional path problem, discarding the non-local geometric properties of the manifold in high-dimensional space.

### **4. Mamba (SSM): Optimal Control on Continuous Manifolds**

Mamba (and the underlying S4/S6 theory) is a geometric correction to RNN. It retains the Dynamical System perspective but introduces **HiPPO Matrix Theory and Selective Scan**.

$$h′(t)=Ah(t)+Bx(t)$$

$$y(t)=Ch(t)$$

The special construction of the HiPPO matrix $$A$$ ensures that the state $$h_t$$ is the optimal projection of all past input manifold history onto an orthogonal polynomial basis. It solves the RNN's "forgetting" problem. Simultaneously, Mamba introduces $$B(x)$$, $$C(x)$$, and $$Δ(x)$$, making the flow field on the manifold a function of input $$x$$. This extends the RNN from a Linear Time-Invariant (LTI) system to a **Linear Time-Variant (LTV)** system.

However, even with these optimizations, the system's information compression remains lossy. Disregarding the advantage of linear complexity, its effectiveness in discrete graph matching tasks is often inferior to the Transformer, and its handling of non-causal data is less intuitive than Attention.

### **5. Transformer: Adaptive Graph Structure Based on Dynamic Metrics**

Initially, the Transformer was viewed as a sequence model, but ignoring the temporal information injected by $$PE$$, it is essentially a set-based dynamic Graph Neural Network. The Transformer treats the "data manifold" as a Complete Graph, where the model learns the edge weights itself. It is no longer constrained by neighborhood definitions in Euclidean space, enabling it to handle non-grid data.

**Self-Attention: Data-Dependent Riemannian Metric**

In manifold learning, the core challenge is defining the distance between two points on the manifold. CNN assumes a fixed Euclidean distance (pixel adjacency implies correlation), and RNN assumes temporal distance. The Transformer discards these fixed metrics via the self-attention mechanism, learning a **Data-Dependent Metric Tensor**. The attention metrics essentially construct a **dynamic adjacency matrix** using the inner product as a kernel function (Riemannian metric). Unlike the isotropy of Euclidean distance, Attention is highly anisotropic. It dynamically adjusts the direction of the tangent space based on Context, allowing the model to ignore Tokens that are distant in sequence position but adjacent on the semantic manifold.

Compared to CNN's local expansion and RNN's path integration, the Transformer can establish "wormholes" on the manifold via the Attention mechanism, directly connecting two points with extreme geodesic distances. This eliminates noise caused by curvature accumulation, allowing gradients to propagate losslessly across the manifold.

**Multi-Head Mechanism: Multiple Sub-Manifold Projections**

From a manifold geometry perspective, a high-dimensional semantic manifold $$M$$ is often the Cartesian product of multiple Sub-manifolds:$$M \approx M_{syntax} \times M_{semantic} \times M_{tone}...$$. For example, one Head might capture the sub-manifold of syntactic structure (subject-verb-object relations), while another captures the sub-manifold of coreference resolution (pronouns and their referents). Multi-Head Attention allows the model to compute geometric relations in different tangent subspaces in parallel, finally recovering the complete manifold structure via linear projection and concatenation.

**In summary, the Transformer's prior actually weakens existing biases. It pays a computational cost of $$O(N^2)$$ in exchange for the ability to capture manifolds of arbitrary topological structure.** This is why it requires massive data; it must learn the manifold's topology from scratch, unlike CNNs or RNNs which possess inherent locality priors.

## **II. Scaling Law and Emergence**

Architecturally, LLMs (like GPT-4, LLaMA) differ little from the original Transformer; the primary difference is scale. Under the Manifold Hypothesis, the shift from small to large models is not merely quantitative accumulation but a qualitative change in manifold topology, coverage density, and connectivity.

### **1. Scaling Law**

In deep learning, Loss essentially measures the distance between the manifold learned by the model and the true data manifold. The Scaling Law describes the power-law decrease of Loss with respect to compute $$C$$, parameters $$N$$, and data $$D$$: $$L(N) \propto N^{-\alpha}$$.

-   **Increasing Parameters $$N$$ (Reducing Bias):** The Scaling Law suggests that as $$N$$ increases, the model's ability to fit high-frequency curvature rises according to a power law. Small models can only learn the global skeleton of the manifold, i.e., the principal components. At this stage, Loss drops rapidly. As $$N$$ increases, the model begins to wrap around high-frequency regions with extreme curvature on the manifold—the rare, complex long-tail samples.

-   **Increasing Data $$D$$(Reducing Variance):** According to coverage number theory, covering a $$d$$-dimensional manifold with precision $$\epsilon$$ requires sample size$$M \propto (1/\epsilon)^d$$. The Scaling Law effectively reveals the decay rate of approximation error as sample density increases for a specific $$d$$. This explains why image generation (high $$d$$) is harder to scale than text classification (low $$d$$). The existence of the Scaling Law proves that deep networks are indeed performing manifold learning, not simple memorization. If it were memorization, the Loss curve would not exhibit this power-law distribution.

From the Foundation Model perspective, the training of an LLM is the ultimate approximation of the language manifold.

### **2. Emergence: Phase Transition on the Manifold**

If we view data distribution as a manifold, learning is the process of establishing connectivity upon it.

-   **Small Models**: The model learns dispersed local neighborhoods on the manifold but fails to establish correct mappings between them. The model cannot perform multi-step reasoning because the inference path is broken.

-   **Critical Point**: When parameter count $$N$$ and training data $$D$$ exceed a certain threshold, the model's coverage density on the manifold reaches the percolation threshold. Dispersed local knowledge suddenly connects into a globally consistent graph.

-   **Emergence**: At this point, the model can not only interpolate but also perform transitive composition on the manifold. For instance, knowing A→B and B→C leads to the emergence of A→C capability. Macroscopically, this manifests as a sudden jump in performance (similar to phase transitions in complex physical systems).

Emergence is often accompanied by geometric reconstruction (linearization and disentanglement) of the representation space. Before the phase transition, different concepts are entangled and twisted on the manifold, inseparable by linear layers. After the transition, the model learns to unfold the curved manifold into a high-dimensional Euclidean space, making complex semantic relations linearly separable. Acquiring this unfolding capability often requires achieving certain depth and width—precisely the moment emergence occurs.

**Why can't other architectures (CNN/RNN) achieve the same degree of emergence?**

The root cause remains **Inductive Bias.**

RNN forces all historical information into a fixed-dimension state vector $$h_t$$. For complex manifold trajectories, this equates to projecting a high-dimensional manifold into a low-dimensional space, inevitably leading to **information loss and Singularities**. As sequences lengthen and trajectories diverge, the model cannot maintain the global geometric structure of the manifold with finite memory, thus failing to emerge long-range reasoning capabilities. While CNN is efficient, its receptive field grows linearly. Covering two distant associated points on the manifold requires stacking extremely deep networks, leading to optimization difficulties. More importantly, CNN's weight sharing assumes identical geometric properties across the manifold, limiting its ability to handle non-stationary semantic manifolds.

Furthermore, the Transformer's Attention is the product of input and weights, or even the quadratic form of the input itself ($Q K^T$). This makes it essentially a second-order (or higher) network, whereas other architectures are mostly first-order accumulations. High-order interactions allow the Transformer to dynamically adjust computation weights based on context. This in-context learning capability is difficult for fixed-weight RNN/CNNs to possess.

## **III. Efficient Modeling of Manifolds**

Even by weakening inductive bias, the Transformer gains context-adaptive modeling capabilities. However, this adaptation currently manifests as adjustments for input/output tokens within a unified data manifold, rather than model-level adaptation for different data distributions/manifolds. If we had infinite labor, data, and compute, we could train a specific small-scale Transformer for every sub-task instead of brute-force fitting a billion-parameter large model. Many current architectural optimizations for Transformers are implicitly doing this. For example, MoE partitions regions in FFN memory, while Sparse/Gated Attention establishes highly flexible and sparse connections between coordinates on the manifold.

### **1. MoE**

The essence of MoE is acknowledging that using a single, globally shared Dense Model to approximate a high-dimensional manifold with extremely complex topological structure and drastic curvature changes is mathematically inefficient and unstable. If the real data distribution consists of multiple sub-manifolds, a Dense Model learns a single operator under a global unified coordinate system (analogous to a second-order MLP); an MoE model learns multiple local operators (covering different sub-manifold regions) plus a router (learning which region an input token belongs to and which local operators to invoke).

**Why is MoE suitable for the Long Tail and Multi-Domain aspect of Foundation Models?**

**Long-tail samples** often fall into regions rarely visited by the Dense Model, equivalent to sparse branches or small fragments of the sub-manifold. Simultaneously, a Dense Model with a fixed compute budget struggles to maintain high-resolution approximation across all regions. MoE allocates resolution on demand: letting specific experts handle specific regions, thereby improving coverage of multi-domain structures under the same token compute budget.

### **2. Sparse Attention**

Traditional Attention builds a graph and performs diffusion/kernel smoothing on the token manifold. If token representations lie on a low-dimensional manifold, the most effective information often comes from manifold neighbors or a few cross-region shortcuts. Fully connected attention introduces massive non-manifold noise connections (long-distance, semantically unrelated token interactions); thus, effective mixing does not require full connectivity. The core of Sparse Attention is approximating the full connection in traditional Attention as a sparse graph:

-   **Top-k / kNN Sparsification**: Retain only the $$k$$ edges with maximum similarity for each query (manifold neighbors).

-   **Block Sparse / Local Window**: Prior assumption that neighbors are also local in sequence position (suitable for local dependencies).

-   **Budget Adaptive Sparsity**: Different sparsity levels for different tokens and different stages (prefill vs decode).

### **3. Gated Attention**

The recently emerging Gated Attention goes beyond fine-tuning the attention matrix (which often introduces new inductive biases) to directly adaptively controlling the projected vector field. The Gated mechanism introduces a gating signal $$g \in [0,1]^d$$ to filter the update vector dimension-wise. Geometrically, this is orthogonal decomposition and suppression of tangent vectors. The Gate identifies and suppresses noise components perpendicular to the current task sub-manifold, retaining only effective components along the geodesic direction. Furthermore, the gating mechanism is most effective when applied to the Q vector, stemming from the rank elevation of the projection matrix brought by sparsity.

When processing long sequences, Gated Attention dynamically adjusts the decay rate of historical information and avoids Attention Sinks. This is equivalent to dynamically adjusting inertia based on the manifold's local curvature. In flat regions of the manifold (semantic coherence), the Gate allows long-distance information transmission; in regions with curvature mutations (semantic shifts), the Gate rapidly truncates historical dependencies and resets the trajectory direction.

These architectural optimizations transcend stabilizing the training process or enhancing feature representation itself. Deeper reasoning often corresponds to **more abstract embeddings and increased information density of single embeddings**. MoE and Sparse/Gated Attention provide the structural foundation for such high signal-to-noise ratio representations. Supported by training data (higher quality) and training methods (deeper RL), the chain-of-thought/graph built by such sparse routing-like mechanisms will demonstrate greater potential. Understood at this level, **the sparsity of thought reflects the model's level of intelligence to some extent**.

## **IV. True and False Reasoning**

### **1. In-Context Learning (ICL)**

Current ICL in LLMs is essentially a mathematical abstraction of dynamic localization and interpolation of high-dimensional data within a low-dimensional topological structure. ICL is not "learning" new knowledge but retrieving, locating, and locally linearizing the low-dimensional manifold structures constructed during pre-training.

In the inference phase, the Transformer uses examples in the Prompt (acting geometrically as anchors on the manifold) to dynamically locate a specific task sub-manifold in the latent space. The Attention mechanism effectively calculates the projection of the input Query onto the tangent space of this sub-manifold. The model merely locks onto a local coordinate system within the vast, learned manifold structure via the Prompt.

The forward inference process of the Transformer is mathematically equivalent to gradient descent updates on a loss function defined by context examples, performed in parameter space or activation space. Given context, the model does not update physical weights $$W$$ but simulates the process of finding the optimal solution on $$M_{task}$$ by changing internal activation states. This makes ICL appear as local fine-tuning within specific regions of the manifold.

However, ICL cannot perform true reasoning or create new knowledge. It can only interpolate within the convex hull or neighborhood of the pre-training manifold. If the logic required for the task is completely orthogonal to the pre-training manifold (i.e., Out-of-Distribution), ICL is bound to fail or hallucinate. Furthermore, as context length increases, conflicting or noisy examples can cause unstable localization of activation points on the manifold, or even collapse into incorrect sub-manifold regions.

### **2. CoT: Path Planning with Explicit Symbols**

From the rigorous perspective of the Manifold Hypothesis, Chain of Thought is not the emergence of reasoning, but a geometric strategy to reduce mapping curvature and perform geodesic interpolation.

In existing LLMs, for simple tasks, input $$x$$and output$$y$$ are close on the manifold, or the manifold is locally flat. In this case, single-step reasoning $$y=f(x)$$ is effective because it can be approximated by simple linear interpolation or shallow non-linear transformation. However, for complex tasks, although $$x$$ and $$y$$ are both on the manifold, the geodesic distance between them is immense, and the manifold structure is highly curled, non-convex, or even contains topological holes. If we force the model to directly predict $$P(y \mid x)$$, the model attempts to walk a straight line connecting $$x$$ and $$y$$ in the ambient space. But this straight line often cuts through OOD regions outside the manifold, leading to hallucinations or logical breaks.

CoT forces the model to generate intermediate steps $$z_1, z_2, ..., z_n$$ between $$x$$ and $$y$$. This effectively decomposes an extremely difficult global mapping problem into a series of local mapping problems with low Lipschitz constants, explicitly eliminating uncertainty. The token sequence generated by CoT is a discretized geodesic path on the manifold surface.

### **3. Latent CoT: Returning to Continuous Manifold Dynamics**

However, human language is a low-dimensional, quantized projection of the thought manifold. Forcing a model to output natural language for reasoning is equivalent to forcing continuous neural network signals through a low-bandwidth Argmax discretization layer. Simultaneously, many intuitive, fuzzy intermediate states cannot be precisely captured by discrete vocabulary. **This quantization noise accumulates in long-chain reasoning, leading to reasoning drift.**

Latent CoT attempts to remove the constraint of discrete Tokens, evolving trajectories directly in Latent Space. In this case, within Latent Space, the model can maintain a superposition state of being both A and B until the final moment of inference, when it collapses into a discrete output. This avoids error propagation caused by premature discretization decisions. Additionally, a single Token can only transmit $$log_2∥V∥$$ bits of information (where $$∥V∥$$ is vocabulary size), whereas a $$d$$-dimensional FP16 vector can theoretically transmit far more information. Latent CoT exploits this high-dimensional broadband channel to transmit complex reasoning states.

This internal thinking essentially increases the depth of the computational graph. From a manifold perspective, this involves progressively stretching and untangling the originally entangled manifold through multiple composite function transformations until it becomes linearly separable.

For Latent CoT, since there is no explicit Ground Truth text, training methods are shifting from SFT to RL. The objective function is no longer predicting the next Token but maximizing the accuracy of the final answer. The model is encouraged to freely explore paths in the latent space. Research finds that the Latent Thoughts learned by the model often exhibit feature distributions incomprehensible to humans—precisely proving that it has broken through the low-dimensional manifold limits of human language and found superior Shortcuts.

**But is reasoning based on Chain of Thought general reasoning?**

The reasoning demonstrated by current LLMs is mathematically primarily **local interpolation** on high-dimensional manifolds. The essence of the Scaling Law is that as parameters increase, the sampling density of the manifold rises exponentially, causing the probability of a test sample falling into the $$\epsilon$$-neighborhood of a training sample to approach 1. Human thinking, however, possesses the capability of extrapolation off the manifold. Humans can establish new logical connections in the holes of the data manifold or in directions completely orthogonal to the manifold.

Additionally, human language is a **low-dimensional, lossy, highly compressed projection** of the physical world. Humans rarely expand their understanding of the real world solely using language; even theorems and conjectures derived from mathematics/physics require experimental verification in the real world to be widely accepted. LLMs attempt to reconstruct $$M_{world}$$ via $$M_{language}$$, i.e., building a world model. But since this projection is not bijective, reverse-engineering the world from language has infinite solutions. Therefore, interaction with the environment remains indispensable.

Finally, existing large models, even if adaptive, remain static. When data distribution (manifold structure) changes, without retraining or fine-tuning, they will still hit the OOD barrier. Catastrophic forgetting caused by retraining and fine-tuning remains a hurdle difficult to overcome (hence the recent surge in continual learning). Without dynamics, humanity is destined to burn astronomical amounts of capital and electricity in the repeated training of models.

## **Epilogue**

I have written too much to ramble further.

In short, amidst the waves of chasing hotspots, do not forget that gold often already exists in the shadows of the path traveled.