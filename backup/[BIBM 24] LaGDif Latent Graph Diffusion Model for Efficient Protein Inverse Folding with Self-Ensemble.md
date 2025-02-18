## LaGDif: 用于高效蛋白质反向折叠的潜在图扩散模型与自集成方法

**论文报告**

### 1. 背景介绍

蛋白质反向折叠 (Protein Inverse Folding) 是计算生物学中的一个核心挑战，其目标是确定能够折叠成给定蛋白质结构的可行的氨基酸序列。这项技术在药物发现、酶工程和生物材料开发等领域具有巨大的应用潜力。理解几个关键术语至关重要：

*   **蛋白质反向折叠 (Protein Inverse Folding):**  也称为蛋白质序列设计，是从给定的蛋白质三维结构反向预测可能的氨基酸序列的过程。这与蛋白质结构预测相反，后者是从氨基酸序列预测三维结构。
*   **氨基酸序列 (Amino Acid Sequence):**  蛋白质的基本组成单元是氨基酸，氨基酸序列是指蛋白质中氨基酸的线性排列顺序。自然界中常见的氨基酸有20种。
*   **蛋白质结构 (Protein Structure):**  蛋白质的三维空间构象。蛋白质结构通常分为四个层次：一级结构（氨基酸序列）、二级结构（局部结构如α螺旋和β折叠）、三级结构（完整的多肽链的三维结构）和四级结构（多个多肽链的组装结构）。
*   **扩散概率模型 (Diffusion Probabilistic Models):**  一种生成模型，灵感来源于非平衡热力学。扩散模型通过逐步向数据中添加噪声（前向扩散过程），然后再学习逆向去除噪声的过程（反向扩散过程），从而实现数据生成。近年来，扩散模型在图像生成、音频合成和分子生成等领域取得了显著的成功。

传统的蛋白质反向折叠方法主要基于能量函数，依赖于物理学原理和经验规则，但往往难以生成多样化的序列。近年来，基于蛋白质语言模型的深度学习方法开始兴起，将反向折叠任务视为类似于自然语言处理的序列生成问题。这些模型通常分为掩码模型和自回归模型，但它们在捕捉蛋白质结构与序列之间复杂的一对多关系方面存在局限性，并且确定性较强，难以充分探索序列空间的多样性。

扩散概率模型作为一种新兴方法，为蛋白质序列生成提供了新的思路。扩散模型通过从噪声到结构化数据的逆向过程生成数据，天然地适合处理不确定性和探索复杂的多模态输出空间。然而，现有的蛋白质反向折叠扩散模型大多在离散数据空间中操作，需要预先定义转移矩阵，限制了连续空间中固有的平滑过渡和梯度，可能导致次优性能。

本文提出的 LaGDif (Latent Graph Diffusion Model for Protein Inverse Folding) 模型，旨在克服现有离散扩散模型的局限性，并充分利用连续扩散模型的优势。

### 2. 核心概述

本文提出了一种用于蛋白质反向折叠的**潜在图扩散模型 LaGDif**。该模型通过**编码器-解码器架构**桥接了离散和连续空间，将蛋白质图数据分布转换为连续潜在空间中的随机噪声。然后，LaGDif 模型通过考虑每个节点的空间配置、生化属性和环境因素，重构蛋白质序列。此外，作者还提出了一种新颖的**反向折叠自集成方法**，通过聚合多个去噪后的蛋白质序列输出来稳定预测结果并进一步提高性能。在 CATH 数据集上的实验结果表明，LaGDif 模型优于现有的最先进技术，在单链蛋白质的序列恢复率方面提高了高达 45.55%，并保持了生成结构与天然结构之间平均 RMSD 为 1.96 Å。LaGDif 在蛋白质反向折叠方面的进步，有望加速治疗和工业应用新型蛋白质的开发。

### 3. 方法与实验细节

#### 3.1 数据集

本文使用了 **CATH 数据集 version 4.2.0**。这是一个广泛使用的蛋白质结构分类数据库。数据集按照之前的工作 [10] 进行了划分：

*   **训练集:** 18,024 个样本
*   **验证集:** 608 个样本
*   **测试集:** 1,120 个样本

为了评估模型处理不同复杂程度蛋白质的能力，测试集进一步分为三个子集：

*   **短蛋白质子集 (Short):**  链长小于 100 个氨基酸的蛋白质。
*   **单链蛋白质子集 (Single-chain):**  仅包含单条多肽链的蛋白质。
*   **所有蛋白质子集 (All):**  测试集中的所有蛋白质，包括多链和更长的蛋白质结构。

数据集的统计信息如图 3 所示，展示了节点（氨基酸）和边（空间连接）的分布情况，体现了数据集结构的多样性。

#### 3.2 算法和模型

LaGDif 模型的核心架构是一个**编码器-解码器的潜在图扩散模型**，并使用了 **自集成 (Self-Ensemble)** 方法。

**模型架构:**

*   **编码器 (Encoder):** 基于 **ESM2 (Evolutionary Scale Model 2)** 预训练的蛋白质语言模型。ESM2 能够捕捉氨基酸之间的关系和蛋白质结构信息，将离散的氨基酸序列编码到高维连续潜在空间中。
*   **解码器 (Decoder):**  一个可学习的线性层，将高维连续潜在表示映射回离散的氨基酸序列空间。
*   **扩散模型 (Diffusion Model):**  在潜在空间中操作的扩散模型。前向扩散过程逐步向蛋白质图的潜在表示添加高斯噪声，反向扩散过程学习从噪声中恢复原始的蛋白质序列。
*   **去噪网络 (Denoising Network):**  使用 **等变图神经网络 (Equivariant Graph Neural Network, EGNN)**。EGNN 能够有效捕捉和利用蛋白质三维结构的**空间等变性 (Spatial Equivariance)**，即蛋白质分子的性质应保持一致，不受其空间位置或方向变化的影响。EGNN 通过一系列**等变图卷积层 (Equivariant Graph Convolution Layers, EGCL)** 处理蛋白质图。

**EGCL 的更新过程 (公式 2 和 3):**

每个 EGCL 层接收节点隐藏状态 $H^l = \{h_1, h_2, ..., h_m\}$，互连节点 i 和 j 的边嵌入 $m_{ij}$，以及节点的空间坐标 $X_{pos} = \{x_{1}^{pos}, x_{2}^{pos}, ..., x_{m}^{pos}\}$ 作为输入，并更新节点状态 $H^{l+1}$ 和节点位置 $X_{pos}^{l+1}$。 具体更新公式如下：

$H^{l+1}, X_{pos}^{l+1} = EGCL(H^l, X_{pos}, M)$  (公式 2)

其中，在每个 EGCL 内部，更新过程定义如下：

$m_{ij} = \phi_e (h_i^l, h_j^l, ||x_i^l - x_j^l||^2, m_{ij})$

$x_i^{l+1} = x_i^l + \frac{1}{n_i} \sum_{j \neq i} (x_i^l - x_j^l) \phi_x (m_{ij})$

$h_i^{l+1} = \phi_h (h_i^l, \sum_{j \neq i} h_j^l m_{ij})$   (公式 3)

*   $\phi_e$, $\phi_x$, 和 $\phi_h$ 是可学习的神经网络函数。
*   $\frac{1}{n_i}$ 项用于归一化坐标更新，以确保稳定性。
*   $m_{ij}$ 基于节点特征、节点间距离和先前的边特征进行更新。
*   $x_i$ 使用相对位置和边特征的可学习函数进行更新。
*   $h_i$ 基于先前的值和聚合的边信息进行更新。

**先验知识的融入:**

模型还利用了蛋白质的**二级结构信息 (Secondary Structure Information)**。通过 **DSSP (Definition of Secondary Structure of Proteins) 方法 [17]** 分析蛋白质的三维结构，将每个氨基酸分类为八种不同的二级结构类型。这些信息通过 one-hot 编码和嵌入层处理后，与扩散模型的其他输入特征集成。

**引导噪声控制和自集成 (Guided Noise Control and Self-Ensemble):**

*   **引导采样 (Guided Sampling) (公式 4):**  为了平衡序列多样性和结构完整性，模型采用了引导采样方法。引导采样过程定义为：

    $\hat{x} = \alpha x + (1 - \alpha) \epsilon$   (公式 4)

    其中，$\hat{x}$ 是初始噪声节点特征，$x$ 是模型输入的蛋白质特征，$\alpha \in [0, 1]$ 是可控参数，$\epsilon \sim \mathcal{N}(0, I)$ 是高斯噪声。 通过调整 $\alpha$，可以控制噪声的强度和方向，从而实现更有效和稳定的去噪过程。

*   **自集成 (Self-Ensemble) (公式 5):** 为了提高预测的鲁棒性和准确性，模型在采样过程中使用了自集成方法。在每个去噪步骤 $t$，模型生成 $K$ 个候选图，并对它们的节点特征进行平均：

    $X_t = \frac{1}{K} \sum_{k=1}^{K} f_\theta (x_{t+1}^k, E, t)$   (公式 5)

    其中，$f_\theta$ 是参数为 $\theta$ 的 EGNN 去噪网络，$x_{t+1}^k$ 是步骤 $t+1$ 的第 $k$ 个采样图节点特征。本文实验中 $K$ 设置为 5。

#### 3.3 训练和评估过程

*   **训练细节:** LaGDif 使用 4 层 EGNN 作为去噪网络，扩散时间步长 $T$ 设置为 1000。模型训练了 20 个 epochs。在采样阶段，在 980 个时间步长引入引导噪声。使用了自集成方法，集成 5 个独立的预测结果。模型实现基于 PyTorch 和 PyTorch-Geometric，在 NVIDIA® 4090 GPU 上运行。

*   **评估指标:**
    *   **困惑度 (Perplexity):**  衡量模型预测氨基酸序列不确定性的指标，值越低表示预测置信度越高，准确性越高。
    *   **序列恢复率 (Recovery Rate):**  直接衡量模型从给定三维结构重建正确氨基酸序列的能力。
    *   **TM-score (Template Modeling score):**  评估预测结构与目标结构拓扑相似性的指标，取值范围为 0 到 1，值越高表示结构相似性越高。
    *   **平均 pLDDT (predicted Local Distance Difference Test):**  评估局部结构准确性的指标，值越高表示局部结构预测越准确。
    *   **平均 RMSD (Root Mean Square Deviation):**  衡量预测结构和目标结构对应原子之间平均距离的指标，值越低表示结构差异越小。

### 4. 研究过程与结论

本文的研究过程围绕 LaGDif 模型的提出、实验验证和性能分析展开。

**研究过程:**

1.  **模型设计:**  提出了 LaGDif 模型，结合了潜在空间扩散模型、EGNN 去噪网络、ESM2 编码器、二级结构信息以及引导采样和自集成方法。
2.  **实验设置:**  在 CATH 数据集上进行了蛋白质反向折叠实验，并将测试集分为短蛋白质、单链蛋白质和所有蛋白质三个子集。
3.  **性能评估:**  将 LaGDif 模型与现有的最先进的反向折叠模型（如 StructGNN, GraphTrans, GCA, GVP, AlphaDesign, ESM-IF1, ProteinMPNN, PIFold, Grade-IF）在序列恢复率、困惑度、TM-score, pLDDT 和 RMSD 等指标上进行了比较。
4.  **消融研究:**  进行了消融实验，评估了自集成方法中集成样本数量 $K$ 和引导噪声对模型性能的影响。
5.  **案例研究:**  对两个具体的蛋白质案例 (2EBO 和 3OUS) 进行了深入分析，可视化了预测结构，并对比了 LaGDif 与其他模型的性能。
6.  **模型复杂度分析:**  比较了 LaGDif 与其他模型的参数量、推理时间和内存使用情况。

**实验结果与结论:**

*   **性能超越现有方法:**  在 CATH 数据集上，LaGDif 模型在所有蛋白质类别上都显著优于现有最先进的反向折叠方法 (Table I)。例如，在短链蛋白质上，序列恢复率从 Grade-IF 的 45.27% 提高到 LaGDif 的 86.97%，提高了 41.7%。单链蛋白质和所有蛋白质类别上，LaGDif 也取得了类似的显著提升。LaGDif 的困惑度也显著低于其他模型，表明其预测具有更高的置信度。
*   **结构质量保持:**  LaGDif 在保持生成蛋白质结构完整性方面表现出色 (Table II)。其平均 TM-score 为 0.82，接近 ProteinMPNN (0.84)，远高于其他模型。平均 RMSD 为 1.96 Å，仅次于 ProteinMPNN (1.76 Å)。对于高分辨率晶体结构，RMSD 小于 2 Å 被认为是高度接近，表明 LaGDif 生成的结构与天然结构高度相似。
*   **自集成有效性:**  消融研究表明，自集成方法显著提高了模型性能 (Table III, Figure 4)。随着集成样本数量 $K$ 的增加，序列恢复率持续提升，并在 $K=5$ 之后趋于稳定。所有自集成配置都优于基线模型，验证了自集成方法的鲁棒性。
*   **案例研究突出优势:**  案例研究 (Figure 5) 表明，对于不同复杂程度的蛋白质 (2EBO 和 3OUS)，LaGDif 始终优于其他模型，在序列恢复率、TM-score 和 RMSD 等指标上都取得了最佳结果，突显了 LaGDif 的鲁棒性和通用性。
*   **模型复杂度适中:**  模型复杂度分析 (Table IV) 表明，LaGDif 在模型参数量、推理时间和内存使用方面取得了良好的平衡。其参数量适中，能够捕捉复杂蛋白质结构，同时避免了过高的计算开销。推理时间虽然略长于 ProteinMPNN 和 Grade-IF，但远快于 GVP，且性能优势显著。

**总体结论:** LaGDif 模型通过潜在空间扩散、EGNN 去噪、ESM2 编码和自集成等技术，有效提升了蛋白质反向折叠的性能，在序列恢复率和结构质量方面都取得了显著的进步，超越了现有最先进的方法。

### 5. 总结与客观评价

本文提出的 LaGDif 模型是一种新颖的蛋白质反向折叠方法，它巧妙地利用了连续空间扩散模型的优势，克服了离散扩散模型和传统蛋白质语言模型的局限性。通过在 CATH 数据集上的充分实验验证，LaGDif 展现了卓越的性能，并在序列恢复率和结构质量上均取得了显著提升。自集成方法的引入进一步提高了模型的鲁棒性和准确性。模型复杂度分析表明，LaGDif 在性能和效率之间取得了良好的平衡。

从客观角度评价，LaGDif 模型在蛋白质反向折叠领域做出了重要的贡献。其提出的潜在空间扩散框架和自集成方法为未来的研究提供了新的思路。实验结果充分支持了论文的结论，方法描述清晰，实验设计合理，评估指标全面。代码和模型公开，方便了后续研究的复现和应用。

然而，本文也存在一些可以进一步研究的方向。例如，可以探索 LaGDif 模型在更具挑战性的蛋白质设计任务中的应用，例如从头蛋白质设计或蛋白质-蛋白质相互作用预测。此外，可以进一步优化模型结构和训练策略，以提升推理速度，并降低内存消耗。

总体而言，LaGDif 是一项高质量的研究工作，其提出的模型和方法具有重要的理论意义和应用价值，有望推动蛋白质设计和工程领域的进步。

### 6. 参考文献与链接

*   **代码链接:** [https://github.com/TaoyuW/LaGDif](https://github.com/TaoyuW/LaGDif)
*   **CATH 数据库:** [https://www.cathdb.info](https://www.cathdb.info)
*   **ESMFold:** 论文中使用了 ESMFold [5] 来生成 3D 结构，相关论文可查阅 [5] 的参考文献。
*   **DSSP 方法:** 参考文献 [17]。
*   **EGNN:** 参考文献 [14]。
*   **Grade-IF:** 参考文献 [10]。
*   **ProteinMPNN:** 参考文献 [22]。

**其他参考文献:**

[1] X. Zhou et al., "Prorefiner: an entropy-based refining strategy for inverse protein folding with global graph attention," Nature Communications, vol. 14, no. 1, p. 7434, 2023.
[2] J. Jänes and P. Beltrao, "Deep learning for protein structure prediction and design-progress and applications,” Molecular Systems Biology, vol. 20, no. 3, pp. 162-169, 2024.
[3] N. Ferruz, S. Schmidt, and B. Höcker, "Protgpt2 is a deep unsupervised language model for protein design," Nature communications, vol. 13, no. 1, p. 4348, 2022.
[4] F. A. Lategan, C. Schreiber, and H. G. Patterton, "Seqprednn: a neural network that generates protein sequences that fold into specified tertiary structures," BMC bioinformatics, vol. 24, no. 1, p. 373, 2023.
[5] Z. Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," Science, vol. 379, no. 6637, pp. 1123-1130, 2023.
[6] C. Hsu et al., "Learning inverse folding from millions of predicted structures," in International conference on machine learning. PMLR, 2022, pp. 8946-8970.
[7] M. McPartlon, B. Lai, and J. Xu, "A deep se (3)-equivariant model for learning inverse protein folding," BioRxiv, pp. 2022-04, 2022.
[8] T. Bepler and B. Berger, "Learning the protein language: Evolution, structure, and function," Cell systems, vol. 12, no. 6, pp. 654-669, 2021.
[9] N. Anand and T. Achim, "Protein structure and sequence generation with equivariant denoising diffusion probabilistic models," arXiv preprint arXiv:2205.15019, 2022.
[10] K. Yi, B. Zhou, Y. Shen, P. Liò, and Y. Wang, "Graph denoising diffusion for inverse protein folding," Advances in Neural Information Processing Systems, vol. 36, 2024.
[11] J. L. Watson et al., "De novo design of protein structure and function with rfdiffusion," Nature, vol. 620, no. 7976, pp. 1089-1100, 2023.
[12] J. J. Yang, J. Yim, R. Barzilay, and T. Jaakkola, "Fast non-autoregressive inverse folding with discrete diffusion," arXiv preprint arXiv:2312.02447, 2023.
[13] Y. Shen and J. Ke, "Staindiff: Transfer stain styles of histology images with denoising diffusion probabilistic models and self-ensemble," in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 549-559.
[14] V. G. Satorras, E. Hoogeboom, and M. Welling, "E (n) equivariant graph neural networks," in International conference on machine learning. PMLR, 2021, pp. 9323-9332.
[15] Z. Lin et al., "Language models of protein sequences at the scale of evolution enable accurate structure prediction," BioRxiv, vol. 2022, p. 500902, 2022.
[16] R. Roche et al., "E (3) equivariant graph neural networks for robust and accurate protein-protein interaction site prediction," PLoS Computational Biology, vol. 19, no. 8, p. e1011435, 2023.
[17] W. Kabsch and C. Sander, "Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features," Biopolymers: Original Research on Biomolecules, vol. 22, no. 12, pp. 2577-2637, 1983.
[18] J. Ingraham et al., "Generative models for graph-based protein design," Advances in neural information processing systems, vol. 32, 2019.
[19] C. Tan, Z. Gao, J. Xia, B. Hu, and S. Z. Li, “Generative de novo protein design with global context," arXiv preprint arXiv:2204.10673, 2022.
[20] B. Jing et al., "Learning from protein structure with geometric vector perceptrons," in International Conference on Learning Representations, 2020.
[21] Z. Gao, C. Tan, and S. Z. Li, "Alphadesign: A graph protein design method and benchmark on alphafolddb," arXiv preprint arXiv:2202.01079, 2022.
[22] J. Dauparas et al., “Robust deep learning-based protein sequence design using proteinmpnn," Science, vol. 378, no. 6615, pp. 49–56, 2022.
[23] Z. Gao, C. Tan, P. Chacón, and S. Z. Li, "Pifold: Toward effective and efficient protein inverse folding," arXiv preprint arXiv:2209.12643, 2022.
