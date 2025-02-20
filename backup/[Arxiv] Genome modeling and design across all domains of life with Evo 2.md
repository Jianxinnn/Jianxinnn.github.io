1. 背景介绍

生命体将所有信息都编码在DNA中。随着基因组测序、合成和编辑技术的革新，生物学研究已经发生了巨大的变革。然而，要智能地构建新的生物系统，还需要对基因组中蕴含的复杂性有深入的理解。基因组的复杂性远远超出了人类直观理解的范围，但人工智能领域的进步为我们提供了一个通用的框架，能够利用大规模数据和计算能力来揭示高阶模式。

专业术语解释:

基因组 (Genome): 一个生物体或病毒包含的全部遗传物质，包括DNA或RNA (对于某些病毒)。在细胞生物中，基因组主要由DNA构成，包含了编码蛋白质、RNA以及调控序列的所有遗传信息。

DNA碱基对 (DNA base pairs): DNA分子是由两条链相互缠绕形成双螺旋结构，链上的碱基通过氢键相互配对。腺嘌呤（A）总是与胸腺嘧啶（T）配对，鸟嘌呤（G）总是与胞嘧啶（C）配对。碱基对是DNA信息存储的基本单位。

生物系统 (Biological systems): 生物体内相互作用的组分集合，这些组分协同工作以执行特定的生物功能。生物系统可以涵盖从分子水平（如蛋白质复合物）到细胞、组织、器官乃至整个生物体的各个层面。

人工智能 (Artificial Intelligence, AI): 研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统的一门新的技术科学。人工智能旨在使计算机能够执行通常需要人类智能才能完成的任务，例如学习、推理、问题解决、感知、语言理解和创造力。

高阶模式 (Higher-order patterns): 数据中复杂且不易察觉的模式，这些模式超越了简单的线性关系，通常需要复杂的模型和算法才能识别和理解。在基因组数据中，高阶模式可能涉及基因之间的复杂相互作用、调控元件的精细调控机制以及基因组结构与功能之间的非线性关系。

论文作者认为，为了让机器能够设计生命功能，需要使其学习对生物复杂性的深入且通用的表征。为了实现这一目标，需要使用跨越生物多样性全谱的数据来训练模型，以发现类似于其他领域中涌现的特性。

2. 核心概述

本文介绍了Evo 2，这是一个生物学基础模型，它在来自生命所有领域的、经过高度整理的基因组图谱中的9.3万亿DNA碱基对上进行了训练。Evo 2模型拥有70亿和400亿参数两种版本，并具有前所未有的100万token上下文窗口和单核苷酸分辨率。研究表明，Evo 2仅从DNA序列中学习，就能准确预测遗传变异的功能影响，无需针对特定任务进行微调，例如预测非编码致病突变和临床上重要的BRCA1变异。通过应用机制可解释性分析，研究者揭示了Evo 2自主学习了广泛的生物学特征，包括外显子-内含子边界、转录因子结合位点、蛋白质结构元件和前噬菌体基因组区域。除了预测能力，Evo 2还能生成线粒体、原核生物和真核生物基因组尺度的序列，其自然性和连贯性优于以前的方法。通过推理时搜索引导Evo 2，可以实现对表观基因组结构的可控生成，并展示了生物学中首个推理时扩展的结果。为了加速生物复杂性的探索和设计，作者完全开源了Evo 2，包括模型参数、训练代码、推理代码和OpenGenome2数据集。

3. 方法与实验细节

本文提出的Evo 2模型涉及训练过程，以下将详细介绍数据集、算法模型、训练和评估过程。

数据集 (Dataset):

名称: OpenGenome2

大小: 9.3万亿 DNA 碱基对 (训练数据), 数据集总计超过8.8万亿核苷酸

构建方法:

高度整理的基因组图谱，跨越生命的所有领域。

数据来源包括细菌、古细菌、真核生物和噬菌体。

数据集来源于精心策划的非冗余核苷酸序列数据。

数据来自GTDB, NCBI genomes, Metagenomics (IMG/VR), EPDnew 等数据库。

强调数据的多样性和代表性，涵盖了广泛的生物类群和基因组类型。

数据集划分: 论文中没有明确划分训练集、测试集和验证集，但提到了验证集用于超参数调整和模型选择，测试集用于评估模型性能。数据集主要用于模型的预训练和中途训练。

算法和模型 (Algorithms and Models):

算法: 基于Transformer架构的生物学基础模型。

模型架构: Evo 2 使用 StripedHyena 2 架构，这是一种卷积多混合架构。

基础模型: StripedHyena 2

关键组件:

多混合架构 (Multi-hybrid architecture): 结合了多种不同类型的算子，包括输入相关的卷积算子和注意力机制，旨在平衡模型质量与训练和推理效率。

Striped pattern: 将不同类型的算子以条纹模式排列，利用它们之间的协同作用。

Hyena 算子: 使用三种不同变体的输入依赖卷积算子 (Short Explicit (SE), Medium Regularized (MR), and Long Implicit (LI) hyena operators)。

注意力机制 (Attention): 模型中也包含注意力层，用于捕捉序列中的长程依赖关系。

Rotary Positional Embeddings (RoPE): 使用旋转位置编码，以便有效地扩展上下文长度。

模型参数: 提供 7B 和 40B 两种参数规模的模型。

上下文窗口 (Context Window): 100万 tokens (单核苷酸分辨率)。

训练损失函数 (Training Loss): 重加权交叉熵损失 (Reweighted Cross Entropy Loss, IWCE)。

公式: 
L
I
W
C
E
=
1
Z
∑
t
w
t
ℓ
C
E
(
t
)
L 
IWCE
​
 = 
Z
1
​
 ∑ 
t
​
 w 
t
​
 ℓ 
CE
​
 (t)

ℓ
C
E
(
t
)
ℓ 
CE
​
 (t)
: 在位置 t 的交叉熵损失。

w
t
w 
t
​
 
: 位置 t 的权重，如果位置 t 在重复区域，则 
w
t
=
0.1
w 
t
​
 =0.1
，否则 
w
t
=
1.0
w 
t
​
 =1.0
。

Z
=
0.1
×
N
r
e
p
e
a
t
+
N
n
o
n
_
r
e
p
e
a
t
Z=0.1×N 
repeat
​
 +N 
non_repeat
​
 
: 归一化因子，确保损失缩放的一致性。

损失函数推导和逻辑: 该损失函数旨在降低重复DNA序列区域对损失函数的贡献，从而提高模型在非重复区域和下游任务上的性能。这种方法可以更好地校准重复序列和非重复序列之间的可能性，已被证明在其他DNA模型中可以提高下游任务的性能。

训练和评估过程 (Training and Evaluation Process):

训练过程: 两阶段训练策略

预训练 (Pretraining):

上下文长度: 8192 tokens

数据加权: 侧重于基因窗口，学习功能性遗传元件。

使用前 3T tokens 的小写 tokens 来标识重复区域。

中途训练 (Midtraining):

多阶段中途训练，逐步扩展上下文长度至 100 万 tokens。

数据组成调整：包含更多完整基因组和更长的平均序列长度。

使用旋转位置编码的变体来适应更长的序列。

实验设计:

消融实验: 对比不同数据组成和损失函数的模型性能，验证数据加权和上下文扩展策略的有效性。

模型扩展: 训练 7B 和 40B 两种参数规模的模型，研究模型规模对性能的影响。

长上下文评估: 设计 "needle-in-a-haystack" 合成评估任务，评估模型在长上下文中的信息检索能力。

下游任务评估: 在多种生物学任务上评估 Evo 2 的性能，包括变异效应预测、基因必要性预测、mRNA 衰减预测、外显子/内含子分类等。

评估指标:

困惑度 (Perplexity): 评估语言模型的训练效果。

AUROC, AUPRC: 评估分类任务的性能，例如变异致病性预测和外显子分类。

Spearman 相关系数: 评估模型预测与实验数据之间的相关性，例如 DMS 实验。

序列恢复率 (Sequence Recovery): 评估基因序列生成任务的性能。

TM-score, pLDDT: 评估蛋白质结构预测的质量。

中间过程:

预训练阶段使用 8192 上下文长度。

中途训练阶段逐步扩展上下文长度至 100 万 tokens。

在不同生物学任务上进行零样本和微调评估。

使用稀疏自编码器 (SAE) 进行机制可解释性分析。

利用推理时搜索进行可控的表观基因组生成。

4. 研究过程与结论

本文的研究过程围绕Evo 2模型的训练、评估和分析展开，旨在验证其在生物序列建模和设计方面的通用能力。

变异效应预测:

实验设计: 作者评估了Evo 2在预测不同类型遗传变异（包括SNV和non-SNV，编码区和非编码区变异）功能影响方面的能力，使用了ClinVar, SpliceVarDB, BRCA1/2 变异数据集。

研究过程: 使用 Evo 2 的零样本可能性评分来预测变异的致病性。还训练了基于 Evo 2 embeddings 的监督分类器来预测 BRCA1 变异的影响。

实验结果: Evo 2 在多种变异效应预测任务上表现出色，尤其是在非编码变异和剪接变异预测方面，取得了领先水平。监督模型在 BRCA1 变异分类任务上超越了零样本方法和其他基线模型。

结论: Evo 2 能够准确预测人类临床变异的效应，证明了其在理解人类疾病遗传基础方面的潜力。

机制可解释性分析:

实验设计: 使用稀疏自编码器 (SAE) 来解释 Evo 2 模型学习到的表征，旨在揭示模型内部与生物学概念相关的潜在维度。

研究过程: 在 Evo 2 的中间层训练 SAE，并使用对比特征搜索方法将 SAE 特征与已知的生物学概念对齐。

实验结果: SAE 揭示了 Evo 2 学到了一系列与关键生物学特征相对应的潜在维度，包括外显子-内含子边界、转录因子结合位点、蛋白质二级结构和前噬菌体区域。

结论: Evo 2 能够自主学习广泛的生物学特征，这些特征可以用于基因组注释和生物学发现。

基因组生成:

实验设计: 评估 Evo 2 在生成不同类型基因组序列方面的能力，包括线粒体基因组、最小细菌基因组和酵母染色体。还探索了使用推理时搜索引导 Evo 2 生成具有特定表观基因组特征的序列。

研究过程: 使用 Evo 2 进行无约束的自回归生成，并评估生成序列的自然性和生物学相关性。使用 Enformer 和 Borzoi 模型作为评分函数，通过 beam search 引导 Evo 2 生成具有可控染色质可及性的 DNA 序列。

实验结果: Evo 2 能够生成自然且连贯的基因组尺度序列，包括完整的线粒体基因组、最小细菌基因组和酵母染色体。通过推理时搜索，Evo 2 实现了对表观基因组结构的可控生成，并成功编码了 Morse 代码信息。

结论: Evo 2 具备强大的基因组设计能力，可以用于生成具有特定功能和表观基因组特征的新型生物序列。

总的来说，论文的结论是 Evo 2 作为一个生物学基础模型，在生物序列的预测和生成方面都展现了强大的通用能力，为计算生物学领域的下游应用提供了坚实的基础。

5. 总结与客观评价

本文深入研究了生物学基础模型 Evo 2，并对其在基因组建模和设计方面的能力进行了全面的评估。Evo 2 模型通过在大规模、多样化的基因组数据集上进行训练，有效地学习了生物序列的复杂规律，并在变异效应预测、机制可解释性和基因组生成等多个方面取得了显著的成果。

客观评价:

优点:

通用性: Evo 2 展现了跨生命领域和生物学尺度的通用建模能力，能够处理 DNA、RNA 和蛋白质序列，并应用于多种生物学任务。

高性能: 在变异效应预测、基因组生成等任务上，Evo 2 取得了与或超越现有专门模型的性能。尤其在非编码变异和长序列生成方面，优势显著。

可解释性: 通过 SAE 分析，揭示了模型学习到的可解释生物学特征，为理解模型的工作机制提供了 insights。

可控生成: 通过推理时搜索，实现了对基因组特性的可控设计，展示了生成式生物学的潜力。

开源: 模型、代码和数据集的完全开源，极大地促进了研究的可重复性和社区的进一步发展。

不足与局限性:

病毒序列预测性能较弱，这与训练数据排除了真核病毒序列有关，但也反映了模型在病毒领域泛化能力的不足。

生成的真核基因组序列在 tRNA 和基因特征密度方面略低于天然基因组，表明生成序列的自然性仍有提升空间。

机制可解释性分析虽然揭示了模型学习到的一些生物学特征，但对更复杂生物学模式的理解仍需深入探索。

总而言之，Evo 2 是一项重要的研究成果，代表了生物学基础模型领域的重大进展。它为生物学研究提供了强大的计算工具，有望加速基因组功能解析和生物系统设计，但也需要在特定领域和应用中持续改进和完善。

6. 参考文献与链接

论文链接: 用户未提供论文链接，请补充论文的DOI或URL以便添加。

代码仓库: [https://github.com/arcinstitute/evo2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Farcinstitute%2Fevo2), [https://github.com/zymrael/savanna](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fzymrael%2Fsavanna), [https://github.com/zymrael/vortex](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fzymrael%2Fvortex)

数据集链接: [https://huggingface.co/datasets/arcinstitute/opengenome2](https://www.google.com/url?sa=E&q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Farcinstitute%2Fopengenome2)

Evo Designer: [https://arcinstitute.org/tools/evo/evo-designer](https://www.google.com/url?sa=E&q=https%3A%2F%2Farcinstitute.org%2Ftools%2Fevo%2Fevo-designer)

Evo Mech Interp Visualizer: [https://arcinstitute.org/tools/evo/evo-mech-interp](https://www.google.com/url?sa=E&q=https%3A%2F%2Farcinstitute.org%2Ftools%2Fevo%2Fevo-mech-interp)

NVIDIA Evo 2 NIM (generation): [https://build.nvidia.com/nvidia/evo2-protein-design](https://www.google.com/url?sa=E&q=https%3A%2F%2Fbuild.nvidia.com%2Fnvidia%2Fevo2-protein-design)

NVIDIA Evo 2 NIM (forward): [https://build.nvidia.com/arc/evo2-40b](https://www.google.com/url?sa=E&q=https%3A%2F%2Fbuild.nvidia.com%2Farc%2Fevo2-40b)

NVIDIA BioNeMo version of Evo 2 code: [https://github.com/NVIDIA/bionemo-framework](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2FNVIDIA%2Fbionemo-framework)