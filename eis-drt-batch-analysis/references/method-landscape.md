# DRT 方法取舍与数据处理决策

调研日期：2026-09-09。本文是服务于本 Skill 数据处理迭代的焦点调研，非完整市场盘点，也不对工具作普适优劣排名。检索覆盖 pyimpspec/DearEIS、DRTtools/pyDRTtools、EISART、BioLogic、Bayesian/BHT、Loewner、GP-DRT、稀疏/elastic-net 及 PyDRT 基函数路线；未安装比较软件，未使用私人参考曲线调参。

证据分层：**实现证据**来自官方文档或作者源码；**方法证据**来自原始论文，摘要可访问与全文可访问须区别；**工程决策**是本 Skill 的实现与验收选择，需合成反例及运行记录支持。上游 `main/master` 链接可能变化，复现以本次运行保存的实际依赖和代码身份为准。

## 方法路线与适用边界

| 路线 | 对本 Skill 有用的部分 | 不能直接继承的结论 |
| --- | --- | --- |
| TR-RBF：pyimpspec/DearEIS、DRTtools/pyDRTtools | 保留复数阻抗重构、基函数/导数阶数记录、自动 λ 建议及固定 λ 敏感性。 | 自动 λ 可能制造尖峰或合并峰；相同 λ 在不同离散化、归一化或惩罚定义下不一定等效。软件名不同也不代表方法独立。[1][2] |
| Bayesian DRT / BHT | 条件可信区间与实部—虚部一致性评分补充确定性反演。 | BHT 存在随机初始化；可信区间取决于模型、先验和噪声假设，不能覆盖全部误差，也不能替代幅值/静置/重复实验。[2][3] |
| EISART | 作者实现将逐点权重、迭代异常点处理、引线相关项和重构残差联动。 | 过强异常点抑制可能排除可用点；降权又会影响其自适应 λ。不能把自动裁点、边界峰抑制或较小加权残差直接当作数据被修复。[4] |
| Loewner | 以离散模态提供另一类数学模型的 RC/RL 交叉检查。 | 不使用 Tikhonov λ 仍需选择模型阶数，且受噪声与分布过程影响；离散贡献不是连续 γ 峰高，不能直接对齐数值大小。[5] |
| GP-DRT / GP 噪声估计 | GP-DRT 提供均值、协方差和条件预测；另一研究路线用重复谱或单谱 GP 估噪后加权。 | 带外预测仍依赖先验，不能升级为测量支持的峰。单谱 GP 噪声加权本轮仅核实到预印本，不作为默认流程。[6][7] |
| 稀疏、elastic-net、解析基函数 | L1/L2 混合或 Cole–Cole/Havriliak–Negami 基函数提供不同形状假设；扩展核可处理普通正 RC-DRT 不适用的响应。 | 更稀疏、更尖的峰不保证更正确；正则、基函数与符号约束改变可恢复形状，需要针对离散峰、宽峰、重叠峰及核失配分别验证。[8][9] |

BioLogic AN60 用模拟 R/Q 响应说明 DRT 的分辨能力，同时讨论基础 Voigt 核面对发散低频响应和高频感抗的局限。将其用作核适用性提醒；不能把该局限推广为所有扩展 DRT 都应删除相应频段。先比较显式 series-L、声明的低频分支或其他核，并保留原谱。[10]

## 本轮默认采用什么

新增的数据变换限于**统一标量数值缩放**：主 TR-RBF 和次 TR-NNLS 使用纳入点的 `s = median(abs(Z))`，内部求解 `Z/s`，随后恢复阻抗、γ 与可用条件区间的输入单位。频率、τ、原点与相对残差不因该变换改变。`--impedance-scaling off` 用于明确的旧数值尺度对比。[numerical_conditioning.py](../scripts/numerical_conditioning.py)

这属于数值条件改善，不是逐频率噪声权重、去噪、平滑或插值。主 TR-RBF 仍使用实部/虚部绝对误差的等权目标。已有高级 signed/GDRT、DDT 与部分重构诊断的模值加权须按各自公式单独报告，不能据主通道缩放宣称全流程采用相同误差模型。

工程依据是目标与停止条件的尺度关系：在同一线性模型和二次惩罚下，阻抗与解同比缩放只给理想目标乘共同常数；有限精度求解器的绝对容差却可能使数值结果改变。上游 mGCV 目标也随阻抗尺度平方变化，而 SLSQP 使用目标变化等停止条件。因此缩放须覆盖自动 λ 建议和固定 λ 求解，不只处理最终曲线。[11][12]

验收须比较同一合成谱在不同整体阻抗尺度下的峰位置、还原后的 γ、积分贡献和重构，包含噪声、非等距频点和 series-L。通过这些检查仅支持所测数值尺度范围，不证明高动态范围、弱小重叠峰或任意实验噪声都已解决。

## λ 决策与边界验收

- mGCV 从用户范围内三个不同内部起点运行，保留每次原始返回值与拒绝原因；越界、贴用户边界或贴上游搜索边界的返回值不经剪裁后参加共识。至少两个合格返回值在当前跨度阈值内才形成起点一致性证据。起点一致不等于优化器成功：上游未暴露的状态必须明确标注。[batch_drt.py](../scripts/batch_drt.py)
- `consensus` 缺少可靠 mGCV 候选时使用声明的网格 L-curve fallback；显式 `package-mgcv` 无合格共识时失败。建议值、实际值、剪裁、fallback 与边界原因分别保留，不能将多个起点被剪裁到同一端点解释为独立收敛。
- L-curve 内点选择本身不足以获得边界验收。主通道检查保护区邻接、保护区更大弦距、未定义拐角以及完整 λ 除以/乘以 10 邻域。高级 signed/GDRT、DDT 同样必须将搜索边界、退化或缺失拐角纳入验收；不能靠排除端点再选出的“内部峰”掩盖搜索范围不足。边界状态需进入下游数值结论，而非只写提示。
- 固定 λ 扫描必须关闭自动 CV；这是既有正确行为，继续保留。不同 λ 的最小训练残差不作为自动最优判据，`rGCV` 也不等于抗离群损失。[1][2]

这些是可审计的工程规则，不是论文证明的普适阈值。继续保留 no-L/series-L 竞争及完整复数重构；TR-NNLS 实部结果只提供次算法检查，其包内指标不充当完整复数拟合验收。

## 本轮不默认采用什么

不默认引入 IRLS 降权、自动删点或单谱 GP 估噪。较大残差也可能来自漂移、符号/单位错误、感抗或扩散核失配；先改变点的影响力可能隐藏这些原因。任何未来降权分支都应保留完整原点、原域残差、权重来源及前后敏感性，不因加权残差更低就自动接受。

不机械地用 GP、稀疏峰或 Loewner 替换 TR-RBF。方法分歧保留为算法/模型不确定性。条件区间、λ 敏感性、重复离散度和频带边界不是同一类不确定性；带外峰与带符号峰均不自动指向唯一机制。

保留训练/holdout 分组锁定，禁止按 holdout 调参；参考图复现不代替科学验收。KK/BHT 一致性、良好重构及尺度测试通过均不能替代重复、幅值和静置对照。缺少实验控制时继续允许标注的探索性计算，但不获得完整接受。具体标准见 [methodology.md](methodology.md)。

## 原始与官方来源

1. pyimpspec 官方 [DRT 指南](https://vyrjana.github.io/pyimpspec/guide_drt.html)、[DRT API](https://vyrjana.github.io/pyimpspec/apidocs_drt.html)；[DearEIS 官方文档](https://vyrjana.github.io/DearEIS/)。实现证据。
2. [DRTtools 作者仓库](https://github.com/ciuccislab/DRTtools)、[pyDRTtools 作者仓库](https://github.com/ciuccislab/pyDRTtools)；Maradesa et al., 2023, *Selecting the Regularization Parameter in the Distribution of Relaxation Times*, [DOI](https://doi.org/10.1149/1945-7111/acbca4)。源码/官方说明已查看；论文元数据与方法引用已核，非全文重评。
3. Liu et al., 2020, *A Bayesian view on the Hilbert transform and the Kramers–Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores*, [DOI](https://doi.org/10.1016/j.electacta.2020.136864)、[作者全文](https://jiapeng-liu.github.io/files/JP-Liu_2020_BHT-DRT_Elec-Acta.pdf)。方法证据。
4. [EISART 作者仓库](https://github.com/leehangyue/EISART)、[权重/迭代源码](https://raw.githubusercontent.com/leehangyue/EISART/version20220527/contents/code/util_tikhonov.py)；Li et al., 2022, *Robust and fast estimation of equivalent circuit model from noisy electrochemical impedance spectra*, [DOI](https://doi.org/10.1016/j.electacta.2022.140474)。源码已查看，论文摘要范围。
5. Rüther et al., 2023, *Introducing the Loewner Method as a Data-Driven and Regularization-Free Approach for the Distribution of Relaxation Times Analysis of Lithium-Ion Batteries*, [原论文](https://www.mdpi.com/2313-0105/9/2/132)、[DOI](https://doi.org/10.3390/batteries9020132)。出版者摘要/对照表与官方实现文档范围。
6. Liu and Ciucci, 2019, *The Gaussian process distribution of relaxation times*, [DOI](https://doi.org/10.1016/j.electacta.2019.135316)、[作者代码](https://github.com/ciuccislab/GP-DRT)。论文摘要/作者说明范围。
7. Bartsch et al., 2024, *Weighted distribution of relaxation time analysis of battery impedance spectra using Gaussian process regression for noise estimation*, [预印本 DOI](https://doi.org/10.26434/chemrxiv-2024-1gxgq)、[作者机构记录](https://juser.fz-juelich.de/record/1035069)。本轮未核实正式期刊版。
8. Kobayashi and Suzuki, *Extended Distribution of Relaxation Time Analysis for Electrochemical Impedance Spectroscopy*, [DOI](https://doi.org/10.5796/electrochemistry.21-00111)、[作者接受稿](https://www.jstage.jst.go.jp/article/electrochemistry/advpub/0/advpub_21-00111/1/_pdf/-char/ja)。已查看扩展核与 elastic-net 方法；另见 Li et al., 2019, [结构稀疏方法原论文](https://doi.org/10.1016/j.electacta.2019.05.010)，摘要范围。
9. Leonhardt et al., 2025, *Reconstructing the distribution of relaxation times with analytical basis functions*, [DOI](https://doi.org/10.1016/j.jpowsour.2025.237403)、[PyDRT 作者仓库](https://github.com/robertleonhardt/PyDRT)。作者实现说明范围。
10. BioLogic, [Application Note 60](https://my.biologic.net/documents/battery-eis-distribution-of-relaxation-times-drt-electrochemistry-application-note-60/)。官方应用说明，网页标注更新于 2024-06-25；不是覆盖所有现代扩展核的通用删除规则。
11. [pyimpspec TR-RBF 上游源码](https://raw.githubusercontent.com/vyrjana/pyimpspec/main/src/pyimpspec/analysis/drt/tr_rbf.py)、[SciPy SLSQP 停止条件](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)。实现证据；本 Skill 未声称修改上游优化器。
12. [CVXOPT 求解器参数](https://cvxopt.org/userguide/coneprog.html#algorithm-parameters)、[官方 coneprog 源码](https://raw.githubusercontent.com/cvxopt/cvxopt/master/src/python/coneprog.py)。绝对/相对容差及显式 options 的实现证据；实际运行以安装版本为准。

## 下一步：方差权重

方差权重需先有真实 repeat metadata：同一 cell/sample、SOC/电位、温度、静置、幅值、协议和阻抗单位，且能识别频率重叠与时间漂移。随后用具有已知频率相关方差的 heteroscedastic 合成谱，连同同方差、离群点、连续漂移及核失配反例，验证权重恢复、噪声底限、峰/面积偏差和条件区间覆盖；再决定是否加入可选加权求解。重复不足或状态不匹配时，不把残差估计出的权重写成实测噪声方差。
