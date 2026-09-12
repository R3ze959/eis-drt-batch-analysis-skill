# EIS–DRT Batch Analysis

### 从一批阻抗文件，到可检查、可比较、可追溯的分析结果。

**1.0.0 · GPL-3.0-or-later · 本地计算 · 数据优先 · Codex Skill / Python CLI**

面向电化学阻抗谱（EIS）的批量处理与弛豫时间分布（DRT）分析工具。既能只整理阻抗、检查数据并输出 Nyquist/Bode 诊断图，也能进一步比较 DRT 模型、检查峰稳定性、评估实验控制证据，并导出可继续分析的数据。

它不是只输出一张图的脚本：**每条谱从哪里来、用了什么参数、哪些峰值得关注、哪些结果仍需复核，都有独立记录。** 适合电池及其他电化学体系的研究性分析；批量电压/循环序列需提供可核验的条件元数据。

[下载 1.0.0](https://github.com/R3ze959/eis-drt-batch-analysis-skill/releases/tag/v1.0.0) · [安装指南](eis-drt-batch-analysis/README.md) · [方法说明](eis-drt-batch-analysis/references/methodology.md) · [发布验证](RELEASE.md) · [许可证](LICENSE)

## 能做什么

| 功能 | 具体能力与价值 |
| --- | --- |
| **独立 EIS 批处理** | 读取多文件、多谱数据；检查频率、单位、虚部符号和原始行；计算实部、虚部、模值与相位。只要求 EIS 时不运行 DRT，也不擅自拟合物理等效电路。 |
| **质量检查与诊断** | EIS 可选 Lin-KK / Z-HIT 一致性检查；DRT 提供重构、残差、正则化敏感性及测量窗口内的峰支持判断。保留异常证据，不以删除电感点或平滑原始数据换取好看的结果。 |
| **DRT 多方法分析** | TR-RBF 主分析、TR-NNLS 对照，自适应比较无串联电感/含串联电感候选，进行正则化选择与敏感性检查，输出峰位置及支持区间内的积分贡献。 |
| **复杂响应候选** | Loewner RC/RL、条件触发的 signed/GDRT 与扩散候选；可选 model-and-reduce，检查减除前后峰稳定性及回加重构。原始分支、减除分支与推荐结果分别保留。 |
| **实验控制证据** | 利用有明确匹配关系的重复测量、振幅和静置时间序列，检查异常一致性、线性及稳定性证据。缺少这些数据时仍可探索性分析，但不会自动判为实验合格。 |
| **数据优先导出** | CSV、JSON、Origin-ready 配对列和绘图长表；按输入、质量、结果、绘图数据及复现记录分类。检查点与严格续跑/重导出机制，减少重复计算与版本混用。 |
| **按需趋势展示** | 在条件匹配、展示资格满足时，生成逐电压 3D 瀑布图和热图，输出 PNG / 可编辑文字 SVG、曲线映射与绘图检查报告。已有结果可单独绘图，无需重新拟合。 |

**默认流程：先处理数据、导出结果并保留诊断图；没有要求展示图时，处理结束后再询问。** 已明确要求的图无需重复确认。原始 Nyquist/Bode、重构、残差等诊断图不受这个询问限制。

## 为什么这样设计

- **一套环境，两条独立分支。** 简单阻抗处理不必启动完整 DRT；需要时再用原始输入进入 DRT 分支。
- **不把所有谱硬塞进普通 RC 模型。** 每条谱单独评估电感、模型适配、边界效应和稳定性；不会按文件序号或电压位置预设剔除规则。
- **计算完成、数值质量、实验有效性分开报告。** 好的重构不等于真实峰，KK 通过也不等于线性和稳定性已经证实。
- **数据能带走，过程能回查。** 不把分析锁在图片里；来源、参数、分支身份及运行状态都有机器可读记录。

## 快速开始

### 方式一：作为 Codex Skill

从 [Release](https://github.com/R3ze959/eis-drt-batch-analysis-skill/releases/tag/v1.0.0) 下载 `eis-drt-batch-analysis-1.0.0.zip` 和同名 `.sha256`。将 ZIP 交给 Codex，可使用下面的安装请求：

> 请检查这个 Skill 的内容，安装为个人 Skill；如果已有同名版本，先备份。检查 Python 3.11，按包内 bootstrap.py 安装独立环境并验证。成功后报告版本，不处理我的实验数据。

安装后可这样提出任务：

> 使用 $eis-drt-batch-analysis，只处理这个文件夹的 EIS，导出数据和诊断图，不做 DRT。

> 使用 $eis-drt-batch-analysis，对这批 EIS 做 DRT。比较适用模型，报告需复核的谱和峰，先导出数据，展示图处理完再问我。

> 使用已有 DRT 结果绘制电压 3D 瀑布图和热图，不重新拟合。按电池、方向和测试条件分组，保留所有符合展示条件的曲线。

### 方式二：直接使用 Python

解压发布 ZIP，进入其中的 `eis-drt-batch-analysis` 目录。若使用 Git，则运行：

```bash
git clone https://github.com/R3ze959/eis-drt-batch-analysis-skill.git
cd eis-drt-batch-analysis-skill/eis-drt-batch-analysis
```

macOS / Linux：

```bash
python3.11 scripts/bootstrap.py
.venv/bin/python scripts/check_runtime.py
.venv/bin/python -m pip check
```

Windows PowerShell：

```powershell
py -3.11 scripts/bootstrap.py
& .\.venv\Scripts\python.exe scripts/check_runtime.py
& .\.venv\Scripts\python.exe -m pip check
```

安装器创建独立环境，不覆盖已有 `.venv`，不修改系统 Python。已有环境验证通过即可使用，不需要重复安装。

处理命令示例（二选一；将示例路径替换为自己的路径）：

```bash
# EIS：数据、质量记录和原始 Nyquist/Bode 诊断图
.venv/bin/python scripts/batch_eis.py /absolute/raw-data --output-dir /absolute/project/eis

# DRT：数值结果与诊断图，默认不生成趋势展示图
.venv/bin/python scripts/batch_drt.py /absolute/raw-data --output-dir /absolute/project/drt
```

Windows 将解释器替换为 `& .\.venv\Scripts\python.exe`，有空格的路径加引号。输出目录使用新目录；不要把已有结果写回原始数据目录。

EIS 需要一致性审查时加 `--kk on --zhit on`。DRT 的减除候选用 `--model-reduce` 显式启用；明确需要趋势展示时用 `--trend-plots auto`，并提供有效分组元数据。详细参数见 [输入输出规范](eis-drt-batch-analysis/references/input-output.md) 和 [EIS 分支说明](eis-drt-batch-analysis/references/eis-only.md)。

### 不用实验数据，先验证安装

```bash
.venv/bin/python scripts/smoke_test.py
```

自检临时生成合成谱，运行真实计算并检查结果。需要保留演示数据时：

```bash
.venv/bin/python scripts/generate_demo.py /absolute/new-demo --nodes 4
.venv/bin/python scripts/batch_eis.py /absolute/new-demo/raw --manifest /absolute/new-demo/manifest.csv --output-dir /absolute/new-demo-eis
```

示例为合成数据，不来自任何私人电池或借用数据集。

## 最后会得到哪些文件

| 目录 | 内容 |
| --- | --- |
| `00_overview` | 阅读入口、输入清单、逐谱状态和索引 |
| `01_inputs` | 规范化观测值、原始行与来源追溯 |
| `02_quality` | 质量状态、重构/残差及对应分支的稳定性证据 |
| `03_results` | EIS 数值，或分支独立的 DRT / signed / 扩散 / 减除结果 |
| `04_plot_data` | CSV 长表、Origin-ready 配对列等可继续作图的数据 |
| `05_figures` | 自动诊断图，以及用户明确要求的展示图 |
| `06_reproducibility` | 参数、配置、代码和运行环境记录 |

**先读 `00_overview/README.md`，再看根目录 `run_manifest.json`。** 用稳定的谱 UID 关联结果，不依赖文件显示名；以最终运行状态判断导出是否完成。各分支的具体字段见 [输入输出规范](eis-drt-batch-analysis/references/input-output.md)。

## 环境与验证范围

- **需要：** CPython 3.11、可写目录，首次安装需要下载依赖；无需 GPU。
- **核心依赖：** pyimpspec、CVXOPT、NumPy、SciPy、Matplotlib、pandas、openpyxl；[完整固定版本](eis-drt-batch-analysis/requirements.lock.txt)。无需 DearEIS、Origin、MATLAB 或另一个 Skill。
- **已验收：** macOS arm64。Windows / Linux 提供入口与条件依赖，但 1.0 尚未完成这些系统的实机验收。
- **字体：** EIS 诊断图需要已合法安装的 Arial；缺失时保留数值结果并报告绘图失败。DRT 可记录明确的 DejaVu Sans 回退；本仓库不打包字体。
- **1.0 验证：** 329 项回归测试、15 项真实 CLI 集成检查、基础与高级合成谱检查通过；独立新环境完成安装和 EIS 运行。测试范围和局限见 [发布验证](RELEASE.md)，不等于任意实验数据均处理正确。

安装环境实测约 477 MiB（macOS，未计系统 Python 和下载缓存）；其他平台和版本可能不同。不要把带缓存的安装耗时当作首次联网下载承诺。

## 必须知道的边界

1. **DRT 峰不是机理标签。** 软件不会自动把峰归因为 SEI、电荷转移或扩散，也不会以拟合优度替代实验控制。
2. **高级分支是候选模型。** 当前 DDT 属于串联阻抗域候选，不是完整并联导纳 DDT；不自动给出扩散系数。弱峰、近邻峰、漂移、相关噪声和测量边界仍需审慎判断。
3. **并非所有数据都适合普通 RC 趋势图。** 不适用或证据不足的结果会保留记录、注明原因或进入诊断分支，不会为了整洁而静默删谱。
4. **输入格式有边界。** TXT/CSV/TSV、IRF 及部分由 pyimpspec 读取的厂商/表格格式可用；具体仪器版本需核对。含参考 DRT、历史输出和原始 EIS 的混合目录应先建立 manifest 选择清单。
5. **输出不自动匿名。** 计算在本地进行，但追溯记录包含用户输入路径与元数据；公开实验结果前审查独立副本。仓库和安装 ZIP 不包含私人案例、真实实验数据、个人运行路径或虚拟环境。
6. **复现不能强行跨版本。** 输入、数值参数、代码或环境变化后应新建运行；严格匹配时可 `--resume --export-only` 重导出，不重新计算。

## 许可与反馈

本项目原创代码和文档采用 **[GNU GPL v3 或更新版本](LICENSE)**。允许商业使用；分发受 GPL 覆盖的软件及修改版本时须履行相应源码与许可义务。第三方组件保留各自许可及署名，见 [第三方说明](eis-drt-batch-analysis/THIRD_PARTY_NOTICES.md)。

本工具是研究辅助软件，按许可证规定不提供担保。报告问题时，优先提交最小合成反例、版本、参数与去敏后的错误信息；**不要直接上传私人实验文件、身份信息或完整本机运行日志。**

---

**English summary:** A data-first, local EIS/DRT batch-analysis Skill and Python CLI. Independent EIS processing, audited DRT candidates, source provenance, structured exports and automatic diagnostic figures; presentation plots are opt-in. GPL-3.0-or-later. Release 1.0 is qualified on macOS arm64 with CPython 3.11; Windows/Linux remain unqualified. Numerical success is not experimental acceptance or mechanism identification.
