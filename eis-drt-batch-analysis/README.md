# EIS–DRT 批量分析 Skill 1.0

**版本 1.0.0 · GPL-3.0-or-later · 本地处理 · 数据优先**

将阻抗文件批量转成可追溯的数据、质量报告与诊断图；明确要求时再进行 DRT 或生成展示图。适合科研探索和辅助分析，不是测量合格认证器，也不自动判定电化学机理。

## 主要功能

| 功能 | 提供什么 |
| --- | --- |
| 独立 EIS | 批量读取、单位/虚部符号校验、模值与相位、Nyquist/Bode；不自动做 DRT |
| 质量检查 | 原始行追溯；EIS 可选 Lin-KK/Z-HIT；DRT 重构、残差与参数敏感性 |
| DRT 分析 | TR-RBF 主分析、TR-NNLS 对照、正则化选择、峰位置与支持区间内面积 |
| 复杂响应 | 自适应串联电感、Loewner RC/RL、条件触发的 signed/GDRT 与扩散候选；减除模型可选 |
| 控制与验收 | 匹配实测重复、振幅和静置序列；缺证据保持未评估，不自动验收通过 |
| 分类导出 | CSV/JSON、Origin-ready、分支独立结果、检查点与可复现参数 |
| 按需展示 | 逐电压 3D 瀑布图、原生网格热图、PNG/可编辑 SVG；不混淆不同条件 |

默认先导出数据并保留辅助理解的诊断图。未要求展示图时，Skill 在处理结束后询问是否需要；已明确要求则直接完成。后续可用现有结果绘图，不重新拟合。

## 1. 收到 ZIP 后怎么用

只转发 `eis-drt-batch-analysis-1.0.0.zip` 和对应 `.sha256` 校验文件，不转发开发目录或已经安装过的环境。包内已有源代码、依赖清单、安装工具、合成示例和许可证。

使用 Codex 时，可以把 ZIP 提供给助手并发送：

> 请先检查这个 Skill 的内容，将它安装为个人 Skill；如果已有同名版本，先备份，不直接覆盖。检查 Python 3.11，按包内 bootstrap.py 安装独立环境并验证，成功后报告版本和路径，不处理我的实验数据。

这是安装请求示例，不代表解压会自动执行。也可将完整目录放入你配置的 skills 目录，或直接使用命令行；不依赖 DearEIS、Origin 或另一个 Skill。

## 2. 环境与安装

需要已安装的 **CPython 3.11**、可写目录，以及首次下载依赖的网络。无需 GPU。ZIP 不含 Python 和科学计算库；环境体积和耗时取决于平台、网络与缓存。

| 平台 / 字体 | 1.0 的支持边界 |
| --- | --- |
| macOS arm64 + Python 3.11 | 本地发布验收平台 |
| Windows / Linux | 提供 Python 入口与条件依赖，尚未实机验收 |
| EIS 诊断图 | 要求已安装且许可适用的 Arial；缺失时保留数据并报告绘图失败 |
| DRT 图 | Arial 优先；支持记录明确的 DejaVu Sans 回退，不打包专有字体 |

以下命令均在解压后的 `eis-drt-batch-analysis` 目录运行。

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

安装器只创建新的 `.venv`，不覆盖既有环境，不修改系统 Python。已存在且验证通过的环境无需重装。另装时使用 `--venv /absolute/new-environment`，之后改用新环境的 Python。

固定依赖包括 pyimpspec、CVXOPT、NumPy、SciPy、Matplotlib、pandas 和 openpyxl，精确版本见 [requirements.lock.txt](requirements.lock.txt)。不静默降级核心依赖。

下载失败会保留未完成环境，不代表安装成功；不要直接删除未知目录。可换新 `--venv` 路径重试。慢网络可设置 `PIP_TIMEOUT=120`、`PIP_RETRIES=5`；PowerShell 用 `$env:PIP_TIMEOUT="120"` 和 `$env:PIP_RETRIES="5"`。显式镜像、代理和证书设置保留，磁盘 pip 配置与安装重定向被隔离。详情见 [环境说明](references/software.md)。

## 3. 处理数据

给 Skill 的请求示例：

> 只处理这个文件夹的 EIS，导出数据和诊断图，暂时不做 DRT。

> 对这些 EIS 做 DRT 分析，优先导出数据和质量结论，展示图结束后再问我。

直接命令（二选一）：

```bash
.venv/bin/python scripts/batch_eis.py /absolute/raw-data --output-dir /absolute/project/eis
.venv/bin/python scripts/batch_drt.py /absolute/raw-data --output-dir /absolute/project/drt
```

Windows 使用 `.venv\Scripts\python.exe` 替换解释器；有空格的路径加引号。输出目录须为空。混有原始 EIS、参考 DRT 或历史导出的目录，先建立 manifest 选择清单，勿盲目整目录导入。

文本支持 TXT/CSV/TSV 和 IRF；部分厂商及 ODS/XLSX 格式由 pyimpspec 读取，具体导出版本需核验。必须确认频率、实部、虚部符号及单位，不凭数值大小猜测。[输入输出规范](references/input-output.md)

EIS 一致性检查可加 `--kk on --zhit on`；DRT 可选 `--model-reduce`。DRT 默认不生成展示趋势图；明确要求后用 `--trend-plots auto`。趋势图需要条件一致、具备电压和电池/样品身份的兼容分组；无合格分组会记录跳过原因，不为出图降低标准。

处理后先打开 `00_overview/README.md` 和根目录 `run_manifest.json`。结果位于 `03_results`，质量证据位于 `02_quality`，Origin-ready 位于 `04_plot_data`。数值完成不等于实验合格。

## 4. 安装后自检

```bash
.venv/bin/python scripts/smoke_test.py
```

测试临时生成双 RC + 串联电感合成谱，检查真实计算、峰恢复和诊断图；结束后自动清理测试目录，不访问实验数据。首次运行可能需要数分钟，不是瞬间完成的版本检查。

需要保留的合成示例时，用 `scripts/generate_demo.py /absolute/new-demo --nodes 4`，再按生成的 manifest 处理。合成谱也可能被标为需复核，不能据此保证趋势图或真实峰验收。

## 5. 分享与边界

- 包内无个人路径、私人配置、真实实验数据、虚拟环境或缓存；第三方署名和许可证依法保留。
- 源码按 GPL-3.0-or-later 分发；重新分发第三方库或整个环境须另行核对其许可证。[第三方说明](THIRD_PARTY_NOTICES.md)
- 分析输出保留原文件路径和实验元数据供追溯，并非自动匿名。分享实验结果前审查独立副本，不改写权威检查点。
- 普通 RC、signed/GDRT 与扩散候选分开导出；DDT 当前是串联阻抗域候选，不是完整并联导纳 DDT。不自动计算扩散系数或给峰指认机理。
- 缺少重复、振幅、静置证据时保持探索性；KK 通过不能单独证明线性、稳定性或测量质量。
- 跨版本、代码、数值参数或环境变化不允许强行续跑。保留旧软件与旧输出，新版本用新目录。相同环境下 `--resume --export-only` 可重导出而不重算。

方法细节见 [方法说明](references/methodology.md)，图形规则见 [绘图规范](references/plotting.md)，验收与限制见 [发布说明](RELEASE_NOTES.md)，历史变更见 [CHANGELOG.md](CHANGELOG.md)。
