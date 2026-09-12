# v1.0.0 — 首次 GitHub 发布

数据优先的 EIS / DRT 批量分析工具，采用 GPL-3.0-or-later。仓库首页提供安装、功能与使用示例；完整 Skill 位于 `eis-drt-batch-analysis/`。

## 本次发布内容

- EIS 独立处理：来源、单位与符号检查，数值导出，Nyquist/Bode 诊断；可选 KK/Z-HIT。
- DRT：TR-RBF / TR-NNLS、自适应串联电感、正则化与峰稳定性检查；Loewner 和条件触发的 signed/GDRT、扩散候选，以及可选 model-and-reduce。
- CSV / JSON / Origin-ready 分类数据、分支独立结果、严格续跑及重导出。
- 诊断图自动保留；电压瀑布图、热图和其他展示图按需生成。
- 独立安装工具、固定依赖、合成数据生成器、测试代码、方法说明与第三方许可。

## 发布包身份

本次直接发布于 2026-09-11 完成本地验收的冻结包，未在上传时修改数值算法或安装包内容。仓库根目录的介绍与本说明为 GitHub 发布新增文档；包内发布说明保留本地冻结时的状态记录，不代表 GitHub 尚未发布。

| 项目 | 值 |
| --- | --- |
| 版本 | `1.0.0` |
| 文件 | `eis-drt-batch-analysis-1.0.0.zip` |
| 大小 | 362,363 字节，约 354 KiB |
| 内容 | 69 个允许发布的源文件及 1 个生成的校验清单 |
| SHA256 | `f6d2b602a57abb44334754326353d10e2eae4b153faba801fd37db9189abde7c` |

GitHub Release 附安装 ZIP 与 `.sha256`。页面自动生成的 “Source code” 压缩包是整个 Git 仓库快照，目录布局不同；普通安装建议下载上述具名 ZIP。

## 已完成的验证

本地冻结验收记录如下；完整机器日志包含本机路径，不作为公共资产上传。测试源码已随包提供，可自行重跑。

| 检查 | 验收结果 |
| --- | --- |
| 单元与回归 | 329 / 329 通过 |
| 真实命令行、来源、导出与续跑 | 15 / 15 通过 |
| 基础合成谱 | 通过 |
| 高级合成谱 | RC、RC+L、RC+RL、RC+Warburg、噪声、漂移、重复异常、振幅/静置控制及高级分支检查通过 |
| 独立新环境 | macOS arm64、CPython 3.11；安装、运行时、依赖检查与 4 条合成 EIS 处理通过 |
| 数据保真 | 独立检查原始复阻抗数值与源哈希一致；电感点保留；未请求的 DRT 未执行 |
| 分发检查 | 源文件允许清单、隐私扫描、ZIP 完整性与重复构建一致性通过 |

独立环境约 477 MiB；安装测试使用了 wheel 缓存，不能作为无缓存下载速度承诺。以上为软件行为测试，不是实验误差或真实峰识别正确率。

## 支持边界

- 已实际验收 macOS arm64；Windows / Linux 尚未实机验收。
- Python 与依赖另行安装，不在 ZIP 中。
- EIS 图需要合法安装的 Arial；DRT 支持有记录的字体回退。
- 不是所有谱都适合普通 RC-DRT；缺少控制数据时不会自动实验验收通过。
- DDT 为串联阻抗域候选，非完整并联导纳 DDT；不自动指认机理或计算扩散系数。
- 软件包不包含私人实验数据；用户运行后的科学输出并非自动匿名。

## 复验命令

在已安装环境的 Skill 目录运行：

```bash
.venv/bin/python scripts/check_runtime.py
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s scripts -p '*_test.py'
.venv/bin/python scripts/integration_test.py --output-root /absolute/new-integration
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/advanced_smoke_test.py
.venv/bin/python scripts/build_release.py --check
```

Windows 使用 `.venv\Scripts\python.exe`。合成测试不需要真实实验数据。不要用私人留出数据反复调参后再称为盲测。
