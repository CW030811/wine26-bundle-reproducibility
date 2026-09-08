# 对最终源码所做的可运行性补丁

`src/deterministic/` 以 `Bundle_code/` 为源。为了让最终源码在论文指定的 PyTorch 2.8 / PyG 2.6 环境中直接运行，只做了以下不改变实验算法的兼容补丁：

1. 旧 checkpoint 是以 `__main__.EdgeScoringGCN` 整对象 pickle 保存的。PyTorch 2.6+ 默认 `weights_only=True`，原加载代码会失败。加载前注册旧类名，并对本包内可信 checkpoint 显式使用 `weights_only=False`。
2. PCP cover-DP 的 mask 可能是 `numpy.int64`，而原代码调用 `.bit_count()`；Python 3.9 也没有该方法。改为 `bin(int(mask)).count("1")`，得到完全相同的 popcount 和排序键。
3. Table 5 脚本只修改了包内路径默认值；核心 Z-fix、实例生成、模型、种子和求解逻辑不变。
4. Appendix C/D 的原临时 driver 未归档。`src/appendix/rerun_seed1.py` 从 seed10 通用 runner 重建，并按最终 provenance 改为 seed1、FCP 30 样本、PCP 独立 10 样本、K 30 样本、每次求解 60 秒上限。抽样集合已经与最终 1080/630 行 CSV 逐项核对。
5. Appendix E 的原 runner、cached-LP 源码和结果从历史版本恢复。公开版另将数据目录默认值改为当前目录、绘图库缓存改为包内路径；不改变求解逻辑。原源码哈希和公开版哈希记录在 `SOURCE_EXPORT.json`。guard 同步采用公开版哈希。重画脚本只把 JSON/输出目录改为包内相对路径。
6. 公开统一入口提供实验选择、独立输出目录、固定样本清单核对、完整 seed/sample 检查与数据驱动绘图。旧评估器的默认路径已改为最终公开数据和模型；训练 summary 的模型地址同步归一化。
7. `data_pipeline.py` 统一生成、标签和训练路径，连接随机数据的 train/eval/test manifest，并设置无界面绘图后端，支持 tmux 和 CI 环境。

关键原始 `Bundle_code` SHA-256 前缀：

| 文件 | SHA-256 前缀 |
|---|---|
| `Training_multi_layer.py` | `503731a08a1904c1` |
| `generate_data_bundle.py` | `3a1fbee1b272bbb3` |
| `generate_data_PCP_cp.py` | `0297b6405910de28` |
| `test_FCP.py` | `2ac79f45badf2bcf` |
| `test_PCP_cp.py` | `be1a6e84b79ad678` |
| `test_FCPLS_score_cached_lp.py` | `f84359f2e1fb7276` |
| `test_BSP.py` | `410388014ffa88d8` |

运行验证见 `docs/ACCEPTANCE.md` 与 `scripts/verify_reference_results.py`。
