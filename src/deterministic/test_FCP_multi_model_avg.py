"""
基于 test_FCP 的多模型评估脚本：
- 支持指定层数（如 2/3/4）和种子列表（默认 1-10），加载对应训练好的模型
- 对每个样本：逐模型独立推理并 MILP 求解，再对该样本的 revenue/time 取平均
- 模型路径默认使用新的多层训练输出目录 models_multi_layer_edge_update
"""

from __future__ import annotations
import os
import time
import argparse
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# 复用 test_FCP 中的模型结构与数据处理 / MILP 求解
from test_FCP import EdgeScoringGCN, process_data, revenue_ratio

# 安全加载 EdgeScoringGCN
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([EdgeScoringGCN])


def parse_list_arg(arg: str) -> List[int]:
    """将逗号分隔的整数字符串解析为列表."""
    return [int(x) for x in arg.split(",") if x.strip()]


def parse_paths_arg(arg: str) -> List[str]:
    """将分号分隔的路径字符串解析为列表."""
    return [p.strip() for p in arg.split(";") if p.strip()]


def _save_layer_result(nl: int, m: int, n: int, results_list: list, result_dir: str, test_folder_name: str = None) -> None:
    """保存单个层级和问题规模的结果到 CSV 文件."""
    import pandas as pd
    
    results_array = np.array(results_list)
    
    # 文件名: test_result_FCP_Nlayer_{test_folder_name}.csv 或 test_result_FCP_Nlayer_m{m}n{n}.csv
    if test_folder_name:
        result_filename = f"test_result_FCP_{nl}layer_{test_folder_name}.csv"
    else:
        result_filename = f"test_result_FCP_{nl}layer_m{m}n{n}.csv"
    result_path = os.path.join(result_dir, result_filename)
    
    # 记录详细信息 - 使用 pd.DataFrame 来避免 format specifier 问题
    df = pd.DataFrame({
        'method': 'FCP',
        'layers': nl,
        'm_segments': m,
        'n_products': n,
        'revenue_ratio': results_array[:, 0],
        'runtime_ratio': results_array[:, 1],
        'avg_total_time': results_array[:, 2],
        'avg_solve_time': results_array[:, 3],
        'avg_gcn_time': results_array[:, 4],
    })
    df.to_csv(result_path, index=False)
    print(f"✅ 已保存: {result_path} ({len(results_array)} samples)")


def _save_seed_averages(nl: int, m: int, n: int, seed_results: dict, result_dir: str, test_folder_name: str = None) -> None:
    """
    保存每个seed对所有数据的平均结果到 CSV 文件.
    
    Args:
        nl: 层数
        m: 段数
        n: 产品数
        seed_results: {seed: [(revenue_ratio, time_ratio, total_time, solve_time, gcn_time), ...]}
        result_dir: 结果保存目录
        test_folder_name: 测试文件夹名称
    """
    import pandas as pd
    
    # 计算每个seed的平均值
    seed_avg_data = []
    for seed, results in sorted(seed_results.items()):
        if len(results) > 0:
            results_array = np.array(results)
            seed_avg_data.append({
                'seed': seed,
                'num_samples': len(results),
                'avg_revenue_ratio': float(np.mean(results_array[:, 0])),
                'std_revenue_ratio': float(np.std(results_array[:, 0])),
                'avg_time_ratio': float(np.mean(results_array[:, 1])),
                'std_time_ratio': float(np.std(results_array[:, 1])),
                'avg_total_time': float(np.mean(results_array[:, 2])),
                'avg_solve_time': float(np.mean(results_array[:, 3])),
                'avg_gcn_time': float(np.mean(results_array[:, 4])),
            })
    
    if not seed_avg_data:
        return
    
    # 文件名: test_result_FCP_Nlayer_{test_folder_name}_seed_avg.csv
    if test_folder_name:
        result_filename = f"test_result_FCP_{nl}layer_{test_folder_name}_seed_avg.csv"
    else:
        result_filename = f"test_result_FCP_{nl}layer_m{m}n{n}_seed_avg.csv"
    result_path = os.path.join(result_dir, result_filename)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(seed_avg_data)
    
    # 添加元数据列
    df.insert(0, 'method', 'FCP')
    df.insert(1, 'layers', nl)
    df.insert(2, 'm_segments', m)
    df.insert(3, 'n_products', n)
    
    df.to_csv(result_path, index=False)
    print(f"✅ 已保存每个seed平均结果: {result_path} ({len(seed_avg_data)} seeds)")


def _save_seed_sample_results(rows: list, result_dir: str, test_folder_name: str) -> str | None:
    """保存每个 seed 、每个样本的可审计长表。"""
    if not rows:
        return None

    import pandas as pd

    columns = [
        "method",
        "layers",
        "seed",
        "sample_file",
        "m_segments",
        "n_products",
        "revenue_ratio",
        "runtime_ratio",
        "total_time",
        "solve_time",
        "gcn_time",
    ]
    result_filename = f"test_result_FCP_{rows[0]['layers']}layer_{test_folder_name}_seed_sample.csv"
    result_path = os.path.join(result_dir, result_filename)
    pd.DataFrame(rows, columns=columns).to_csv(result_path, index=False)
    print(f"✅ 已保存 seed×sample 长表: {result_path} ({len(rows)} rows)")
    return result_path


def load_models(
    model_root: str,
    layers: List[int],
    seeds: List[int],
    device: torch.device,
) -> dict:
    """加载指定层数和种子的模型集合。返回 {layer: [(seed, model, path), ...]}."""
    loaded = {nl: [] for nl in layers}
    for nl in layers:
        for sd in seeds:
            cand_paths = [
                os.path.join(model_root, f"best_model_edge_{nl}layer_seed{sd}.pt"),
                os.path.join(model_root, f"model_edge_{nl}layer_seed{sd}.pt"),
                os.path.join(model_root, f"model-{nl}layer_seed{sd}.pt"),
            ]
            path = next((p for p in cand_paths if os.path.exists(p)), None)
            if path is None:
                print(f"⚠️ 未找到模型: layer={nl}, seed={sd}, searched={cand_paths}")
                continue
            try:
                import __main__
                __main__.EdgeScoringGCN = EdgeScoringGCN
                mdl = torch.load(path, map_location=device, weights_only=False)
                mdl.to(device)
                mdl.eval()
                loaded[nl].append((sd, mdl, path))
                print(f"✅ 已加载模型: layer={nl}, seed={sd}, path={path}")
            except Exception as e:
                print(f"❌ 加载失败 layer={nl}, seed={sd}, path={path}: {e}")
    return loaded


def run_one_model(mdl, data, n, m_segments):
    """单模型 forward，并返回阈值后的组合；阈值时间不计入结果."""
    with torch.no_grad():
        inference_start = time.time()
        raw_out = mdl(data)
        inference_time = time.time() - inference_start

        if isinstance(raw_out, dict):
            if "logit_matrix" in raw_out:
                logits_nm = raw_out["logit_matrix"].detach().cpu().numpy()
            elif "edge_logits" in raw_out:
                s = raw_out["edge_logits"].detach().cpu().numpy()
                logits_nm = s.reshape(n, m_segments)
            else:
                raise ValueError("Unexpected model output keys: " + ",".join(raw_out.keys()))
            pred_assort = (logits_nm.T >= 0.0).astype(int)
        else:
            output = raw_out[:n, :].detach().cpu().numpy()
            pred_assort = (np.exp(output) / (np.exp(output) + np.exp(1)) >= 0.5).astype(int).T
    return pred_assort, inference_time


def main():
    parser = argparse.ArgumentParser(description="多模型平均评估（基于 test_FCP）")
    parser.add_argument("--data_dir", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    parser.add_argument("--test_subdirs", type=str, default="data/deterministic/test_m10n10_1e_3;data/deterministic/test_m20n10_1e_3;data/deterministic/test_m30n10_1e_3", 
                        help="测试数据子目录（多个用分号分隔）")
    parser.add_argument     ("--model_dir", type=str, default="models/base_4layer", help="模型目录")
    parser.add_argument("--layers", type=str, default="4", help="要使用的层数列表, 逗号分隔")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10", help="要使用的seed列表, 逗号分隔")
    parser.add_argument("--result_dir", type=str, default="results/fcp", help="结果保存目录")
    parser.add_argument("--save_result", type=bool, default=True, help="是否保存结果文件")
    args = parser.parse_args()

    dir_path = args.data_dir
    test_subdirs = parse_paths_arg(args.test_subdirs)
    model_root = os.path.join(dir_path, args.model_dir)
    result_dir = os.path.join(dir_path, args.result_dir) if args.save_result else None

    layers = parse_list_arg(args.layers)
    seeds = parse_list_arg(args.seeds)

    # 创建结果目录
    if args.save_result:
        os.makedirs(result_dir, exist_ok=True)

    print(f"📂 data_dir: {dir_path}")
    print(f"🧪 test_subdirs: {test_subdirs}")
    print(f"🗂 model_root: {model_root}")
    print(f"🔢 layers: {layers}")
    print(f"🌱 seeds: {seeds}")
    if args.save_result:
        print(f"💾 result_dir: {result_dir}")
    else:
        print(f"💾 不保存结果文件")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载模型（按层组织）
    models_by_layer = load_models(model_root, layers, seeds, device)
    if not any(models_by_layer.values()):
        print("未加载到任何模型，退出。")
        return

    # 逐个数据集测试
    for test_subdir in test_subdirs:
        # 为每个数据集单独创建结果存储（避免累积）
        # 结构: {layer: {(m, n): [(revenue_ratio, time_ratio, total_time, solve_time, gcn_time), ...]}}
        results_by_layer = {nl: {} for nl in layers}
        
        # 按层和seed存储结果（每个seed单独的结果）
        # 结构: {layer: {seed: {(m, n): [(revenue_ratio, time_ratio, total_time, solve_time, gcn_time), ...]}}}
        results_by_seed = {nl: {sd: {} for sd, _, _ in models_by_layer[nl]} for nl in layers}
        seed_sample_rows_by_layer = {nl: [] for nl in layers}
        test_data_path = os.path.join(dir_path, test_subdir)
        
        if not os.path.exists(test_data_path):
            print(f"⚠️ 测试数据路径不存在: {test_data_path}，跳过")
            continue

        dir_list = os.listdir(test_data_path)
        test_dataset = []
        misc_dataset = []
        file_names = []

        print(f"\n📊 开始读取测试集: {test_subdir}")
        for fname in dir_list:
            if fname == ".DS_Store":
                continue
            fpath = os.path.join(test_data_path, fname)
            try:
                dat, misc = process_data(fpath)
                test_dataset.append(dat)
                misc_dataset.append(misc)
                file_names.append(fname)
            except Exception as e:
                print(f"读取 {fname} 失败: {e}")
                continue

        sample_num = len(test_dataset)
        print(f"✅ 共加载 {sample_num} 条样本")
        if sample_num == 0:
            continue

        for i in tqdm(range(sample_num), desc=f"Evaluating {test_subdir}"):
            try:
                dat = test_dataset[i].to(device)
                misc = misc_dataset[i]
                n, m_segments, unit_cs, ship_cs, unit_us, Ns, opt_bundles, opt_prices, opt_rev, running_time, gap = misc
                
                # 对每一层，使用该层的所有seed进行推理
                for nl in layers:
                    if nl not in models_by_layer or not models_by_layer[nl]:
                        continue
                    
                    model_ratios = []
                    model_time_ratios = []
                    model_total_times = []
                    model_solve_times = []
                    model_gcn_times = []

                    for sd, mdl, path in models_by_layer[nl]:
                        try:
                            pred_assort, gcn_time = run_one_model(mdl, dat, n, m_segments)

                            ratio, solve_time = revenue_ratio(n, m_segments, unit_cs, ship_cs, unit_us, Ns, opt_rev, pred_assort)
                            total_time = gcn_time + solve_time
                            time_ratio = total_time / running_time if running_time > 0 else float("inf")

                            model_ratios.append(ratio)
                            model_time_ratios.append(time_ratio)
                            model_total_times.append(total_time)
                            model_solve_times.append(solve_time)
                            model_gcn_times.append(gcn_time)
                            
                            # 记录每个seed的单独结果
                            key = (m_segments, n)
                            if key not in results_by_seed[nl][sd]:
                                results_by_seed[nl][sd][key] = []
                            results_by_seed[nl][sd][key].append([ratio, time_ratio, total_time, solve_time, gcn_time])
                            seed_sample_rows_by_layer[nl].append({
                                "method": "FCP",
                                "layers": nl,
                                "seed": sd,
                                "sample_file": file_names[i],
                                "m_segments": m_segments,
                                "n_products": n,
                                "revenue_ratio": float(ratio),
                                "runtime_ratio": float(time_ratio),
                                "total_time": float(total_time),
                                "solve_time": float(solve_time),
                                "gcn_time": float(gcn_time),
                            })
                            
                        except Exception as e_model:
                            print(f"模型失败 sample={file_names[i] if i < len(file_names) else i}, layer={nl}, seed={sd}, path={path}: {e_model}")
                            continue

                    if len(model_ratios) > 0:
                        key = (m_segments, n)
                        if key not in results_by_layer[nl]:
                            results_by_layer[nl][key] = []
                        
                        result_entry = [
                            float(np.mean(model_ratios)),
                            float(np.mean(model_time_ratios)),
                            float(np.mean(model_total_times)),
                            float(np.mean(model_solve_times)),
                            float(np.mean(model_gcn_times)),
                        ]
                        results_by_layer[nl][key].append(result_entry)
            except Exception as e:
                print(f"样本 {file_names[i] if i < len(file_names) else i} 评估失败: {e}")
                continue

        # 本数据集处理完后，保存结果
        test_folder_name = os.path.basename(test_subdir.rstrip('/'))
        if args.save_result:
            print(f"\n💾 保存 {test_folder_name} 的结果...")
            for nl in layers:
                for key, results in results_by_layer[nl].items():
                    if results:
                        m, n = key
                        # 保存样本级别的平均结果
                        _save_layer_result(nl, m, n, results, result_dir, test_folder_name)
                        
                        # 保存每个seed的平均结果
                        seed_results_for_key = {
                            sd: results_by_seed[nl][sd].get(key, [])
                            for sd in results_by_seed[nl].keys()
                        }
                        _save_seed_averages(nl, m, n, seed_results_for_key, result_dir, test_folder_name)
                        _save_seed_sample_results(
                            seed_sample_rows_by_layer[nl],
                            result_dir,
                            test_folder_name,
                        )
        
        # 打印当前数据集的统计结果
        print(f"\n{'='*80}")
        print(f"📊 {test_folder_name} 评估结果")
        print(f"{'='*80}")
        for nl in layers:
            if results_by_layer[nl]:
                total_samples = sum(len(v) for v in results_by_layer[nl].values())
                print(f"\n【Layer {nl}】样本数: {total_samples}")
                for (m, n), results in sorted(results_by_layer[nl].items()):
                    if results:
                        results_array = np.array(results)
                        print(f"  m={m}, n={n}: {len(results_array)} samples")
                        print(f"    Revenue Ratio - Mean: {np.mean(results_array[:, 0]):.4f}, Std: {np.std(results_array[:, 0]):.4f}, "
                              f"Min: {np.min(results_array[:, 0]):.4f}, Max: {np.max(results_array[:, 0]):.4f}")
                        print(f"    Time Ratio - Mean: {np.mean(results_array[:, 1]):.4f}, Std: {np.std(results_array[:, 1]):.4f}, "
                              f"Min: {np.min(results_array[:, 1]):.4f}, Max: {np.max(results_array[:, 1]):.4f}")
                        print(f"    Avg Times - Total: {np.mean(results_array[:, 2]):.4f}s, Solve: {np.mean(results_array[:, 3]):.4f}s, GCN: {np.mean(results_array[:, 4]):.4f}s")
        print(f"{'='*80}\n")

    # 最终汇总统计
    print("\n" + "="*80)
    print("🎉 所有数据集评估完成！")
    print("="*80)
    print(f"测试数据集数: {len(test_subdirs)}")
    print(f"层数: {layers}")
    print(f"每层模型数: {len(seeds)}")
    if args.save_result:
        print(f"结果保存目录: {result_dir}")
    else:
        print(f"结果文件: 未保存")
    print("="*80)


if __name__ == "__main__":
    main()
