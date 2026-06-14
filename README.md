# Lightweight HE-FL Runtime Experiments

联邦学习 + 同态加密（HE）轻量级运行时仿真实验框架。用于论文中 HE-FL 系统的性能评估、布局策略对比和泄漏分析。

---

## 项目结构

```text
hefl_runtime/
├── hefl/                          # 核心包
│   ├── config.py                  # 配置加载与合并
│   ├── data.py                    # 数据集加载与分区（IID / Dirichlet）
│   ├── fedavg.py                  # FedAvg 联邦平均训练
│   ├── layout.py                  # 密文布局规划（bundle / slot 分配）
│   ├── leakage.py                 # 元数据泄漏审计（最近质心攻击）
│   ├── metrics.py                 # 误差分解（pack / he / total）
│   ├── models.py                  # 模型定义（TinyCNN / TinyMLP / ResNet18）
│   ├── profiler.py                # 运行时 profiling
│   ├── reporting.py               # CSV / Markdown / 图表输出
│   ├── sim_he.py                  # SimHE 后端（模拟 CKKS 编码/加密/聚合）
│   ├── types.py                   # 数据类型定义
│   └── utils.py                   # 工具函数
├── configs/                       # 实验配置
│   ├── smoke.json                 # 快速冒烟测试
│   ├── cifar10_tinycnn_iid.json   # CIFAR-10 IID
│   ├── cifar10_tinycnn_dirichlet05.json
│   ├── cifar10_tinycnn_dirichlet01.json
│   ├── fashionmnist_tinymlp_dirichlet05.json
│   └── resnet18_cifar10_profile.json
├── run_experiment.py              # 单次实验入口
├── run_suite.py                   # 多配置多种子实验套件
├── aggregate_results.py           # 跨种子结果聚合
├── calibrate_he.py                # TenSEAL 真实 CKKS 微基准校准
├── summarize_results.py           # 结果目录重新汇总
├── run_autodl.sh                  # AutoDL 快速运行脚本
├── run_final_autodl.sh            # AutoDL 最终实验脚本
└── requirements-autodl.txt        # 额外依赖
```

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch + torchvision
- numpy, pandas, matplotlib, psutil

```bash
pip install torch torchvision numpy pandas matplotlib psutil
```

### 冒烟测试

```bash
python run_experiment.py --config configs/smoke.json
```

输出目录：`results/<timestamp>_smoke/`

### Windows 本地运行

```powershell
python.exe run_experiment.py --config configs\smoke.json
python.exe run_experiment.py --config configs\cifar10_tinycnn_iid.json
```

### AutoDL 云端运行

```bash
cd /root/autodl-tmp/crypto
bash experiments/hefl_runtime/run_final_autodl.sh
```

---

## 实验流程

### 单次实验

```bash
python run_experiment.py --config <config.json>
```

流程：`FedAvg → LayoutPlan → SimHEBackend → Profiler → LeakageAuditor → 输出`

### 完整实验套件（多配置 × 多种子）

```bash
# 1. 校准（可选，需安装 tenseal）
python calibrate_he.py --backend tenseal --out results/final/he_calibration.json

# 2. 运行套件
python run_suite.py --suite final_autodl \
    --out results/final \
    --data-root /root/autodl-tmp/data \
    --calibration-path results/final/he_calibration.json

# 3. 跨种子聚合
python aggregate_results.py --suite-dir results/final
```

### 重新汇总已有结果

```bash
python summarize_results.py --results results/<run_id>
```

---

## 配置说明

```jsonc
{
  "run_name": "smoke",              // 实验名称
  "seed": 2026,                     // 随机种子
  "device": "auto",                 // "auto" | "cpu" | "cuda"
  "dataset": {
    "name": "fake_cifar",           // "cifar10" | "fashionmnist" | "mnist" | "fake_cifar"
    "num_samples": 512,             // FakeData 样本数
    "batch_size": 64,
    "download": false,              // 是否下载真实数据集
    "allow_fake_fallback": true,    // 下载失败时是否回退到 FakeData
    "subset_train": null,           // 训练集子采样数（null 为全量）
    "subset_test": null
  },
  "model": {
    "name": "tiny_cnn"              // "tiny_cnn" | "tiny_mlp" | "resnet18"
  },
  "fed": {
    "num_clients": 4,
    "rounds": 2,
    "local_epochs": 1,              // 0 = synthetic update（仅 profiling）
    "partition": "iid",             // "iid" | "dirichlet"
    "dirichlet_alpha": 0.5,
    "clients_per_round": 4,
    "lr": 0.05
  },
  "layout": {
    "template_policy": "bucketed_fixed"  // 布局策略
  },
  "sim_he": {
    "quant_bits": 0,                // 量化位数（0 = 不量化）
    "clip_norm": null,              // 梯度裁剪范数（null = 不裁剪）
    "add_noise": true               // 是否添加 HE 噪声
  },
  "baselines": [
    "plain_fedavg",                 // 明文基线
    "simhe_layer_order",            // 按层顺序打包
    "simhe_manual_packed",          // 手动打包
    "simhe_fixed_template_runtime"  // 固定模板
  ],
  "experiment": {
    "seeds": [2026, 2027, 2028]     // 套件模式下多种子
  }
}
```

---

## 输出产物

每次运行在 `results/<timestamp>_<name>/` 下生成：

| 文件 | 说明 |
|------|------|
| `config.json` | 原始配置 |
| `config_resolved.json` | 完整配置（含环境、模型参数、分区信息） |
| `hardware.json` | 运行环境（Python/PyTorch/GPU） |
| `round_trace.csv` | 逐轮追踪（时延分解、通信量、误差） |
| `client_metrics.csv` | 逐客户端指标（loss、update_norm） |
| `error_decomposition.csv` | 误差分解（pack/he/total L2/Linf/MSE） |
| `metadata_features.csv` | 元数据特征（用于泄漏分析） |
| `summary.csv` | 按 baseline 聚合统计 |
| `layout_plan.json` | 密文布局方案 |
| `leakage_report.json` | 泄漏代理指标 |
| `run_summary.json` | 初始/最终评估结果 |
| `manifest.json` | 运行元信息与产物清单 |
| `report.md` | Markdown 汇总报告 |
| `*.png` | 可视化图表（时延分解、槽利用率、误差分解） |

---

## 基线对比

| 基线 | 说明 |
|------|------|
| `plain_fedavg` | 明文联邦平均，无 HE 开销 |
| `simhe_layer_order` | 按层顺序打包，tight bundle |
| `simhe_manual_packed` | 手动打包，tight bundle |
| `simhe_fixed_template_runtime` | 固定模板，桶对齐 bundle |
| `dynamic_template_or_selective` | 动态模板消融（需开启 `run_dynamic_template_ablation`） |

---

## CKKS 仿真参数

默认仿真配置：

| 参数 | 值 |
|------|-----|
| Scheme | CKKS-like |
| poly_modulus_degree (N) | 8192 |
| Slots | 4096 |
| coeff_mod_bit_sizes | [60, 40, 40, 60] |
| scale_bits | 40 |
| 密文大小 (ciphertext_size) | 2 |
| 原始密文字节 | 524,288 bytes/ct |
| 安全级别 | 128-bit nominal |

如需校准真实 CKKS 计时：

```bash
python calibrate_he.py --backend tenseal --out he_calibration.json \
    --poly-modulus-degree 8192 \
    --coeff-mod-bit-sizes "60,40,40,60" \
    --scale-bits 40 \
    --vector-length 4096 \
    --trials 5
```

---

## 依赖

核心依赖（需预先安装 PyTorch）：

```
numpy
pandas
matplotlib
psutil
```

可选依赖（用于真实 CKKS 校准）：

```
tenseal
```
