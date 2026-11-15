# stage-vol — 舞台体积光预览器

在程序生成的舞台、幕布与挡板场景中预览聚光灯束穿过均匀薄雾时的体积光效果。
距离单位为米，σ（雾消光系数）单位为 1/m；积分距离与步数均有硬上界。

## 图形入口（主要使用方式）

**材质预览窗口是舞台体积光的图形入口：**

```sh
cargo run --release --bin stage-vol-ui
```

（等同 `cargo build --release` 后运行 `./target/release/stage-vol-ui`。）

窗口内容：

- 左侧 **预览区**：实时光束遮挡与体积积分画面（Composite / 体积单独贡献 /
  阴影 / 不透明场景四种视图）。**在预览图上左右拖动可直接移动挡板**，
  挡板位置经内核 `set_control("occluder_x", …)` 校验后写入 GPU uniform。
- 右侧 **控件面板**（可滚动）：滑块由内核 control schema 生成，范围由内核
  约束；非法值被内核拒绝并在状态栏显示原因。雾密度 σ、积分步数 N、
  步长误差容差、空区跳过、灯强度/范围/锥角、挡板位置/尺寸等都在这里。
- **空区跳过（保守界）** 复选框：开 = 步进空区跳过；关 = 同预算均匀步进参照，
  便于对照同一状态下的灯束遮挡与积分画面。
- **运行光束成本基准** 按钮：在后台运行分级负载基准（完整帧 → 稀疏光束 →
  密度突变），结果显示最差亮度误差、样本节省与速度；失败信息为红色。
- **渲染参考帧**：以 `MAX_STEPS` 均匀积分在 CPU 上渲染参考帧到
  `out/ui_reference.png`。
- 状态栏同时显示帧耗时、设备与 GPU 校验/未捕获错误；出现异常时提示
  “灯束遮挡或积分画面可能异常：请降低参数或点击‘恢复设备资源’”。

需要有显示服务器与 GPU。无头环境请使用下面的命令行（**仅作辅助**）。

## 命令行（辅助）

`stage-vol-render` 用于无头测试、参考帧与成本基准，不替代图形预览：

```sh
cargo run --release --bin stage-vol-render -- test          # 既有场景流程 + 报告
cargo run --release --bin stage-vol-render -- reference -o out/ref.png
cargo run --release --bin stage-vol-render -- frame --scene scenes/default.json --view volume
cargo run --release --bin stage-vol-render -- bench --max-error 0.02
cargo run --release --bin stage-vol-render -- save-scenes   # 重写 scenes/*.json
cargo run --release --bin stage-vol-render -- model-notes
```

`--backend auto|cpu|gpu` 选择后端（auto 优先 GPU，失败回退 CPU 参考）。

## 步进空区跳过（本次优化）

### 保守界

每个启用的灯用 **锥（外锥角）∩ 球（range）** 作为光束影响范围的保守界。
`gobo` 遮片与遮挡只会*减少*光，因此该界是受光区域的严格超集；界内被遮片挡掉
或位于锥边缘的采样仍会求值并计为“空区样本”，所以跳过不会漏光。

相机光线与该界求交得到受光区间（凸集，每灯至多一段），区间补集以
**精确透过率更新** `T *= e^{−σΔt}` 跳过，不做光源求值。区间内步数按
`error_bound_steps(len, σ, tol) = ceil(len / sqrt(2·tol/σ²))` 的误差约束
逐区间分配，并对每段受光区间保底 `MIN_STEPS_PER_LIT_SPAN = 2` 步；
`steps` 仍是整条光线的预算，每段上限 `MAX_SPAN_STEPS`。

### 摘要失效

`InfluenceSummary` 以 `SummaryKey` 缓存，键包含每盏灯的位置/朝向/外锥角/
range/启用/intensity 位、雾 σ 位、`occluder_x_at(time)` 位、挡板可见性与
尺寸。灯光或介质（σ、挡板）改变或动画时间推进都会重建摘要——不会复用
过时范围而漏掉照明。`SummaryCache` 的 `rebuilds/reuses` 计数用于展示失效
确实发生。

### 单位不变

体积透过率仍按无单位累积（`T ∈ (0,1]`）；遮挡查询的最大距离仍为米制
（`d − SHADOW_EPS`）。跳过只改变**求值位置与顺序**，不改变任何单位。

### 计算顺序的改变

均匀参照：整条光线按 `steps` 等分，逐段 `L += T·S·(1−e^{−σΔt})`、`T *= e^{−σΔt}`。

空区跳过：先求每灯保守界的相机光线区间，排序合并；对每个受光区间按误差
约束定步数并在区间内求值，区间之间只更新透过率（无光源求值）。因此
**采样点落在区间内部而非整条光的均匀网格上**；当区间覆盖整条光线时与参照
等价。误差容差 `1.0`（参照路径）退化为整条光线的均匀网格。

### 成本基准与执行证据

`stage-vol-render bench`（或 UI 按钮）分级增加负载：

1. `full-frame`：默认步数的整帧（原负载）；
2. `sparse-beam`：低步数，光束只占画面小部分；
3. `density-jump`：同一预算下 σ=0.03 与 σ=0.35 的密度突变。

每档记录实际执行步数（`CostBreakdown`：样本、空区样本、高密度样本、跳过
距离、遮挡查询、透过率更新、区间数）、相对 `MAX_STEPS` 收敛参照的亮度误差
（跳过路径**与同预算均匀路径分别测量**）、以及两条路径的耗时（7 次交错取
最小值）。结果必须携带执行证据（`rays/samples/baseline_samples` 非零）才会
报告速度；空基准、缺证据、误差超限或跳过路径劣于同预算均匀路径超过
`0.005` 一律判失败（`validate_benchmark`）。

> 复核记录：初版基准在 `sparse-beam` 档出现 0.55×（跳过后更慢），定位为
> 每光线堆分配与单次计时抖动，而非样本节省不足。改为无分配区间求解并采用
> 交错 best-of-7 计时后，各档稳定为 1.40×–1.72×，样本数/误差不变，输出帧
> 逐字节一致。

## 模型边界

单散射、各向同性相位、均匀介质的近似模型（详见 `stage-vol-render model-notes`
与 `src/lib.rs` 文档）。适合灯光设计预览与参数调试，不替代支持非均匀介质、
多次散射或物理相函数的完整体积渲染器。
