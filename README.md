---
title: 方寸 · 极客禅印章生成器
emoji: 🖋
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
---

# 方寸 · 极客禅印章生成器

Fangcun — Chinese Seal Generator for Geek-Zen

数字化传统金石艺术，代码刻出的方寸之美。

中国传统电子印章生成工具。从书法字库自动抓取字形，渲染出具有金石质感的朱文/白文印章。

## 样品 Gallery

| 禅（1字·白文·竖椭圆） | 苏轼（名章·白文·竖椭圆） | 极客禅（品牌章·白文·竖椭圆） | 天人合一（4字·白文·方章） |
|:---:|:---:|:---:|:---:|
| <img src="docs/samples/c01_zen_bw_oval.png" width="160"> | <img src="docs/samples/c11_sushi_bw_oval.png" width="160"> | <img src="docs/samples/c16_geekzen_bw_oval.png" width="160"> | <img src="docs/samples/c20_tianren_bw_sq.png" width="160"> |

| 卢修齐（名章·白文·方章） | 大观园（3字·朱文·竖椭圆） | 宇宙洪荒（4字·白文·方章） | 朝朝（重字·朱文·竖椭圆） |
|:---:|:---:|:---:|:---:|
| <img src="docs/samples/c18_luxiuqi_bw_sq.png" width="160"> | <img src="docs/samples/c23_daguan_zw_oval.png" width="160"> | <img src="docs/samples/宇宙洪荒_baiwen_square.png" width="160"> | <img src="docs/samples/朝朝_zhuwen_oval.png" width="160"> |

_均为自动生成，字形来自 ygsf.com 书法字库（统一字源，主要取自《中国篆刻大字典》《汉印文字征》等正统汉印印谱）。_

## 功能

- **朱白文渲染**：白文（红底白字实心填充）、朱文（透明底红字+红框，冲刀破边）
- **形制选择**：方章、竖椭圆（腰圆章，内圈双线框）
- **书法字源**：自动从 ygsf.com 抓取篆书/隶书/楷书（优先字典 tab，次选真迹 tab，不用字库）
- **同源同体**：同一方印章内所有字自动统一书体，并尽量选同一碑帖来源（5级来源统一策略）
- **繁体优先**：篆书形成于先秦，正体字是原生形态。严肃字典（中国篆刻大字典、汉印文字征）以正体字索引，fangcun 自动先查繁体再查简体
- **印章类型**：名章（强制篆书，不降级）、闲章（允许隶楷）、品牌章（任何字体）
- **智能字源选择**：结构完整度评分（防止碎片化异体字）、装饰性字源屏蔽（鸟虫篆）、单字印偏好汉印正统字源
- **印谱兼容**：自动识别印谱图片（鸟虫篆全书等），三层防御确保极性正确
- **金石质感**：6层石刻效果——按压不均、框边残破（壳层+断线双路径）、崩边、印泥颗粒、墨池加深、色温微变。朱白文走不同路径：白文按压改 RGB 亮度（保持 alpha 不透明），朱文按压调 alpha（允许淡化）；白文的纸色镂空区自动排除噪点/色偏/压力调制
- **智能排版**：1–4字自动竖排，极端扁平字（一二三）反向构造，笔画粗细归一化，视觉重心补偿
- **朱文冲边**：方章笔画穿过外框线（text_scale=0.98 + 4% bleed），椭圆章自动适配防裁剪
- **三级缓存**：API 响应缓存 + 图片 CDN 缓存 + 选定图缓存，二次生成零网络请求
- **双入口**：Gradio Web UI（交互）+ CLI（批量自动化 + 缓存管理 + 调试模式）
- **质量保证**：88 项单元 + 10 项回归 + 40 例视觉基线，纹理种子可复现（每次调用独立 `np.random.Generator`，无全局状态），确定性逐字节验证
- **类型化错误**：管线失败按场景细分（`UpstreamApiError`、`RateLimitedError`、`SourceInconsistencyError` 等），便于上游处理

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### Web UI（主入口）

```bash
python app.py
# → 打开 http://localhost:7860
```

### CLI 批量生成

```bash
# 单个
python cli.py --text "禅" --shape oval --style baiwen --type leisure

# 名章（强制篆书，不降级到隶楷）
python cli.py --text "卢修齐" --shape square --style baiwen --type name

# 批量（每行一个词，用于视频封面流水线）
python cli.py --batch chars.txt --shape oval --style baiwen --type leisure --output-dir ./seals/

# 批量并发（推荐 2-4 worker；每个 worker 独立 SealGenerator，过高易触发上游限流）
python cli.py --batch chars.txt --jobs 3 --output-dir ./seals/

# 缓存管理
python cli.py --cache-info     # 查看缓存统计
python cli.py --clear-cache    # 清除全部缓存
python cli.py --no-api-cache --text "禅"   # 跳过缓存

# 调试
python cli.py --debug --text "卢修齐" --type name  # 详细日志（候选列表、打分细节）
python cli.py --text "禅" --debug-extract   # 保存提取中间步骤（normalized/binary/denoised/cropped）；多字按 {idx}_{char}/ 嵌套
python cli.py --text "天人合一" --debug-layout  # 保存版面调试图（cell/ink bbox/centroid 叠加）

# 可复现纹理（同 seed → 字节相同输出）
python cli.py --text "禅" --seed 42 --output-dir ./out

# 严格一致性（仅接受 Level 1-2 统一来源，否则报错）
python cli.py --text "卢修齐" --type name --strict-consistency
```

### chars.txt 格式

```
禅
苏轼
极客禅
天人合一
```

## 项目结构

```
core/
├── __init__.py      # SealGenerator 统一入口
├── scraper.py       # ygsf.com 书法字抓取 + 5级来源统一 + 评分选图 + 3级缓存
├── extractor.py     # 三层极性归一化 + 二值化 + 两阶段印谱提取
├── layout.py        # 6阶段排版：适配 → 平衡 → 检测 → 反向构造 → 笔宽归一 → 重心放置
├── renderer.py      # 白文蒙版贴图 / 朱文 max-alpha 融合渲染 / 椭圆适配
└── texture.py       # 6层金石质感滤镜
app.py               # Gradio Web UI
cli.py               # CLI 批量入口 + 缓存管理 + --debug
```

## 印章类型

| 类型 | 字体优先级 | 降级 | 适用场景 |
|------|-----------|------|---------|
| 名章 `name` | 篆书 | 不允许 | 个人姓名印、落款印 |
| 闲章 `leisure` | 篆→隶→楷 | 允许 | 书斋印、座右铭、引首章 |
| 品牌章 `brand` | 篆→隶→楷 | 允许 | logo、社交媒体、装饰章 |

名章强制篆书是金石学传统——"篆刻"二字本身就暗示了这一点。

## 来源统一策略

多字印章自动寻找能覆盖所有字的单一书法来源（如中国篆刻大字典），确保笔墨风格统一：

1. **统一来源** (n=5) → 扩大搜索 (n=10) → **多数来源** (>50%覆盖) → **最小损失** → 各字最优
   - 延迟状态机：宁可降级到隶书同源，也不用篆书异源拼凑
2. 古奥/碎片字源自动降权，装饰性字源（鸟虫篆）名章完全屏蔽
3. 单字/重字印章自动短路 Pass 2，偏好汉印正统字源
4. 同源多变体笔画匹配：若某字在统一来源下存在粗细两版（知足场景），以兄弟字的相对笔画宽为锚点择优，保证通篇粗细协调

## 印谱兼容

来自印谱来源（如鸟虫篆全书、汉印分韵等）的图片与普通字帖极性相反。工具通过三层检测 + 两阶段提取自动处理：

**三层极性检测**：白名单 → 结构检测 → 形态学兜底

**两阶段提取**：聚合 Alpha CCA → 边缘内缩扫描

## 缓存

三级缓存架构，二次生成零网络请求：

| 层级 | 路径 | 内容 | TTL |
|------|------|------|-----|
| API 响应 | `~/.seal_gen/cache/_api/` | JSON 字形列表 | 正缓存 30 天，负缓存 7 天 |
| 图片 CDN | `~/.seal_gen/cache/_img/` | 候选图片 PNG | 无过期 |
| 选定图 | `~/.seal_gen/cache/{char}_{font}_{tab}.png` | 最终选中图 | 无过期 |

## 测试

98 项测试覆盖单字提取、版面、渲染、纹理、字源选择、错误类型、并发，外加 40 例视觉回归基线。

```bash
# 全部单元测试 + 回归测试
uv run python -m pytest tests/ -v

# 仅单元（快速，不联网）
uv run python -m pytest tests/ -m unit

# 仅回归（联网或命中缓存）
uv run python -m pytest tests/ -m regression

# 视觉回归套件（生成 40 例 + HTML 报告）
uv run python tests/regression/run.py --run-id baseline
# → tests/regression/output/baseline/report.html

# 对比两次运行（本次改动 vs baseline）
uv run python tests/regression/run.py --run-id my_change
uv run python tests/regression/compare.py baseline my_change
# → tests/regression/output/compare_baseline_vs_my_change.html
```

回归套件用 MD5 衍生的固定 seed 保证字节级可复现：相同代码 + 相同缓存 → 相同输出。

## 部署

```bash
# 本地运行
python app.py

# 部署到 Hugging Face Spaces（免费永久可访问）
# 1. 在 HF 新建 Space，选择 Gradio SDK
# 2. 上传所有文件
# 3. 自动构建，获得公开链接
```
