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

## 功能

- **朱白文渲染**：白文（红底白字实心填充）、朱文（透明底红字+红框，冲刀破边）
- **形制选择**：方章、竖椭圆（腰圆章，内圈双线框）
- **书法字源**：自动从 ygsf.com 抓取篆书/隶书/楷书（优先字典 tab，次选真迹 tab，不用字库）
- **同源同体**：同一方印章内所有字自动统一书体，并尽量选同一碑帖来源（5级来源统一策略）
- **繁体优先**：篆书形成于先秦，正体字是原生形态。严肃字典（中国篆刻大字典、汉印文字征）以正体字索引，fangcun 自动先查繁体再查简体
- **印章类型**：名章（强制篆书，不降级）、闲章（允许隶楷）、品牌章（任何字体）
- **智能字源选择**：结构完整度评分（防止碎片化异体字）、装饰性字源屏蔽（鸟虫篆）、单字印偏好汉印正统字源
- **印谱兼容**：自动识别印谱图片（鸟虫篆全书等），三层防御确保极性正确
- **金石质感**：6层石刻效果——按压不均、框边残破（壳层+断线双路径）、崩边、印泥颗粒、墨池加深、色温微变
- **智能排版**：1–4字自动竖排，极端扁平字（一二三）反向构造，笔画粗细归一化，视觉重心补偿
- **朱文冲边**：方章笔画穿过外框线（text_scale=0.98 + 4% bleed），椭圆章自动适配防裁剪
- **三级缓存**：API 响应缓存 + 图片 CDN 缓存 + 选定图缓存，二次生成零网络请求
- **双入口**：Gradio Web UI（交互）+ CLI（批量自动化 + 缓存管理 + 调试模式）

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

# 缓存管理
python cli.py --cache-info     # 查看缓存统计
python cli.py --clear-cache    # 清除全部缓存
python cli.py --no-api-cache --text "禅"   # 跳过缓存

# 调试
python cli.py --debug --text "卢修齐" --type name  # 详细日志
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

## 部署

```bash
# 本地运行
python app.py

# 部署到 Hugging Face Spaces（免费永久可访问）
# 1. 在 HF 新建 Space，选择 Gradio SDK
# 2. 上传所有文件
# 3. 自动构建，获得公开链接
```
