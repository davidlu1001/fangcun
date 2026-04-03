# 极客禅 · 印章生成器

中国传统电子印章生成工具。从书法字库自动抓取字形，渲染出具有金石质感的朱文/白文印章。

## 功能

- **朱白文渲染**：白文（红底白字）、朱文（透明底红字+红框）
- **形制选择**：方章、竖椭圆（腰圆章，内圈双线框）
- **书法字源**：自动从 ygsf.com 抓取篆书/隶书/楷书，按章类优先级自动降级
- **金石质感**：边框毛边、笔画崩边、印泥颗粒感，可调节强度
- **传统排版**：1–4字自动竖排，遵循从右到左阅读顺序
- **双入口**：Gradio Web UI（交互）+ CLI（批量）

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

# 批量（每行一个词）
python cli.py --batch chars.txt --shape oval --style baiwen --type leisure --output-dir ./seals/
```

### chars.txt 格式

```
禅
苏轼
极客禅
```

## 项目结构

```
core/
├── __init__.py      # SealGenerator 统一入口
├── scraper.py       # ygsf.com 书法字抓取 + 缓存
├── extractor.py     # 字形二值化提取
├── layout.py        # 传统竖排排版
├── renderer.py      # 白文/朱文渲染
└── texture.py       # 金石质感滤镜
app.py               # Gradio Web UI
cli.py               # CLI 批量入口
```

## 字体选择规则

| 章类 | 优先级 | 禁止 |
|------|--------|------|
| 闲章 | 篆书 → 隶书 → 楷书 | 行书、草书 |
| 名章 | 隶书 → 楷书 | 草书、行书 |

字体找不到时自动降级，UI/CLI 中会显示降级提示。

## 部署到 Hugging Face Spaces

```bash
# 1. 在 HF 新建 Space，选择 Gradio SDK
# 2. 上传所有文件
# 3. 自动构建，获得公开链接
```

## 缓存

抓取的书法图片缓存在 `~/.seal_gen/cache/`，避免重复请求。
