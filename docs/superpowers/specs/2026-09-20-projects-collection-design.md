# Projects Collection 设计文档

日期：2026-09-20
状态：已确认，待实现

## 1. 目标

在个人主页中新增一个可扩展的 `projects` collection，用于展示工程型 / 系统型 demo。首个条目是作者在 Mercedes-Benz AG 主导的一个 demo，素材（视频、图片、文字）由作者后续提供。

## 2. 背景与约束

### 2.1 站点现状

本站基于 al-folio 主题，已被改造为自定义的暗色编辑风格视觉系统：

- 背景 `#080706`，强调色 `#b89278`，扁平排版、细分隔线。
- `README.md` 明确记录设计原则："Flat section rhythm with thin dividers instead of heavy rounded cards."
- 页面宽度 `max_width: 930px`。
- 样式 token 位于 `_sass/_variables.scss` 与 `_sass/_themes.scss`，样式入口为 `assets/css/main.scss`。

现有导航：Home（`/`）、Publications（`nav_order: 1`）、Blogs（`nav_order: 2`）、CV（`nav_order: 3`）。`_pages/repositories.md` 存在但 `nav: false`。

### 2.2 al-folio 原有的 projects collection

原模板确实带有 `projects` collection，在 commit `f9a5162`（"Clean up template remnants and deployment setup"）中被整体删除。原结构为：

| 组成                                   | 内容                                                                                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `_config.yml`                          | `collections.projects.output: true`，开关 `enable_project_categories`                                                                 |
| `_pages/projects.md`                   | `layout: page`、`permalink: /projects/`、`nav_order: 3`、`display_categories: [work, fun]`、`horizontal: false`                       |
| `_projects/*.md`                       | `layout: page`、`title`、`description`、`img`、`importance`、`category`、`redirect`、`github`、`github_stars`、`related_publications` |
| `_includes/projects.liquid`            | 三列卡片网格                                                                                                                          |
| `_includes/projects_horizontal.liquid` | 两列横向卡片                                                                                                                          |
| 详情页                                 | 无专用 layout，复用 `layout: page`                                                                                                    |
| 输出 URL                               | `/projects/<filename>/`                                                                                                               |

### 2.3 核实过的技术事实

以下三点在设计前经代码核实，直接影响了设计取舍：

1. **不依赖 collection 级 `defaults`。** `_config.yml` 中 `collections.news.defaults.layout: post` 的写法是否生效，无法从现有代码证明——`_news/announcement_1.md` 本身就显式写了 `layout: post`。因此 `_projects/*.md` 一律显式声明 `layout: project`，与 `_news` 的实际做法保持一致。
2. **不恢复 `github` / `github_stars` 字段。** 对应的 GitHub stars 前端脚本已不在仓库中（`_includes/scripts.liquid` 与 `assets/js/` 均无相关代码），字段恢复后不会有任何效果。改用通用的 `links` 数组承载全部外链。
3. **`_sass/_base.scss` 第 726–763 行的 `.projects` 块是死代码。** 全仓库已无任何文件使用 `class="projects"`。新样式另起 `.project-list` 命名空间，并在本次改动中删除这段死代码。

另外核实到：`.card.hoverable` 在 `_sass/_pages.scss:559` 已被重新配色为暗色风格，但该样式当前仅服务于博客的 featured-posts，本设计不使用卡片网格，因此不涉及。

### 2.4 收录边界

三个板块职责互不重叠：

- **Publications** —— BibTeX 驱动的论文条目。
- **Blogs** —— 论文解读与技术博客。
- **Projects** —— 工程型 / 系统型 demo、原型、工具链。

论文本身不进入 Projects。SpaceDrive、AGO、PowerBEV 作为论文，归 Publications；其解读归 Blogs。

需要区分的一种情况：**同一研究方向的工程化产物属于 Projects**。例如 `papers.bib` 中的 `li2025ago`（ICCV 2025 的 AGO 方法论文）归 Publications，而基于该方向在公司内部落地的自动标注流水线是可规模化的工程系统，归 Projects。两者不是同一个条目。为避免读者混淆，Projects 中的条目名应体现工程属性，不与论文同名。

语言：Projects 只做英文。数据模型中不设语言字段，也不做 `_posts/` 那样的中英双目录结构。

## 3. 数据模型

每个条目为 `_projects/<slug>.md`，front matter 如下：

```yaml
---
layout: project # 显式声明，不依赖 collection defaults
title: <Demo 名称>
description: <一句话定位；列表页摘要与详情页副标题共用>
category: Demo # 作为 eyebrow 标签显示，不做分组
importance: 1 # 列表排序，数值小的在前
img: assets/img/project_images/<slug>/thumbnail.jpg
year: 2026
org: Mercedes-Benz AG
role: Project Lead
stack: [PyTorch, CUDA, ROS 2]
links:
  - name: Project Page
    url: https://example.com
giscus_comments: false
redirect: # 可选；填写后列表项直接跳转外链，不生成详情页内容
---
```

正文为普通 Markdown，多媒体通过现有的 `_includes/figure.liquid` 与 `_includes/video.liquid` 插入。

字段取舍依据：

- `title`、`description`、`img`、`importance`、`category`、`redirect` 沿用 al-folio 原有约定，便于日后与上游对照。
- `year`、`org`、`role`、`stack`、`links` 为本场景（公司主导的 demo）所必需的结构化信息。放入 front matter 由 layout 统一渲染，避免每个条目在正文里重复手写格式。
- `category` 不作为分组键，只作为 eyebrow 标签渲染。当前只有一个类别，做分组是空架子。

## 4. 文件改动清单

| 文件                                | 动作                                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `_config.yml`                       | 新增 `collections.projects`（`output: true` 与 `permalink: /:collection/:path/`）；在 `exclude` 中加入 `docs` 与 `tools` |
| `_pages/projects.md`                | 新建。`layout: page`、`permalink: /projects/`、`nav: true`、`nav_order: 3`                                               |
| `_pages/cv.md`                      | `nav_order` 由 `3` 改为 `4`                                                                                              |
| `_layouts/project.liquid`           | 新建。详情页 layout                                                                                                      |
| `_includes/project_entry.liquid`    | 新建。列表页单条目                                                                                                       |
| `_sass/_projects.scss`              | 新建                                                                                                                     |
| `assets/css/main.scss`              | `@import` 列表中在 `"pages"` 之后加入 `"projects"`                                                                       |
| `_sass/_base.scss`                  | 删除第 726–763 行的死 `.projects` 块                                                                                     |
| `assets/img/project_images/<slug>/` | 新建资源目录，组织方式对齐现有 `assets/img/blog_images/<slug>/`                                                          |

`docs` 必须加入 `_config.yml` 的 `exclude`：Jekyll 默认会构建所有非 `_` / `.` 开头的顶层目录，否则本设计文档会被发布到 `https://edwardleelpz.github.io/docs/` 下。

collection 输出 URL 为 `/projects/<slug>/`，**需要在 collection 上显式配置 `permalink: /:collection/:path/`**。原先认为可依赖 Jekyll 默认值，实测不成立：`_config.yml:85` 的全局 `permalink:` 为空值，会使本 collection 输出 `/projects/<slug>.html`。该全局设置服务于博客文章，不作改动。

## 5. 页面排版

### 5.1 列表页 `/projects/`

横向条目流，扁平、细分隔线，对齐首页与 CV 的编辑风格，不使用卡片网格：

```
Projects
Engineering demos, system prototypes, and internal tooling.
─────────────────────────────────────────────
              DEMO
 [缩略图]     <Demo 名称>                 2026
              一句话定位描述文字
              Mercedes-Benz AG · Project Lead
─────────────────────────────────────────────
```

条目按 `importance` 升序排列。条目链接目标：若 `redirect` 存在则跳该外链，否则跳 `/projects/<slug>/`。

### 5.2 详情页 `/projects/<slug>/`

由 `_layouts/project.liquid` 固定渲染：

```
DEMO
<Demo 名称>
一句话定位
─────────────────────────────────────────────
Year    2026
Org     Mercedes-Benz AG
Role    Project Lead
Stack   PyTorch · CUDA · ROS 2
Links   Project Page ↗
─────────────────────────────────────────────
[正文：视频 / 图 / 文字]
```

元信息栏中每个字段均由 `{% if %}` 包裹，字段缺失时不渲染对应行，不产生空行。条目之间不要求字段齐全。

`_layouts/project.liquid` 结构：

```liquid
---
layout: default
---
<div class="project">
  <header class="project-header">...</header>
  <dl class="project-meta">...</dl>
  <article class="project-body">{{ content }}</article>
  {% if site.giscus and page.giscus_comments %}{% include giscus.liquid %}{% endif %}
</div>
```

## 6. 媒体与仓库体积策略

仓库当前未启用 Git LFS，`.git` 已达约 322 MB，`assets/` 约 120 MB。因此新素材一律按下列规格入库。

**图片一律使用 WebP，不使用 PNG。** 正文栏宽为 `--site-content-width: 1080px`，因此图片宽度上限取 1860px（约 2x）即可，更大无收益。质量档按内容类型分流，这是实测结论而非经验判断：

| 内容类型                         | 命令                                                                   | 实测效果                                    |
| -------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------- |
| 渲染截图、可视化、照片           | `convert src.png -resize '1860x>' -quality 85 out.webp`                | 相对原 PNG 省 95–99%                        |
| 烧有小号文字标签的面板、表格类图 | `convert src.png -resize '1860x>' -define webp:lossless=true out.webp` | 零振铃伪影，且对纯色为主的图往往比 q85 更小 |

实测样本（本仓库现有图片）：`spacedrive_architecture.png` 1392×768，8.77 MB → 0.11 MB；`spacedrive_teaser.png` 2953×987，10.21 MB → 0.16 MB；`closeloop_eval.png` 文字密集，q85 为 0.18 MB 而无损仅 0.10 MB。

WebP 已端到端验证：Jekyll 正常输出到构建产物，HTTP 返回 `image/webp`；`_includes/figure.liquid` 的 `<img src>` 直接透传 `path`，因 `imagemagick.enabled: false`，其 `<picture><source>` 分支整块跳过，不受影响。

其余规则：

- 文件名不得以 `_` 或 `.` 开头，Jekyll 会静默忽略这类文件且不报错。
- 视频用 mp4（h.264），由 `_includes/video.liquid` 渲染，参数 `autoplay` + `muted` + `loop` + `controls`；封面从首帧抽取后转 WebP，控制在 150 KB 以内。不使用 GIF，同内容下 mp4 严格优于 GIF。
- 单文件硬上限 5 MB。超长演示视频改用外链，`video.liquid` 对非 mp4/webm/ogg 路径自动走 iframe。
- GitHub 单文件硬上限 100 MB，仓库整体超过 1 GB 触发告警。

## 7. 交互能力

本节在实现阶段被推翻重写。原先判断"交互式能力需要另行开发，提前做只会产生死代码"，该判断基于未核实 `_includes/scripts.liquid` 的错误前提。

实际情况：`_layouts/default.liquid` 会 include `_includes/scripts.liquid`，而其中大量能力由 **page front matter 开关**驱动。`_layouts/project.liquid` 继承 `default`，因此以下能力对 `_projects/*.md` 开箱可用，无需任何新代码：

| 能力               | front matter 开关                                        | 版本 / 依赖                 |
| ------------------ | -------------------------------------------------------- | --------------------------- |
| 图片对比滑块       | `images: {compare: true}`                                | img-comparison-slider 8.0.6 |
| 图片轮播           | `images: {slider: true}`                                 | swiper                      |
| 灯箱               | `images: {photoswipe/lightbox2/spotlight/venobox: true}` | 四选一                      |
| Chart.js 图表      | `chart: {chartjs: true}`                                 | 4.4.1                       |
| ECharts 图表       | `chart: {echarts: true}`                                 | 5.5.0，跟随明暗主题         |
| Plotly / Vega-Lite | `chart: {plotly: true}` / `{vega_lite: true}`            |                             |
| Mermaid 流程图     | `mermaid: {enabled: true, zoomable: true}`               | zoomable 追加 d3            |
| typograms          | `typograms: true`                                        |                             |
| 美化表格           | `pretty_table: true`                                     | 与 `code_diff` 互斥         |
| 代码 diff          | `code_diff: true`                                        | diff2html                   |
| 侧边目录           | `toc: {sidebar: left}`                                   |                             |
| 图片点击放大       | 全局已开 `enable_medium_zoom`                            | figure 加 `zoomable=true`   |
| 数学公式           | 全局已开 `enable_math`                                   | MathJax                     |

图表数据写在 Markdown 的围栏代码块内（渲染脚本扫描 `pre > code.language-echarts` / `.language-chartjs` 并 `JSON.parse`），不使用独立数据文件。

可折叠深读由 `_plugins/details.rb` 提供的 `{% details 标题 %}...{% enddetails %}` 实现，渲染为原生 `<details>`。

仍需自定义 CSS、属于范围扩大的版面元素：hero 大图区、指标卡 / KPI 数字行、图文交替布局、时间线。本轮不做，需要时单独评估。

## 8. 验证方式

按 `README.md` 规定的提交前检查：

```bash
npx prettier . --check
tools/jekyll-docker.sh check
tools/jekyll-docker.sh build
```

人工核对项：

- `/projects/` 可访问，导航栏中 Projects 项高亮正确。
- `/projects/<slug>/` 详情页元信息渲染正确。
- 元信息字段缺失时不渲染空行。
- 手机宽度（约 375px）下列表条目不横向溢出。
- 暗色与亮色两种主题下排版均不破。
- 构建产物 `_site/` 中不包含 `docs/` 目录。

构建环境已在设计阶段打通并实测通过，见 `docs/2026-09-20-local-build-environment.md`。本机系统 Ruby 为 3.0.2，装不上本仓库依赖，因此本地构建统一走 `tools/jekyll-docker.sh`，容器内 Ruby 版本与 `.github/workflows/deploy.yml` 所钉的 3.3.5 一致。

## 9. 本轮明确不做的事

- **不改动首页。** `_layouts/about.liquid` 中 `home-flow` 的 `01`–`05` 编号为硬编码，插入新段落需重排后续全部编号。待 demo 内容上线且确认需要首页露出时，再作为独立改动处理。
- 不做 category 分组。
- 不恢复 al-folio 的卡片网格 include（`projects.liquid` / `projects_horizontal.liquid`）。
- 不恢复 `enable_project_categories` 开关。
