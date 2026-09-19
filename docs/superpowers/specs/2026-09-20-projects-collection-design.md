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

论文项目（SpaceDrive、AGO、PowerBEV 等）不进入 Projects。

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

| 文件                                | 动作                                                                       |
| ----------------------------------- | -------------------------------------------------------------------------- |
| `_config.yml`                       | 新增 `collections.projects: {output: true}`；在 `exclude` 中加入 `docs`    |
| `_pages/projects.md`                | 新建。`layout: page`、`permalink: /projects/`、`nav: true`、`nav_order: 3` |
| `_pages/cv.md`                      | `nav_order` 由 `3` 改为 `4`                                                |
| `_layouts/project.liquid`           | 新建。详情页 layout                                                        |
| `_includes/project_entry.liquid`    | 新建。列表页单条目                                                         |
| `_sass/_projects.scss`              | 新建                                                                       |
| `assets/css/main.scss`              | `@import` 列表中在 `"pages"` 之后加入 `"projects"`                         |
| `_sass/_base.scss`                  | 删除第 726–763 行的死 `.projects` 块                                       |
| `assets/img/project_images/<slug>/` | 新建资源目录，组织方式对齐现有 `assets/img/blog_images/<slug>/`            |

`docs` 必须加入 `_config.yml` 的 `exclude`：Jekyll 默认会构建所有非 `_` / `.` 开头的顶层目录，否则本设计文档会被发布到 `https://edwardleelpz.github.io/docs/` 下。

collection 输出 URL 使用 Jekyll 默认的 `/:collection/:path/`，即 `/projects/<slug>/`，无需显式配置 permalink。

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

仓库当前未启用 Git LFS，已有 3 个 3MB 以上的 PDF（`assets/pdf/Poster_*.pdf`）与一张 1.6MB 的 JPG。策略：

- 图片、短 GIF、webm：单文件不超过 5MB，直接放入 `assets/img/project_images/<slug>/`。
- 完整演示视频：优先使用外链 iframe（`_includes/video.liquid` 已支持 mp4/webm/ogg 本地文件与 iframe 外链两种模式），或压制为 720p webm 后入库。
- GitHub 单文件硬上限 100MB；仓库整体超过 1GB 会触发告警。

具体取舍待作者提供实际素材后按体积确定。

## 7. 交互式 demo 的预留

`_layouts/project.liquid` 在正文之后预留一个 `{% if page.interactive %}` 分支的位置，**本轮不写任何实现**。等到确有交互式 demo 需求时，再决定采用 iframe 嵌入外部页面，还是在页内挂载 `assets/js` 依赖并在 `_includes/scripts.liquid` 中引入。提前实现只会产生死代码。

## 8. 验证方式

按 `README.md` 规定的提交前检查：

```bash
npx prettier . --check
bundle check
bundle exec jekyll build --config _config.yml --disable-disk-cache
```

人工核对项：

- `/projects/` 可访问，导航栏中 Projects 项高亮正确。
- `/projects/<slug>/` 详情页元信息渲染正确。
- 元信息字段缺失时不渲染空行。
- 手机宽度（约 375px）下列表条目不横向溢出。
- 暗色与亮色两种主题下排版均不破。
- 构建产物 `_site/` 中不包含 `docs/` 目录。

风险说明：本地 Ruby 环境是否可用尚未验证（未执行过 `bundle check`）。若实现阶段发现无法构建，将明确说明哪些验证跑不了、风险是什么。

## 9. 本轮明确不做的事

- **不改动首页。** `_layouts/about.liquid` 中 `home-flow` 的 `01`–`05` 编号为硬编码，插入新段落需重排后续全部编号。待 demo 内容上线且确认需要首页露出时，再作为独立改动处理。
- 不做 category 分组。
- 不恢复 al-folio 的卡片网格 include（`projects.liquid` / `projects_horizontal.liquid`）。
- 不恢复 `enable_project_categories` 开关。
