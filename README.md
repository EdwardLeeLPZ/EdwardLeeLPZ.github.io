# Peizheng Li Personal Homepage

Source code for [edwardleelpz.github.io](https://edwardleelpz.github.io), the personal research homepage of Peizheng Li.

The site is a customized Jekyll homepage focused on Spatial Intelligence, Multimodal Foundation Models, World Models, 3D/4D scene understanding, embodied systems, and research-oriented career presentation.

## Current site

- **Home**: concise research positioning, selected updates, featured publications, writing highlights, and open-position statement.
- **Publications**: BibTeX-driven publication list with venue/year metadata, previews, paper links, posters, videos, and selected paper badges.
- **Blogs**: research notes and technical writing in English and Chinese.
- **CV**: structured CV sections from `_data/cv.yml` plus an embedded high-DPI PDF.js viewer for `assets/pdf/Resume.pdf`.

## Visual system

The original template look has been replaced with a custom dark editorial system:

- Warm near-black background and low-saturation gray-orange accents.
- Flat section rhythm with thin dividers instead of heavy rounded cards.
- Connected point-line network texture used as a restrained visual motif.
- Shared typography, colors, title sizing, navigation behavior, and page width across Home, Publications, Blogs, and CV.
- Responsive layouts for both desktop and mobile reading.

Primary style entry points:

- `_sass/_variables.scss` and `_sass/_themes.scss` define the shared palette tokens.
- `_sass/_homepage.scss` controls the homepage composition.
- `_sass/_pages.scss` controls shared subpage styling, publications, blog, CV, and navigation.
- `_layouts/about.liquid`, `_layouts/bib.liquid`, and `_layouts/cv.liquid` provide the main custom page layouts.

## Content sources

- `_pages/about.md` stores homepage profile copy.
- `_data/cv.yml` stores structured CV content.
- `_bibliography/papers.bib` stores publication metadata.
- `_posts/` stores blog posts.
- `assets/img/publication_preview/` stores publication preview images.
- `assets/pdf/` stores posters and the embedded resume PDF.

## Local development

Install dependencies:

```bash
bundle install
npm ci
```

Start a local preview:

```bash
bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

For faster Windows preview outside OneDrive output paths, keep a local-only `_config.local.yml` and run:

```bash
bundle exec jekyll serve --host 127.0.0.1 --port 4000 --livereload --config _config.yml,_config.local.yml --disable-disk-cache
```

`_config.local.yml`, `_site/`, local cache folders, local agent settings, and personal planning notes are intentionally ignored by Git.

## Validation

Before committing changes, run:

```bash
npx prettier . --check
bundle check
bundle exec jekyll build --config _config.yml,_config.local.yml --disable-disk-cache
```

Production deployment uses `.github/workflows/deploy.yml` and builds with `_config.yml`.

## Deployment

Changes pushed to `master` trigger GitHub Actions:

- `Deploy site` builds the Jekyll site and publishes `_site` to GitHub Pages.
- `Prettier code formatter` checks formatting.

After deployment, verify:

- <https://edwardleelpz.github.io/>
- <https://edwardleelpz.github.io/publications/>
- <https://edwardleelpz.github.io/blog/>
- <https://edwardleelpz.github.io/cv/>

## Maintenance notes

- The CV resume is embedded as a PDF rendered through PDF.js, not as a raster image. The viewer renders canvas backing pixels using `window.devicePixelRatio` for clearer desktop and mobile display.
- Publication source names are rendered as direct venue/journal names with year-only metadata.
- The site no longer keeps unused al-folio demo pages, project/book/profile/teaching pages, Docker/devcontainer files, or JSON Resume template includes.

## License and ownership

This site is customized from the Jekyll-based al-folio theme. Theme code remains under the original MIT license in `LICENSE`; personal content, images, CV data, and publication materials belong to their respective authors.
