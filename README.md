# Peizheng Li Personal Homepage

Source for [edwardleelpz.github.io](https://edwardleelpz.github.io), the personal research homepage of Peizheng Li.

## Local development

```bash
bundle install
npm ci
bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

Run formatting before opening a pull request:

```bash
npx prettier . --write
```

## Site structure

- `_pages/` contains the public pages: home, CV, publications, blog, news, and repositories.
- `_data/` stores structured CV, social, venue, coauthor, and repository data.
- `_bibliography/` contains publication metadata.
- `_posts/` contains personal research notes and blog posts.
- `assets/` contains profile images, publication posters, and site assets.

## Theme notice

This site is customized from the Jekyll-based al-folio theme. Theme code remains under the original MIT license in `LICENSE`; personal content, images, CV data, and publication materials belong to their respective authors.
