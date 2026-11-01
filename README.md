# lijiemingjimmy.github.io

Jieming Li's static academic and personal homepage, designed for direct deployment to GitHub Pages.

## Information architecture

The site uses a single HTML entry point and hash-addressable views:

- `#main` — research positioning, profile links, and news;
- `#info` — education, research experience, projects, skills, and fun facts;
- `#research` — publication-style research and system entries;
- `#blog` — categorized writing archive;
- `#blog/<slug>` — individual article view;
- `#notes` — searchable research notes, linked to rendered standalone pages.

## Local preview

```bash
python3 -m http.server 4173
```

Open `http://127.0.0.1:4173/#main`.

## Tests

```bash
npm test
```

Tests cover routing, public research metadata, the page's semantic contract, Blog filtering, and core responsive/accessibility CSS contracts.

## Content maintenance

- Profile and view templates: `index.html`
- Research, news, and Blog content: `assets/script.js`
- Visual system and responsive layout: `assets/styles.css`
- Locally hosted profile and project visuals: `assets/images/`

Every Research item follows the same data shape in `siteData.research`: identifier, date, venue/status, title, authors, summary, image/alt text, and named links.

## Deployment

The live GitHub Pages source is the `gh-pages` branch, root directory. The `main` branch contains the historical al-folio template and is not the current homepage. Publish changes to `gh-pages`. Pages are served as static files (`.nojekyll`).

This checkout is at `/Users/lijieming/Downloads/Ideas/personal-site`. The earlier standalone source copy is at `/Users/lijieming/Downloads/写作与沟通/lijiemingjimmy.github.io`.

Notes are generated from the Markdown files directly under the parent Ideas directory. See `scripts/build-notes.*` for the import/rendering workflow. Keep original notes unchanged; regenerate the published HTML when notes change.

The RUA-Eval external blog URL was supplied by the site owner; it returned 404 when checked on 2026-11-01. Retain the intended link without inventing article content.
