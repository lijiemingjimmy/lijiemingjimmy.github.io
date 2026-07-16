# lijiemingjimmy.github.io

Jieming Li's static academic and personal homepage, designed for direct deployment to GitHub Pages.

## Information architecture

The site uses a single HTML entry point and hash-addressable views:

- `#main` — research positioning, profile links, and news;
- `#info` — education, research experience, projects, skills, and fun facts;
- `#research` — publication-style research and system entries;
- `#blog` — categorized writing archive;
- `#blog/<slug>` — individual article view.

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

Publish this directory as the root of the `lijiemingjimmy.github.io` repository. No build step or external runtime dependency is required.
