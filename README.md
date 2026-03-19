# CandlePick

Pin/V pattern stock screening with a static web viewer.

## Static Site (Vercel-ready)

- Entry: `index.html`
- Data:
  - `data/pin_data.json`
  - `data/v_data.json`
  - `data/results_展示200条.csv`

This static site displays the first 200 records for each pattern (Pin and V) and can be deployed directly on Vercel without Flask.

## Local preview

Run any static file server in repo root, for example:

```bash
python -m http.server 8080
```

Then open: `http://127.0.0.1:8080`

## Vercel deploy

Import this repository in Vercel and deploy with default settings.
