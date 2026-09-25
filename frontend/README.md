# frontend/

The web interface: two plain HTML files, each carrying its own CSS and JavaScript.

| File | Page | What it does |
|---|---|---|
| `dashboard.html` | `/dashboard` | Upload an image or a video, see the annotated result, per-person statuses and the violation log |
| `live.html` | `/live` | Live webcam monitoring with per-frame statuses |

## Why there is no npm, no React and no build step

Everything these pages need is already in the browser:

* `fetch()` calls `/api/v1/predict`, `/detect/video` and `/api/v1/violations`
* a plain `<img src="/live/stream">` renders the MJPEG camera stream natively
* the annotated picture arrives as a `data:` URI inside the JSON, so there is nothing to host
* CSS custom properties give light and dark themes in a dozen lines

A React app would add a toolchain, a `node_modules` folder and a build step, and would not
show a judge anything these pages do not. Editing a page here means opening the file and
refreshing the browser.

## How they are served

FastAPI returns them with `FileResponse` (`server/main.py` and `server/routes/live.py`); the
folder is located once, as `FRONTEND_DIR` in `server/config.py`. There is no static mount,
because two files do not need one.

## Editing

Change the file, save, then **hard-refresh** the browser (Ctrl+F5). The server does not need
restarting - the page is read from disk on every request.
