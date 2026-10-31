# Deploying Verdict

The backend (Python, the agents, the API) runs on **Render**. The dashboard (the `web/` folder) runs on **Vercel**. Both are free and deploy straight from GitHub, so every `git push` updates the live site.

Do the backend first, because the frontend needs its URL.

## 1. Backend on Render

1. Sign in at [render.com](https://render.com) with your GitHub account.
2. Click **New → Blueprint** and pick the `Verdict` repository. Render reads `render.yaml` and sets everything up.
3. It will ask for three values:
   - `GEMINI_API_KEY`: your Gemini key. It's stored on Render only and never in the code.
   - `TAVILY_API_KEY`: your free key from [tavily.com](https://tavily.com), used for web search.
   - `ALLOWED_ORIGINS`: leave it as `*` for now. You'll change it in step 3.
4. Click **Apply** and wait for the first deploy (a few minutes).
5. Open `https://<your-service>.onrender.com/api/config`. If you see some JSON with the model name, the backend is live. Copy the base URL.

> The free plan sleeps after 15 minutes without visitors. The first request after that takes up to a minute while it wakes up; the dashboard shows a "waking up" message during that time.

## 2. Frontend on Vercel

1. Open `web/config.js` and put your Render URL in it:
   ```js
   window.VERDICT_API = IS_LOCAL ? "" : "https://<your-service>.onrender.com";
   ```
   Commit and push.
2. Sign in at [vercel.com](https://vercel.com) with GitHub and click **Add New → Project**.
3. Import the `Verdict` repository and set:
   - **Root Directory:** `web`
   - **Framework Preset:** Other
   - Leave the build command empty. It's a plain HTML/JS site, so there's nothing to build.
4. Click **Deploy**. You'll get a URL like `https://verdict-xyz.vercel.app`.

## 3. Lock the backend to your site

Back on Render, go to your service → **Environment** and set `ALLOWED_ORIGINS` to your Vercel URL (no slash at the end):

```
https://verdict-xyz.vercel.app
```

Save, and Render redeploys. Now only your dashboard can call the API from a browser.

## Things to know

- **History resets.** Render's free disk is temporary, so saved runs disappear when the server restarts or redeploys. That's fine for a demo; a database would fix it later.
- **Quota protection.** Each visitor can ask 5 questions an hour (`RUNS_PER_HOUR`), and only one crew runs at a time, so strangers can't burn through your free Gemini quota.
- **Use Tavily on the server.** DuckDuckGo is much stricter with requests from cloud servers than from home internet. In testing, the first research round found 4 sources on Render versus 12+ on a laptop, so most claims ended up with a single source and got dropped. With `TAVILY_API_KEY` set, Verdict searches with Tavily and only falls back to DuckDuckGo if Tavily fails. `/api/config` shows which search engine is in use.
- **Your key stays secret.** It lives in Render's environment settings. `.env` is in `.gitignore`, and nothing in `web/` ever sees the key, because the browser only talks to your backend.
