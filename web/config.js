// Where the backend (server.py) lives.
// - Running locally, the same server serves this page and the API, so it stays "".
// - On Vercel, the API is on Render: replace the URL below with your Render service URL.
const IS_LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
window.VERDICT_API = IS_LOCAL ? "" : "https://verdict-api-knuw.onrender.com";
