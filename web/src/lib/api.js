// Talks to the FastAPI wrapper, with a cached fallback for the sample messages.
//
// Layer 2 fetches live URLs, so a request can take several seconds or fail on
// venue wifi. When the API cannot be reached, the three precomputed samples are
// still available, and nothing else is: a typed message needs the live
// pipeline, and the page says so instead of guessing.

const REQUEST_TIMEOUT_MS = 30000;
const SAMPLES_URL = "/demo/samples.json";

export class AnalyzeError extends Error {
  constructor(message, { offline = false } = {}) {
    super(message);
    this.name = "AnalyzeError";
    this.offline = offline;
  }
}

let samplesPromise = null;

export function loadSamples() {
  if (!samplesPromise) {
    samplesPromise = fetch(SAMPLES_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`samples ${r.status}`);
        return r.json();
      })
      .catch(() => ({ samples: [] }));
  }
  return samplesPromise;
}

export async function checkApiHealth() {
  try {
    const r = await fetch("/api/health", { signal: AbortSignal.timeout(3000) });
    return r.ok;
  } catch {
    return false;
  }
}

function findCachedSample(samples, sender, text) {
  const normalisedSender = (sender || "").trim();
  const normalisedText = text.trim();
  return samples.find(
    (s) => (s.sender || "") === normalisedSender && s.text === normalisedText,
  );
}

export async function analyze({ sender, text, files = [] }) {
  const form = new FormData();
  form.append("sender", sender?.trim() || "");
  form.append("text", text.trim());
  for (const file of files) form.append("images", file, file.name);

  let response;
  try {
    response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch {
    return fromCache(sender, text, files, "The live check is not reachable.");
  }

  if (response.status === 422) {
    const data = await response.json().catch(() => null);
    const detail = Array.isArray(data?.detail)
      ? data.detail.map((d) => d.msg).join(" ")
      : data?.detail;
    throw new AnalyzeError(detail || "The message could not be read.");
  }

  if (!response.ok) {
    return fromCache(sender, text, files, "The live check did not return a result.");
  }

  return { result: await response.json(), source: "live" };
}

async function fromCache(sender, text, files, reason) {
  const { samples } = await loadSamples();
  const cached = files.length === 0 && findCachedSample(samples, sender, text);
  if (cached) {
    return { result: cached.result, source: "cached", reason };
  }
  throw new AnalyzeError(
    `${reason} Only the three examples work offline. Start the API to check your own message.`,
    { offline: true },
  );
}
