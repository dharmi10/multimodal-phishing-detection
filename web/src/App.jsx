import { useEffect, useRef, useState } from "react";
import { analyze, AnalyzeError, checkApiHealth, loadSamples } from "./lib/api.js";
import { Analyzer } from "./components/Analyzer.jsx";
import { Result } from "./components/Result.jsx";

// One page, one tool. The analyzer is the first thing on it, and the result
// appears directly under the form that produced it.
export default function App() {
  const [samples, setSamples] = useState([]);
  const [apiLive, setApiLive] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [outcome, setOutcome] = useState(null);
  const resultRef = useRef(null);

  useEffect(() => {
    loadSamples().then((data) => setSamples(data.samples ?? []));
    checkApiHealth().then(setApiLive);
  }, []);

  async function run({ sender, text, files }) {
    setBusy(true);
    setError(null);
    try {
      const next = await analyze({ sender, text, files });
      setOutcome(next);
      setApiLive(next.source === "live");
      // Wait a frame so the result is in the DOM before scrolling to it.
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ block: "start" }));
    } catch (err) {
      setError(err instanceof AnalyzeError ? err.message : "Something went wrong. Try again.");
      if (err instanceof AnalyzeError && err.offline) setApiLive(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Header apiLive={apiLive} />
      <main>
        <Analyzer
          samples={samples}
          busy={busy}
          error={error}
          apiLive={apiLive}
          onRun={run}
        />
        <div ref={resultRef} className="scroll-mt-20">
          {outcome && (
            <Result
              key={outcome.result.fusion.score + outcome.result.input.raw}
              result={outcome.result}
              source={outcome.source}
              sourceReason={outcome.reason}
            />
          )}
        </div>
      </main>
      <Footer />
    </>
  );
}

function Header({ apiLive }) {
  return (
    <header className="sticky top-0 z-10 border-b border-rule bg-paper/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <a href="#top" className="font-display text-xl font-medium tracking-tight">
          Smishing Check
        </a>
        <nav aria-label="Page sections" className="flex items-center gap-5 text-sm">
          <StatusPill apiLive={apiLive} />
        </nav>
      </div>
    </header>
  );
}

function StatusPill({ apiLive }) {
  if (apiLive === null) return null;
  const live = apiLive === true;
  return (
    <span className="hidden items-center gap-2 text-xs text-muted md:inline-flex">
      <span
        aria-hidden="true"
        className={`size-2 rounded-full ${live ? "bg-ok" : "bg-warn"}`}
      />
      {live ? "Live check" : "Examples only"}
    </span>
  );
}

function Footer() {
  return (
    <footer className="border-t border-rule">
      <div className="mx-auto max-w-5xl px-4 py-10 text-sm text-muted sm:px-6">
        <p>A low score does not make a message safe. If a message asks for money or codes, call the organisation on a number you already have.</p>
      </div>
    </footer>
  );
}
