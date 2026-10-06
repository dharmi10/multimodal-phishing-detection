import { motion, useReducedMotion } from "motion/react";
import {
  buildStages,
  buildVerdict,
  explanationRows,
  percent,
  scoredStageCount,
  unscoredStages,
} from "../lib/view.js";

// One orchestrated sequence: the three stages land in turn, then the verdict.
// The step is the only motion on the page that is not a response to a click.
const STEP_S = 0.45;

const TONE = {
  risk: { text: "text-risk", bar: "bg-risk", tint: "bg-risk-tint", border: "border-risk/40" },
  ok: { text: "text-ok", bar: "bg-ok", tint: "bg-ok-tint", border: "border-ok/40" },
  warn: { text: "text-warn", bar: "bg-warn", tint: "bg-warn-tint", border: "border-warn/40" },
};

export function Result({ result, source, sourceReason }) {
  const reduce = useReducedMotion();
  const verdict = buildVerdict(result);
  const stages = buildStages(result);
  const tone = TONE[verdict.tone];
  const reveal = (delay) =>
    reduce
      ? { initial: false }
      : {
          initial: { opacity: 0, y: 10 },
          animate: { opacity: 1, y: 0 },
          transition: { duration: 0.4, delay, ease: [0.2, 0.7, 0.2, 1] },
        };

  return (
    <section id="result" aria-labelledby="result-heading" className="border-t border-rule bg-surface">
      <div className="mx-auto max-w-5xl px-4 py-20 sm:px-6">
        <h2 id="result-heading" className="sr-only">
          Result
        </h2>

        <SourceNote source={source} reason={sourceReason} />

        <div className="mt-6 grid gap-10 lg:grid-cols-[1fr_18rem] lg:gap-14">
          <div>

            {result.layer4_image?.images_scored > 0 && (
              <ImageFindings result={result} />
            )}

            <ol className="mt-8 space-y-0 border-t border-rule">
              {stages.map((stage, i) => (
                <motion.li
                  key={stage.number}
                  {...reveal(i * STEP_S)}
                  className="border-b border-rule py-7"
                >
                  <StageRow stage={stage} />
                </motion.li>
              ))}
            </ol>

            <motion.div {...reveal(stages.length * STEP_S)} className="mt-10">
              <Rules rules={verdict.rules} />
            </motion.div>
          </div>

          <motion.aside {...reveal(stages.length * STEP_S + 0.1)} className="lg:sticky lg:top-24 lg:self-start">
            <VerdictBlock verdict={verdict} tone={tone} stagesScored={scoredStageCount(stages)} total={stages.length} unscored={unscoredStages(stages)} />
          </motion.aside>
        </div>

        <motion.div {...reveal(stages.length * STEP_S + 0.25)} className="mt-16">
          <Arithmetic result={result} verdict={verdict} />
        </motion.div>
      </div>
    </section>
  );
}

function SourceNote({ source, reason }) {
  if (source === "live") {
    return null;
  }
  return (
    <p className="rounded-md border border-warn/40 bg-warn-tint px-4 py-3 text-sm text-warn">
      Saved result. {reason}
    </p>
  );
}

function VerdictBlock({ verdict, tone, stagesScored, total, unscored }) {
  return (
    <div className={`rounded-lg border ${tone.border} ${tone.tint} p-6`}>
      <p className="text-sm font-medium text-muted">Verdict</p>
      <h3 className={`mt-2 font-display text-3xl leading-tight font-medium ${tone.text}`}>
        {verdict.label}
      </h3>

      <p className="mt-5 text-sm text-muted">Phishing score</p>
      <p className="font-display text-4xl font-medium tabular-nums">{verdict.score.toFixed(2)}</p>
      <ScoreBar value={verdict.score} barClass={tone.bar} />

      <dl className="mt-6 grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
        <dt className="text-muted">Confidence</dt>
        <dd className="text-right font-medium capitalize">{verdict.confidence}</dd>
        <dt className="text-muted">Stages scored</dt>
        <dd className="text-right font-medium tabular-nums">
          {stagesScored} of {total}
        </dd>
      </dl>

      {verdict.reason && <p className="mt-5 text-sm">{verdict.reason}</p>}

      {verdict.overridden && (
        <p className="mt-5 text-sm">
          The model alone said <span className="font-medium">{verdict.modelLabel}</span>.
          A sender rule changed the label. The score is unchanged.
        </p>
      )}

      {unscored.length > 0 && (
        <div className="mt-5 border-t border-rule pt-4 text-sm">
          <p className="font-medium">Not scored</p>
          <ul className="mt-2 space-y-1 text-muted">
            {unscored.map((s) => (
              <li key={s.number}>
                Stage {s.number}, {s.name.toLowerCase()}: {s.notScoredReason}
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-5 border-t border-rule pt-4 text-sm text-muted">{verdict.summary}</p>
    </div>
  );
}

function StageRow({ stage }) {
  return (
    <div className="grid gap-x-8 gap-y-3 md:grid-cols-[12rem_1fr]">
      <div>
        <p className="text-sm font-medium text-muted">Stage {stage.number}</p>
        <h3 className="mt-1 text-lg font-semibold">{stage.name}</h3>
        <p className={`mt-2 text-sm font-medium ${stage.scored ? "text-ink" : "text-muted"}`}>
          {stage.scored ? "Scored" : "Not scored"}
        </p>
      </div>

      <div className="min-w-0">
        {stage.scored ? (
          <>
            <div className="flex items-baseline justify-between gap-4">
              <p className="text-base">{stage.finding}</p>
              <p className="font-display text-2xl font-medium tabular-nums">{stage.score.toFixed(2)}</p>
            </div>
            <ScoreBar value={stage.score} barClass={stage.score >= 0.5 ? "bg-risk" : "bg-ok"} />
            {stage.contribution !== null && (
              <p className="mt-2 text-sm text-muted tabular-nums">
                Moved the score {stage.contribution > 0 ? "up" : "down"} by{" "}
                {Math.abs(stage.contribution).toFixed(2)} log-odds
                {stage.decisive ? ", and removing it would flip the verdict" : ""}.
              </p>
            )}
          </>
        ) : (
          <p className="text-base text-muted">{stage.notScoredReason}</p>
        )}

        {stage.number === 2 && stage.links?.length > 0 && <LinkList links={stage.links} />}
        {stage.number === 3 && stage.scored === false && stage.header && (
          <p className="mt-3 font-mono text-sm text-muted">Header read: {stage.header}</p>
        )}
        {stage.number === 3 && stage.scored && <SenderDetail stage={stage} />}
        {stage.number === 1 && stage.details.map((d) => (
          <p key={d} className="mt-3 text-sm text-muted">{d}</p>
        ))}
      </div>
    </div>
  );
}

function LinkList({ links }) {
  return (
    <ul className="mt-4 divide-y divide-rule border-y border-rule text-sm">
      {links.map((link) => (
        <li key={link.url} className="grid grid-cols-[1fr_auto] gap-4 py-3">
          <span className="min-w-0 break-all font-mono">{link.url}</span>
          <span className="text-right tabular-nums">
            {link.error ? (
              <span className="text-muted">Could not fetch</span>
            ) : link.abstained ? (
              <span className="text-muted">Not scored</span>
            ) : (
              <span className="font-medium">{link.score.toFixed(2)}</span>
            )}
            {link.note && <span className="block text-xs text-muted">{link.note}</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}

function SenderDetail({ stage }) {
  return (
    <div className="mt-4 space-y-2 text-sm">
      <p>
        <span className="font-mono">{stage.header}</span>
        {stage.status === "registered" && stage.entity && (
          <span className="text-muted"> is registered to {stage.entity}.</span>
        )}
      </p>
      {stage.impersonation && (
        <p className="text-risk">
          Impersonation: one character from {stage.resembles}
          {stage.resemblesEntity ? `, a header for ${stage.resemblesEntity}` : ""}.
        </p>
      )}
      {stage.invalidHeader && (
        <p className="text-risk">This header has a shape no registered sender uses.</p>
      )}
    </div>
  );
}

// What the image decoder found, shown before the stages because the image's
// text and links are fed into stages 1 and 2. Layer 4 has no score of its own.
function ImageFindings({ result }) {
  const l4 = result.layer4_image;
  const l2Links = result.layer2_url.results.map((r) => r.url);
  const typedLinks = result.input.urls;
  const ocrLinks = l2Links.filter((u) => !typedLinks.includes(u) && !l4.qr_urls.includes(u));
  const senderFromImage = result.input.sender_source === "read from the image"
    ? result.input.sender_from_image
    : null;

  return (
    <div className="mt-8 border-t border-rule pt-6">
      <h3 className="text-lg font-semibold">Image</h3>
      <p className="mt-1 text-sm text-muted">
        {l4.images_scored} {l4.images_scored === 1 ? "image" : "images"} read.
        The text and links found in it are checked in Stages 1 and 2.
      </p>

      <dl className="mt-5 grid grid-cols-[auto_1fr] gap-x-6 gap-y-4 text-sm">
        {l4.qr_urls.length > 0 && (
          <>
            <dt className="text-muted">QR code links</dt>
            <dd className="min-w-0 break-all font-mono">{l4.qr_urls.join(", ")}</dd>
          </>
        )}
        {ocrLinks.length > 0 && (
          <>
            <dt className="text-muted">Links in the image text</dt>
            <dd className="min-w-0 break-all font-mono">{ocrLinks.join(", ")}</dd>
          </>
        )}
        <dt className="text-muted">Text in the image</dt>
        <dd>
          {l4.ocr_degraded
            ? "Too damaged to read. Discarded, not scored."
            : l4.ocr_empty
              ? "No text found."
              : <span className="italic">"{l4.ocr_text}"</span>}
        </dd>
        {senderFromImage && (
          <>
            <dt className="text-muted">Sender read from image</dt>
            <dd>
              <span className="font-mono">{result.input.sender_id}</span>, line{" "}
              {senderFromImage.line_index} of image {senderFromImage.image}
            </dd>
          </>
        )}
      </dl>

      {l4.notes.length > 0 && (
        <ul className="mt-4 space-y-1 text-sm text-muted">
          {l4.notes.map((note) => <li key={note}>{note}</li>)}
        </ul>
      )}
    </div>
  );
}

function Rules({ rules }) {
  return (
    <div>
      <p className="text-sm font-medium text-muted">Rules checked beside the model</p>
      <ul className="mt-3 divide-y divide-rule border-y border-rule text-sm">
        {rules.map((rule) => (
          <li key={rule.name} className="grid grid-cols-[auto_1fr] gap-4 py-3">
            <span className={`font-medium ${rule.fired ? "text-risk" : "text-muted"}`}>
              {rule.fired ? "Fired" : "Clear"}
            </span>
            <span>
              <span className="font-medium">{capitalise(rule.name)}.</span>{" "}
              <span className="text-muted">{capitalise(rule.effect)}.</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Arithmetic({ result, verdict }) {
  const rows = explanationRows(result);
  return (
    <div>
      <h3 className="text-xl font-semibold">How the score was built</h3>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Each row is the change in log-odds that one stage contributed. The
        rows add up exactly to the final score.
      </p>
      <table className="mt-6 w-full max-w-3xl text-sm tabular-nums">
        <thead>
          <tr className="border-b border-ink text-left text-muted">
            <th scope="col" className="py-2 font-medium">Step</th>
            <th scope="col" className="py-2 text-right font-medium">Change</th>
            <th scope="col" className="py-2 text-right font-medium">Share of movement</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-rule">
          <tr>
            <td className="py-3">No evidence (model prior)</td>
            <td className="py-3 text-right text-muted">{percent(rows.baseline)}</td>
            <td className="py-3 text-right text-muted">–</td>
          </tr>
          {rows.layers.map((layer) => (
            <tr key={layer.key}>
              <td className="py-3">
                {layer.name}
                {!layer.contributed && <span className="text-muted"> (did not contribute)</span>}
              </td>
              <td className="py-3 text-right">
                {layer.contributed ? (layer.contribution > 0 ? "+" : "") + layer.contribution.toFixed(2) : "–"}
              </td>
              <td className="py-3 text-right text-muted">
                {layer.contributed ? percent(layer.share) : "–"}
              </td>
            </tr>
          ))}
          <tr className="font-medium">
            <td className="py-3">Final score</td>
            <td className="py-3 text-right">{rows.final.toFixed(3)}</td>
            <td className="py-3 text-right">{verdict.label}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function ScoreBar({ value, barClass }) {
  return (
    <div
      className="mt-3 h-2 w-full overflow-hidden rounded-full bg-rule/70"
      role="img"
      aria-label={`${Math.round(value * 100)} out of 100`}
    >
      <div className={`h-full rounded-full ${barClass}`} style={{ width: `${Math.max(2, value * 100)}%` }} />
    </div>
  );
}

function capitalise(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
