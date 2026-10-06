// Turns one pipeline result into what the page shows.
//
// Nothing here decides anything. Scores, labels and verdicts are read straight
// from the result dict that api/main.py returns. This file only decides how to
// word and group them, so a "not scored" stage says why it was not scored.

// The three stages the page shows. Layer 4 (image decoder) is an input step,
// not a stage, because the analyzer takes text and sender only.
const LAYER_BY_STAGE = { 1: "l1", 2: "l2", 3: "l3" };

function explanationFor(result, layerKey) {
  return result.fusion.explanation.layers.find((l) => l.layer === layerKey);
}

export function verdictTone(label) {
  if (!label) return "warn";
  if (label.startsWith("PHISHING")) return "risk";
  if (label === "LEGITIMATE") return "ok";
  return "warn"; // UNCERTAIN, INSUFFICIENT EVIDENCE
}

export function percent(value) {
  return `${Math.round(value * 100)}%`;
}

export function buildVerdict(result) {
  const f = result.fusion;
  return {
    label: f.label,
    tone: verdictTone(f.label),
    score: f.score,
    reason: f.label_reason || null,
    overridden: Boolean(f.sender_override),
    modelLabel: f.model_label,
    confidence: f.confidence,
    layersUsed: f.layers_used,
    summary: f.explanation.summary,
    decision: f.explanation.decision,
    rules: f.explanation.rules,
  };
}

export function buildStages(result) {
  const { layer1_sms: l1, layer2_url: l2, layer3_sender: l3 } = result;
  const e1 = explanationFor(result, LAYER_BY_STAGE[1]);
  const e2 = explanationFor(result, LAYER_BY_STAGE[2]);
  const e3 = explanationFor(result, LAYER_BY_STAGE[3]);

  const stage1 = {
    number: 1,
    name: "Message text",
    scored: l1.smish_probability !== null,
    score: l1.smish_probability,
    finding: e1?.finding ?? null,
    notScoredReason: l1.note ?? "Nothing in the wording could be scored.",
    details: [],
    contribution: e1?.contribution ?? null,
    decisive: Boolean(e1?.decisive),
  };
  if (stage1.scored && l1.cleaned_text) {
    stage1.details.push(`Read as: "${l1.cleaned_text}"`);
  }

  const scoredLinks = l2.results.filter((r) => r.phish_score !== null);
  const stage2 = {
    number: 2,
    name: "Link",
    scored: l2.url_phish_score !== null,
    score: l2.url_phish_score,
    finding: e2?.finding ?? null,
    notScoredReason: result.input.urls.length === 0
      ? "No link in this message."
      : "Each link was checked, but none returned a score.",
    details: [],
    links: l2.results.map((r) => ({
      url: r.url,
      score: r.phish_score,
      error: r.error,
      abstained: r.abstained,
      note: r.note,
    })),
    contribution: e2?.contribution ?? null,
    decisive: Boolean(e2?.decisive),
  };
  if (l2.results.length > 0 && scoredLinks.length === 0) {
    stage2.notScoredReason = "Links were found, but none could be scored.";
  }

  const stage3 = {
    number: 3,
    name: "Sender ID",
    scored: l3.sender_phish_score !== null,
    score: l3.sender_phish_score,
    finding: e3?.finding ?? null,
    notScoredReason: l3.note ?? "No sender ID was provided.",
    details: [],
    status: l3.status,
    header: l3.header,
    entity: l3.entity_name,
    impersonation: Boolean(l3.impersonation),
    resembles: l3.resembles,
    resemblesEntity: l3.resembles_entity,
    invalidHeader: Boolean(l3.invalid_header),
    contribution: e3?.contribution ?? null,
    decisive: Boolean(e3?.decisive),
  };

  return [stage1, stage2, stage3];
}

// Stages that produced no score, in the order the reader meets them. Shown as
// a plain sentence rather than an animated "stopped early" state, because the
// pipeline never stops early: every stage runs and a missing score is just
// reported as missing.
export function unscoredStages(stages) {
  return stages.filter((s) => !s.scored);
}

export function scoredStageCount(stages) {
  return stages.filter((s) => s.scored).length;
}

export function explanationRows(result) {
  const e = result.fusion.explanation;
  return {
    baseline: e.baseline.score,
    layers: e.layers.map((l) => ({
      key: l.layer,
      name: l.name,
      contributed: l.contributed,
      contribution: l.contribution,
      share: l.share,
      direction: l.direction,
      scoreWithout: l.score_without,
      decisive: l.decisive,
    })),
    final: e.arithmetic.score,
  };
}
