import { useEffect, useMemo, useState } from "react";

const KIND_LABEL = {
  legitimate: "Legitimate",
  phishing: "Phishing",
  borderline: "Borderline",
};

const MAX_TEXT = 1000;
const MAX_SENDER = 40;
const MAX_IMAGES = 5;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;

export function Analyzer({ samples, busy, error, apiLive, onRun }) {
  const [sender, setSender] = useState("");
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [formError, setFormError] = useState(null);

  // Object URLs for the thumbnails. Revoked when the files change, so a
  // removed or replaced image does not stay in memory.
  const previews = useMemo(() => files.map((file) => URL.createObjectURL(file)), [files]);
  useEffect(() => () => previews.forEach((url) => URL.revokeObjectURL(url)), [previews]);

  function addFiles(event) {
    const picked = Array.from(event.target.files ?? []);
    event.target.value = "";
    const problem = checkFiles([...files, ...picked]);
    if (problem) {
      setFormError(problem);
      return;
    }
    setFormError(null);
    setFiles([...files, ...picked]);
  }

  function removeFile(index) {
    setFiles(files.filter((_, i) => i !== index));
  }

  function submit(event) {
    event.preventDefault();
    if (!text.trim() && files.length === 0) {
      setFormError("Paste the message text or add an image.");
      return;
    }
    setFormError(null);
    onRun({ sender, text, files });
  }

  function useExample(sample) {
    setSender(sample.sender);
    setText(sample.text);
    setFiles([]);
    setFormError(null);
    onRun({ sender: sample.sender, text: sample.text, files: [] });
  }

  const shownError = formError || error;

  return (
    <section id="check" className="mx-auto max-w-5xl px-4 pt-10 pb-16 sm:px-6 sm:pt-14">
      <div className="grid gap-12 lg:grid-cols-[1fr_20rem] lg:gap-16">
        <div>
          <h1 className="font-display text-3xl font-medium tracking-tight sm:text-4xl">
            Check a message
          </h1>

          <form onSubmit={submit} className="mt-10 space-y-7" noValidate>
            <div>
              <label htmlFor="sender" className="block text-sm font-medium">
                Sender ID
              </label>
              <input
                id="sender"
                type="text"
                value={sender}
                maxLength={MAX_SENDER}
                onChange={(e) => setSender(e.target.value)}
                placeholder="VM-HDFCBK"
                autoComplete="off"
                spellCheck="false"
                className="mt-2 w-full rounded-md border border-rule bg-surface px-4 py-3 font-mono text-base placeholder:text-muted/60 focus:border-ink"
              />
            </div>

            <div>
              <label htmlFor="message" className="block text-sm font-medium">
                Message text
              </label>
              <textarea
                id="message"
                rows={5}
                value={text}
                maxLength={MAX_TEXT}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste the message here"
                className="mt-2 w-full resize-y rounded-md border border-rule bg-surface px-4 py-3 text-base placeholder:text-muted/60 focus:border-ink"
              />
              <p className="mt-2 text-right text-sm text-muted tabular-nums">
                {text.length} / {MAX_TEXT}
              </p>
            </div>

            <div>
              <p className="text-sm font-medium">
                Image <span className="font-normal text-muted">(optional)</span>
              </p>
              <label className="mt-2 flex cursor-pointer items-center justify-center rounded-md border border-dashed border-rule bg-surface px-4 py-6 text-center text-sm text-muted hover:border-ink focus-within:outline-2 focus-within:outline-offset-3 focus-within:outline-ink">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={addFiles}
                  className="sr-only"
                />
                Add a screenshot or a QR code. Up to {MAX_IMAGES} images, 5 MB each.
              </label>

              {files.length > 0 && (
                <ul className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                  {files.map((file, i) => (
                    <li key={`${file.name}-${i}`} className="relative overflow-hidden rounded-md border border-rule bg-surface">
                      <img
                        src={previews[i]}
                        alt={`Attached image ${i + 1}: ${file.name}`}
                        className="h-32 w-full object-cover"
                      />
                      <div className="flex items-center justify-between gap-2 border-t border-rule px-3 py-2 text-xs">
                        <span className="truncate text-muted">{file.name}</span>
                        <button
                          type="button"
                          onClick={() => removeFile(i)}
                          className="shrink-0 font-medium underline underline-offset-2 hover:text-risk"
                        >
                          Remove
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-4">
              <button
                type="submit"
                disabled={busy}
                className="rounded-md bg-ink px-5 py-3 text-base font-medium text-paper transition-colors hover:bg-ink/85 disabled:cursor-wait disabled:opacity-60"
              >
                {busy ? "Checking the message" : "Run analysis"}
              </button>
              {busy && (
                <p className="text-sm text-muted" role="status">
                  Links are fetched live, so this can take several seconds.
                </p>
              )}
            </div>

            {shownError && (
              <p role="alert" className="rounded-md border border-risk/30 bg-risk-tint px-4 py-3 text-sm text-risk">
                {shownError}
              </p>
            )}
          </form>
        </div>

        <aside aria-labelledby="examples-heading" className="lg:pt-[3.25rem]">
          <h2 id="examples-heading" className="text-base font-semibold">
            Examples
          </h2>

          {samples.length > 0 ? (
            <ul className="mt-5 divide-y divide-rule border-y border-rule">
              {samples.map((sample) => (
                <li key={sample.id}>
                  <button
                    type="button"
                    onClick={() => useExample(sample)}
                    disabled={busy}
                    className="group block w-full py-4 text-left disabled:opacity-60"
                  >
                    <span className="block text-sm font-medium text-muted">
                      {KIND_LABEL[sample.kind] ?? sample.kind}
                    </span>
                    <span className="mt-1 block font-mono text-sm">{sample.sender}</span>
                    <span className="mt-1 block text-sm text-ink/80 group-hover:text-ink">
                      {truncate(sample.text, 96)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-5 text-sm text-muted">The examples could not be loaded.</p>
          )}

          {apiLive === false && (
            <p className="mt-6 text-sm text-muted">Live check offline. Examples show saved results.</p>
          )}
        </aside>
      </div>
    </section>
  );
}

function checkFiles(list) {
  if (list.length > MAX_IMAGES) return `Attach at most ${MAX_IMAGES} images.`;
  const big = list.find((f) => f.size > MAX_IMAGE_BYTES);
  if (big) return `${big.name} is larger than 5 MB.`;
  const notImage = list.find((f) => !f.type.startsWith("image/"));
  if (notImage) return `${notImage.name} is not an image.`;
  return null;
}

function truncate(text, length) {
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}
