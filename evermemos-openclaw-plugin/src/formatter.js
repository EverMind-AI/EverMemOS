export const CONTEXT_BOUNDARY = "user\u200b原\u200b始\u200bquery\u200b：\u200b\u200b\u200b\u200b";

function timestampToLabel(ts) {
  if (ts == null || ts === "") return "";

  if (typeof ts === "number") {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return "";
    const p = (n) => `${n}`.padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  if (typeof ts === "string") {
    const s = ts.trim();
    if (!s) return "";
    // Unix epoch as string
    if (/^\d{10,13}$/.test(s)) return timestampToLabel(Number(s));
    // ISO 8601: extract date and HH:MM
    const dateEnd = s.indexOf("T");
    if (dateEnd === 10 && s.length > 15) return `${s.slice(0, 10)} ${s.slice(11, 16)}`;
    return s;
  }

  return "";
}

export function parseSearchResponse(raw) {
  if (raw?.status !== "ok" || !raw?.result) return null;

  const allMemories = raw.result.memories ?? [];

  // episodic memories
  const episodic = allMemories
    .filter((m) => m.memory_type === "episodic_memory" && (m.score ?? 0) >= 0.1)
    .map((m) => {
      const body = m.summary || m.episode || m.content || "";
      const subject = m.subject || "";
      return {
        text: subject ? `${subject}: ${body}` : body,
        timestamp: m.timestamp ?? null,
      };
    });

  const traits = (raw.result.profiles ?? []).filter((p) => (p.score ?? 0) >= 0.1).map((p) => {
    const label = p.category || p.trait_name || "";
    let kind = p.item_type || "";
    if (kind === "explicit_info") kind = "explicit";
    else if (kind === "implicit_trait") kind = "implicit";
    return {
      text: label ? `[${label}] ${p.description || ""}` : (p.description || ""),
      kind,
    };
  });

  // agent_case: top 1 by score (min 0.01)
  const topCase = allMemories
    .filter((m) => m.memory_type === "agent_case" && (m.score ?? 0) >= 0.01)
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0] || null;

  // agent_skill: top 1 by score (min 0.01)
  const topSkill = allMemories
    .filter((m) => m.memory_type === "agent_skill" && (m.score ?? 0) >= 0.01)
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0] || null;

  return { episodic, traits, case: topCase, skill: topSkill };
}

function oneLiner(text) {
  return text == null ? "" : String(text).replace(/[\r\n]+/g, " ").trim();
}

function factLine(fact) {
  const t = oneLiner(fact.text);
  if (!t) return "";
  const when = timestampToLabel(fact.timestamp);
  return when ? `  - [${when}] ${t}` : `  - ${t}`;
}

function traitLine(trait) {
  const t = oneLiner(trait.text);
  if (!t) return "";
  const k = trait.kind?.toLowerCase() ?? "";
  const badge = k.includes("explicit") ? " [Explicit]"
    : k.includes("implicit") ? " [Implicit]"
    : trait.kind ? ` [${trait.kind.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}]`
    : "";
  return `  -${badge} ${t}`;
}

function caseBlock(c) {
  if (!c) return [];
  const intent = oneLiner(c.task_intent || "");
  const approach = c.approach || "";
  if (!intent && !approach) return [];
  const when = timestampToLabel(c.timestamp ?? c.created_at ?? null);
  return [
    "  <case>",
    ...(when ? [`    - time: ${when}`] : []),
    ...(intent ? [`    - intent: ${intent}`] : []),
    ...(approach ? [`    - approach: ${approach}`] : []),
    "  </case>",
  ];
}

function skillBlock(s) {
  if (!s) return [];
  const name = oneLiner(s.name || "");
  const desc = oneLiner(s.description || "");
  const content = s.content || "";
  if (!name && !content) return [];
  return [
    "  <skill>",
    ...(name ? [`    - name: ${name}`] : []),
    ...(desc ? [`    - description: ${desc}`] : []),
    ...(content ? [`    - content: ${content}`] : []),
    "  </skill>",
  ];
}

export function buildMemoryPrompt(parsed, opts = {}) {
  if (!parsed) return "";

  const episodicLines = parsed.episodic.map(factLine).filter(Boolean);
  const traitLines = parsed.traits.map(traitLine).filter(Boolean);
  const caseLines = caseBlock(parsed.case);
  const skillLines = skillBlock(parsed.skill);

  if (!episodicLines.length && !traitLines.length && !caseLines.length && !skillLines.length) return "";

  const xmlBlock = [
    "<memory>",
    ...(episodicLines.length ? ["  <episodic>", ...episodicLines, "  </episodic>"] : []),
    ...(traitLines.length ? ["  <trait>", ...traitLines, "  </trait>"] : []),
    ...(caseLines.length ? ["  <!-- Similar past case. Use as reference if applicable to the current task. -->", ...caseLines] : []),
    ...(skillLines.length ? ["  <!-- Relevant skill. Use as reference if applicable to the current task. -->", ...skillLines] : []),
    "</memory>",
  ];

  const memSection = opts.wrapInCodeBlock ? ["```text", ...xmlBlock, "```"] : xmlBlock;
  const nowLabel = timestampToLabel(Date.now());

  return [
    "Note: Reference memory below. Build on past successes; avoid repeating failed approaches.",
    ...(nowLabel ? [`- Time: ${nowLabel}`] : []),
    "",
    ...memSection,
    "",
    "**Note**: for memory, please not read from or write to local `MEMORY.md` or `memory/*` files as they are provided above.",
    "",
    CONTEXT_BOUNDARY,
  ].join("\n");
}
