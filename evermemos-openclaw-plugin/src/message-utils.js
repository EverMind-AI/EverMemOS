import { CONTEXT_BOUNDARY } from "./formatter.js";

const MAX_CHARS = 20000;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

export function toText(content) {
  if (!content) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.reduce((out, block) => {
    if (block?.type !== "text" || !block.text) return out;
    return out ? `${out} ${block.text}` : block.text;
  }, "");
}

function stripContext(text) {
  if (!text) return text;
  const cut = text.lastIndexOf(CONTEXT_BOUNDARY);
  return cut < 0 ? text : text.slice(cut + CONTEXT_BOUNDARY.length).replace(/^\s+/, "");
}

function cap(s) {
  return s && s.length > MAX_CHARS ? `${s.slice(0, MAX_CHARS)}…` : (s || "");
}

/* ------------------------------------------------------------------ */
/*  OpenClaw unified format helpers                                    */
/*                                                                     */
/*  Tool call:   assistant content block  type:"toolCall"              */
/*  Tool result: standalone message       role:"toolResult"            */
/* ------------------------------------------------------------------ */

/** Convert an OpenClaw toolCall block to EverMemOS tool_calls item */
function toToolCallItem(block) {
  const name = block.name || "unknown";
  const args = block.arguments;
  let argsStr = "";
  if (args != null) {
    try {
      argsStr = typeof args === "string" ? args : JSON.stringify(args);
    } catch {
      argsStr = String(args);
    }
  }
  return {
    id: block.id || undefined,
    type: "function",
    function: { name, arguments: argsStr },
  };
}

function toolResultContent(msg) {
  const c = msg.content;
  if (!c) return "";
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    return c.reduce((out, b) => {
      if (b?.type === "text" && b.text) return out ? `${out}\n${b.text}` : b.text;
      return out;
    }, "");
  }
  return "";
}

/* ------------------------------------------------------------------ */
/*  Entry conversion                                                   */
/* ------------------------------------------------------------------ */

function toEntries(msg) {
  if (!msg?.role) return [];

  /* --- toolResult message --- */
  if (msg.role === "toolResult") {
    const text = cap(toolResultContent(msg) || toText(msg.content));
    if (!text) return [];
    return [{
      role: "tool",
      tool_call_id: msg.toolCallId || undefined,
      content: text,
    }];
  }

  /* --- user message --- */
  if (msg.role === "user") {
    const text = cap(stripContext(toText(msg.content)));
    return text ? [{ role: "user", content: text }] : [];
  }

  /* --- assistant message --- */
  if (msg.role === "assistant") {
    const text = cap(toText(msg.content));
    const toolCalls = Array.isArray(msg.content)
      ? msg.content.filter((b) => b?.type === "toolCall").map(toToolCallItem)
      : [];
    if (!text && !toolCalls.length) return [];
    const effectiveText = text || toolCalls.map((tc) => `[tool_call: ${tc.function.name}]`).join(", ");
    const entry = { role: "assistant", content: effectiveText };
    if (toolCalls.length) entry.tool_calls = toolCalls;
    return [entry];
  }

  return [];
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * Collect the last turn (final user message and everything after it).
 * Skips over toolResult messages when searching for the real user message.
 */
export function collectMessages(messages) {
  let pivot = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") { pivot = i; break; }
  }
  if (pivot < 0) return [];

  return messages.slice(pivot).flatMap((m) => toEntries(m));
}

/* ------------------------------------------------------------------ */
/*  Session reset prompt detection                                     */
/* ------------------------------------------------------------------ */

export const BARE_SESSION_RESET_PROMPT =
  "A new session was started via /new or /reset. Execute your Session Startup sequence now - read the required files before responding to the user. Then greet the user in your configured persona, if one is provided. Be yourself - use your defined voice, mannerisms, and mood. Keep it to 1-3 sentences and ask what they want to do. If the runtime model differs from default_model in the system prompt, mention the default model. Do not mention internal steps, files, tools, or reasoning.";

/** Levenshtein edit distance (space-optimised O(m*n) time, O(n) space) */
function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const prev = Array.from({ length: n + 1 }, (_, i) => i);
  const curr = new Array(n + 1);
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      curr[j] = a[i - 1] === b[j - 1]
        ? prev[j - 1]
        : 1 + Math.min(prev[j - 1], prev[j], curr[j - 1]);
    }
    prev.splice(0, n + 1, ...curr);
  }
  return prev[n];
}

/**
 * Returns true when the query is within 20% length of BARE_SESSION_RESET_PROMPT
 * AND the edit-distance ratio is below 0.20 (i.e. ≥80% similar).
 */
export function isSessionResetPrompt(query) {
  if (!query) return false;
  const promptLen = BARE_SESSION_RESET_PROMPT.length;
  const queryLen = query.length;
  // Fast path: length must be within ±20% of the prompt
  if (Math.abs(queryLen - promptLen) / promptLen > 0.20) return false;
  const dist = levenshtein(query, BARE_SESSION_RESET_PROMPT);
  return dist / Math.max(queryLen, promptLen) < 0.20;
}
