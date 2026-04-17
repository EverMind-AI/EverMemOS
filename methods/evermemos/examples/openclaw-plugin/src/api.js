import { createHash } from "node:crypto";
import { request } from "./http.js";

const noop = { info() {}, warn() {} };
const TAG = "[evermind-ai-everos]";

/** Generate a deterministic message ID scoped by idSeed.
 *  Same seed + role + content always produces the same ID.
 *  Different seeds (different turns/sessions) produce different IDs,
 *  so repeated short messages like "ok" won't collide across turns. */
function messageId(idSeed, role, content) {
  const hash = createHash("sha256").update(`${idSeed}:${role}:${content}`).digest("hex").slice(0, 24);
  return `em_${hash}`;
}

export async function searchMemories(cfg, params, log = noop) {
  const { memory_types, user_id, group_id, ...baseParams } = params;

  const SEARCHABLE = new Set(["episodic_memory"]);
  const searchTypes = (memory_types ?? []).filter((t) => SEARCHABLE.has(t));

  if (!searchTypes.length) {
    return { status: "ok", result: { memories: [], pending_messages: [] } };
  }

  // v1 API: user_id/group_id must be inside filters object
  const p = {
    ...baseParams,
    memory_types: searchTypes,
    filters: { user_id, group_id },
  };
  log.info(`${TAG} POST /api/v1/memories/search`);
  const r = await request(cfg, "POST", "/api/v1/memories/search", p);
  log.info(`${TAG} POST response`);

  return {
    status: "ok",
    result: {
      memories: r?.data?.episodes ?? [],
      pending_messages: [],
    },
  };
}

export async function saveMemories(cfg, { userId, groupId, messages = [], flush = false, idSeed = "" }) {
  if (!messages.length) return;
  const stamp = Date.now();

  // v1 API: batch format with user_id at top level, messages in array
  const payload = {
    user_id: userId,
    messages: messages.map((msg, i) => {
      const { role = "user", content = "" } = msg;
      return {
        content,
        timestamp: stamp + i,
        role,
        sender_name: role === "assistant" ? "assistant" : userId,
      };
    }),
  };

  await request(cfg, "POST", "/api/v1/memories", payload);

  if (flush) {
    await request(cfg, "POST", "/api/v1/memories/flush", { user_id: userId });
  }
}
