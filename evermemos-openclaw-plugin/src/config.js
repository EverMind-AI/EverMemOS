const DEFAULT_URL = "http://localhost:1995";

export const TIMEOUT_MS = 60000;

export function resolveConfig(pc = {}) {
  return {
    serverUrl: (pc.baseUrl || DEFAULT_URL).replace(/\/*$/, ""),
    userId: pc.userId || "evermemos-user",
    groupId: pc.groupId || "evermemos-group",
    topK: pc.topK ?? 5,
    memoryTypes: pc.memoryTypes ?? ["episodic_memory", "profile", "agent_skill", "agent_case"],
    retrieveMethod: pc.retrieveMethod ?? "hybrid",
  };
}
