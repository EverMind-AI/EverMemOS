# Use Cases

This folder contains apps, demos, and integrations that show what persistent memory enables in real products and workflows. Some examples are complete local projects; others are focused integrations that can be adapted into your own agent stack.

## Included Use Cases

| Use case | What it shows | Start here |
| :--- | :--- | :--- |
| **Claude Code Plugin** | Persistent memory for Claude Code, including hooks, commands, local services, and memory recall. | [claude-code-plugin/](claude-code-plugin/) |
| **Game of Thrones Memories** | An interactive Q&A demo over long-form story memory. | [game-of-throne-demo/](game-of-throne-demo/) |
| **OpenHER** | Memory-oriented app and integration examples. | [openher/](openher/) |

Use the top-level [Use Cases](../README.md#use-cases) section for the visual catalogue of demos and external integrations.

## Contributing Guidelines

Before submitting a use case, please follow these best practices to keep the repository clean and lightweight.

### No images in the repo

Do not commit image files (`.png`, `.jpg`, `.gif`, `.svg`, etc.) to this repository. Images bloat the Git history and cannot be removed after the fact.

Instead, upload images to [GitHub user-attachments](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#uploading-assets) and reference them by URL in your README:

```markdown
![Screenshot](https://github.com/user-attachments/assets/your-image-id)
```

### No generated or dependency files

Do not commit files that can be regenerated locally:

- `node_modules/` — run `npm install` to regenerate
- `package-lock.json` — already in `.gitignore` for this folder
- `dist/`, `build/`, `.next/` — build output
- `.env` — use `.env.example` with placeholder values instead

### Keep code DRY

Avoid duplicating logic across your use case. If multiple files share the same functionality, extract it into a shared utility. This makes examples easier to follow and maintain.

### General checklist

Before opening a PR, verify:

- [ ] No image files committed (use external URLs)
- [ ] No `node_modules`, lock files, or build artifacts
- [ ] No secrets or API keys (only `.env.example` with placeholders)
- [ ] README included with setup instructions
- [ ] Code is concise and avoids unnecessary repetition
