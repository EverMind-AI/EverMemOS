# AGENTS.md

This repository is organized around the same reader journey as the top-level
README:

1. **Use cases** show what persistent memory enables in real products and
   workflows.
2. **Quick Start** gets EverCore running locally.
3. **Architecture methods** document the memory systems included in EverOS.
4. **Benchmarks** and **Evaluation** show how to measure and reproduce results.

## Project Map

- `methods/EverCore/` - long-term memory operating system for agents.
- `methods/HyperMem/` - hypergraph-based hierarchical memory architecture.
- `benchmarks/EverMemBench/` - memory quality evaluation.
- `benchmarks/EvoAgentBench/` - agent self-evolution evaluation.
- `use-cases/` - apps, demos, and integrations built on top of the memory layer.

## README Guidance

- Keep the top-level README flow smooth: overview, use cases, quick start,
  architecture methods, benchmarks, evaluation, citations, community.
- Avoid repeating the three-part project framing after the overview. Later
  sections should act as catalogues or action paths.
- Use root-relative links in the README, and verify that active relative links
  resolve before finishing.
- Keep commented-out README blocks out unless they are intentionally preserved
  for a near-term restoration.

## Quick Commands

```bash
cd methods/EverCore
docker compose up -d          # Start infrastructure
uv sync                       # Install dependencies
uv run python src/run.py      # Run application
make test                     # Run tests
make lint                     # Run formatting/i18n checks
uv run pyright                # Type check, if pyright is installed
```

## Key Entry Points

- `methods/EverCore/src/run.py` - EverCore application entry.
- `methods/EverCore/src/agentic_layer/memory_manager.py` - core memory manager.
- `methods/EverCore/src/infra_layer/adapters/input/api/` - REST API controllers.
- `methods/EverCore/docs/` - EverCore setup, usage, and architecture docs.
- `methods/EverCore/evaluation/` - EverCore evaluation runner and reports.

## Development Notes

- All I/O is async; use `await`.
- EverCore is multi-tenant; data must remain tenant-scoped.
- Prompts live in `methods/EverCore/src/memory_layer/prompts/` with EN/ZH
  variants.
- Prefer existing repo patterns and component boundaries before adding new
  abstractions.
