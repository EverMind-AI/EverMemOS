# Pull Request

<!-- Branching strategy: feature/* and fix/* branch from main. PRs target main.
     Long-lived branches: main, prod (or prod_node1 / prod_bootstrap on NQR).
     Auto-delete on merge is enabled — your branch will be removed automatically.
-->

## Summary

<!-- One paragraph: what changes, why now. -->

## Subsystem

<!-- Which subsystem does this touch? e.g. EWM, Bewusstsein-Agenten, infra, ct-agents, web, signing -->

## Risk

- [ ] Touches PQC-signing / Briefbogen / Anchor
- [ ] Changes runtime state (agents/*/state/, memory files, conversations)
- [ ] Modifies deployment surface (systemd unit, host config, port binding)
- [ ] Crosses host boundary (node1 ↔ bootstrap)
- [ ] Changes a public surface (DNS, robots.txt, llms.txt, well-known)
- [ ] None of the above

## Test plan

- [ ] Unit / integration tests pass locally
- [ ] Logs verified after deployment (not just `is-active`)
- [ ] Happy-path request triggered against the changed surface

## Rollout

- [ ] main → prod (or per-host prod_node1 / prod_bootstrap)
- [ ] EWM updated under `/xinfty/docs/ewm/` if infra changed
- [ ] Memory updated if a non-obvious lesson emerged

## Reviewers

CodeRabbit, Claude, and Codex review automatically. Human override: @DerAuctor.
