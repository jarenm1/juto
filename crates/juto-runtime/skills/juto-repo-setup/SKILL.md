---
name: juto-repo-setup
description: Initialize a new or existing target repository for effective Juto use. Use for repo onboarding, refreshing agent instructions, or preparing multi-agent collaboration across operating systems.
---

# Initialize a repository for Juto

Prepare the **target repository**, not Juto's own source tree, for effective agent
work. Init creates or improves repository instructions and, where supported,
agent definitions for frequent multi-agent collaboration. Adapt to the repo's
age, stack, operating systems, and existing conventions.

## Procedure

1. **Discover before writing.** Resolve the target root and inspect uncommitted
   work. Read applicable repository instructions and the relevant README,
   manifests, scripts, CI, and existing agent definitions. Identify the purpose,
   component boundaries, supported environments, verification entry points, and
   areas requiring coordinated edits. Preserve human work and local policy.
   Complete when each finding has a repository source or is explicitly unknown.

2. **Choose the right starting point.**
   - **Existing repo:** patch its instructions and agent definitions in place;
     retain useful constraints and replace stale guidance only with evidence.
   - **New or empty repo:** establish a lean instruction file and collaboration
     workflow. Describe only the structure and tooling actually present or
     explicitly chosen by the user. Record an undecided stack or missing checks
     as such; init is not permission to scaffold an application.
   - **Repeat init:** update gaps and stale facts without duplicating sections,
     creating competing instruction files, or replacing custom agent roles.
   Init authorizes these instruction edits. Ask only for unresolved product
   choices or changes beyond setup, such as installing dependencies or changing
   the toolchain, CI, or application code.

3. **Write a concise, Juto-readable entry point.** Prefer root `AGENTS.md`, or
   the repo's existing `.juto/AGENTS.md`. Preserve instructions maintained for
   other tools; consolidate essential guidance into the Juto-readable entry
   point rather than assuming another tool's file or import syntax is loaded.
   Record component ownership and non-obvious constraints, with pointers to
   authoritative workflow docs or scripts. Include actionable verification
   entry points and prerequisites; avoid copying an entire directory listing
   or restating every manifest command. Add scoped instructions only when a
   component genuinely needs different rules.

4. **Make multi-agent work explicit.** Add or update a collaboration section
   covering the following, adapted to this repo:
   - **Decomposition:** delegate independent, bounded slices with a goal,
     owned files/components, shared interface contracts, acceptance criteria,
     and required verification. Keep small or tightly coupled work together.
   - **Ownership:** one integration owner controls shared VCS state, dependency
     manifests/lockfiles, and integration. Assign nonoverlapping write ownership;
     sequence overlapping edits or return a proposal to the owner.
   - **Isolation:** use the existing VCS and supported worktree/workspace
     mechanism when available, with a documented starting revision. Otherwise
     serialize writes or use explicit nonoverlapping ownership. Give concurrent
     workers separate writable caches, temporary files, ports, and services.
   - **Roles:** identify read-only discovery, implementation, independent review,
     and integration responsibilities. Modify existing agent/persona definitions
     when their supported format is known and the repo needs specialization;
     otherwise keep these responsibilities in the instruction file. Create a
     reusable specialist only for a real recurring repo-specific job, using a
     format the installed tooling actually supports.
   - **Handoff:** return changed files, decisions, verification evidence, and
     blockers. Surface contract conflicts to the owner before conflicting edits.
     The owner integrates, resolves conflicts, and runs final verification;
     individual agents do not claim integration from a local pass.

5. **Make the workflow OS-aware.** Discover the target's supported operating
   systems and the current host's shell. Reuse portable project tooling. Where
   commands differ, label OS/shell-specific alternatives and prerequisites,
   including paths, quoting, environment variables, and process cleanup. Use
   the repo's existing isolation/resource controls where supported; document
   unavailable controls and a safe reduced-concurrency workflow when needed.
   Juto's own stack is not a prerequisite: do not prescribe Rust, Nix, jj,
   systemd, a particular wrapper path, or GPU tooling unless the target requires
   it. Distinguish the repo's supported platforms from hosts actually checked;
   never promise runtime portability based on instructions alone.

6. **Validate and hand off.** Check that documented paths and commands exist,
   instructions are discoverable by Juto, ownership rules are consistent, and
   a repeated init would leave one coherent workflow. On the current host, run
   a safe representative documented check and observe its result, using required
   repository resource controls. For an empty repo with no runnable workflow,
   validate instruction discovery and report that limitation. Do not manufacture
   passing checks or run credential-dependent/destructive setup. Report edited
   files, how agents should split and integrate work, exact checks and results,
   and missing prerequisites or unverified OS branches.

## Completion criteria

The target has concise, discoverable instructions grounded in its actual state;
multi-agent ownership, isolation, roles, handoffs, and final verification are
actionable; existing custom guidance is preserved; reruns avoid duplication;
OS-specific requirements and unverified branches are explicit. No unrelated
application, dependency, CI, or runtime changes are part of init.
