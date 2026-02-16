# AI Transparency & Usage Disclaimer

> **Last Updated:** February 2026
> **Author:** Zenith
> **Status:** Living Document, updated as the project evolves

---

## Overview

TerraNova is built with significant assistance from AI tooling, specifically Claude by Anthropic. Approximately **70%** of the codebase is AI-generated. This document exists to be fully transparent about how AI is used, where it's used, what role I play as the developer, and how security is handled.

I couldn't have built TerraNova as fast as I did without AI, and I have no interest in hiding that. What matters is that the process is honest, the code is reviewed, and the end result serves the creators who use it.

---

## Development Workflow

Every feature in TerraNova follows the same structured pipeline. Every feature starts as my idea, and AI supports the process along the way.

### The Process

1. **Brainstorm** – I identify the feature, the problem it solves, and how it fits into TerraNova's vision. This is entirely human-driven.
2. **Feature Planning** – I scope out requirements, edge cases, and how the feature interacts with existing systems.
3. **Documentation & Outlining** – I write docs outlining the feature's purpose, expected behavior, and technical constraints.
4. **Roadmap Placement** – The feature is prioritized and slotted into the development roadmap.
5. **Development with Claude** – This is where the actual coding happens, and it varies depending on the feature. Sometimes I write the initial code and bring Claude in to expand on it. Other times I outline what I need and Claude writes the first pass, which I then review, modify, and build on. It's a back-and-forth collaboration, not a one-way street in either direction.
6. **Code Review with Claude** – Once the feature is working, I run it back through Claude for a second pass to review logic, catch issues, and suggest improvements.
7. **Bug Squashing** – I identify and fix bugs through testing and iteration, often collaborating with Claude to diagnose and resolve issues.
8. **Optimization** – Performance tuning, refactoring, and cleanup.
9. **Ship** – The feature is merged and released.

### Where AI Is Used

| Area | AI Involvement | Notes |
|------|---------------|-------|
| TypeScript code | Moderate to Heavy | I'm primarily a TS dev, so I understand the vast majority of what's written. Sometimes I write the code and Claude refines it, sometimes Claude writes it and I refine it. Either way, I review and understand roughly 80% of it deeply. |
| Rust code | Heavy | I had minimal Rust experience prior to TerraNova. Claude writes most of the Rust code based on my direction. I review and learn from it. My independent comprehension sits around 20-30%, mostly general syntax and logic flow. |
| Complex math/simulation logic | Heavy | Many of the mathematical models and simulation calculations are written by Claude based on my requirements. This work wouldn't have been feasible without AI. |
| Architecture & planning | Light | I drive all architecture decisions. Claude helps expand on technical specifics. |
| Documentation | Light | I write the core docs. Claude assists with formatting and expansion. |
| Bug fixing & optimization | Moderate | A collaborative process. I identify issues and work with Claude to diagnose and fix them. |

### What I Do

I want to be honest about what this looks like in practice. Claude writes a lot of the code. That's the reality, and I'm not going to downplay it. But every feature starts as my idea, and I am actively involved in every stage of development.

Here's what I bring to every feature:

- I define every feature, its purpose, and its scope
- I make all architectural and design decisions
- I write code and build on what Claude generates, or vice versa depending on the feature
- I review every piece of AI-generated code before it ships
- I test, debug, and iterate until things work correctly
- I maintain the project vision and roadmap entirely on my own

Claude is my most valuable collaborator on this project. It helps me write code, review code, solve problems, and build things faster than I could alone. But the direction, the decisions, and the accountability are all mine.

---

## On Security

I want to be upfront here: I am not a senior software engineer. The codebase is functional and actively maintained, but I'm not going to sit here and claim it's bulletproof.

This is one of the core reasons TerraNova has remained in **closed alpha**. The project is starting to gain real traction, which is genuinely exciting, but I care too much about this community to distribute something widely before it's ready.

### What I'm Doing About It

- **Security audit planned** – I intend to get a formal security audit from a qualified engineer once the product is in a more stable state.
- **Ongoing review** – All code, whether AI-generated or hand-written, goes through review before being merged.
- **Responsible scope** – Features are released incrementally so I can monitor for issues and respond quickly.
- **Community accountability** – If you find something, please report it. I take every single report seriously.

### Open Source as a Security Feature

TerraNova is fully open-source. That means every line of code, AI-generated or otherwise, is publicly visible and inspectable. You never have to take my word for it on security. You can read the source yourself.

Open-sourcing this project is a deliberate choice. It invites the community to audit, contribute, and hold the project accountable in a way that closed-source software simply cannot. If there's a vulnerability, anyone can find it, flag it, and help fix it. That kind of collective transparency is arguably more powerful than any single security audit.

I wouldn't have open-sourced TerraNova if I didn't have genuine understanding of and involvement in the code.

---

## Why This Document Exists

A community member raised concerns about the extent of AI usage in TerraNova and the potential security implications that come with it. After a conversation with Xaphedo, it became clear that the community deserves full visibility into how this project is built.

This isn't a response to criticism. It's a commitment to the people who are investing their time and trust in this tool. I believe AI is incredibly powerful for independent developers, and using it openly and responsibly is better for everyone than pretending it doesn't exist.

At the end of the day, my goal with TerraNova has always been to give creators the same feeling I got when I first used VoxelSniper. That sense of pure magic when a tool just clicks and suddenly you can build things you never thought possible. AI is helping me get there faster, and I want every one of you along for the ride.

---

## Questions or Concerns

If you have questions about AI usage, security, or anything else about how TerraNova is built, please don't hesitate to reach out. I'm an open book on this, always.

---

*This document will be updated as the project matures, security audits are completed, and development practices evolve.*
