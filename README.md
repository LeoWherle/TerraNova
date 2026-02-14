# TerraNova

[![Latest Release](https://img.shields.io/github/v/release/HyperSystemsDev/TerraNova?label=version)](https://github.com/HyperSystemsDev/TerraNova/releases)
[![License](https://img.shields.io/badge/license-LGPL--2.1-green)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?logo=discord&logoColor=white)](https://discord.gg/NHPzsjkQeu)
[![GitHub Stars](https://img.shields.io/github/stars/HyperSystemsDev/TerraNova?style=social)](https://github.com/HyperSystemsDev/TerraNova)

**Offline design studio for Hytale World Generation V2.** Visual node editor, live terrain preview, and validated JSON export — no server required.

**[Releases](https://github.com/HyperSystemsDev/TerraNova/releases)** | **[Discord](https://discord.gg/NHPzsjkQeu)** | **[Docs](docs/planning/ABOUT.md)**

![TerraNova](docs/images/header.webp)

## Features

**Node-Based Editor** — Drag-and-drop all 200+ V2 types with category-colored nodes, auto-layout, minimap, and search.

**Live Preview** — Real-time 2D heatmaps with contour lines and 3D voxel terrain with SSAO, updated as you tweak sliders.

**Comparison View** — Side-by-side before/after preview for iterating on terrain changes.

**Biome Editor** — Edit Terrain, Material, Pattern, Position, and Prop sections within a single biome file with per-section undo history.

**Curve Editor** — Draggable control points for spline and curve functions.

**Material Layers** — Visual stack editor for material providers with layer ordering.

**Template System** — 10 bundled templates (Forest, Desert, Mountains, Floating Islands, and more) or create your own.

**Schema Validation** — Real-time diagnostics on type errors, missing fields, and invalid ranges with severity badges on nodes.

**Bridge Integration** — Push exported asset packs directly to a running Hytale server.

**Offline-First** — No server, no internet, no telemetry. All data stays on your machine.

## Quick Start

1. Download the latest release from [Releases](https://github.com/HyperSystemsDev/TerraNova/releases)
2. Open TerraNova, pick a template or start from scratch
3. Design your worldgen using the node editor and live preview
4. Export and drop the JSON files in your server's `mods/` folder

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo |
| `Ctrl+S` | Save |
| `Ctrl+F` | Search nodes |
| `Ctrl+A` | Select all |
| `Delete` | Delete selected |
| `Ctrl+D` | Duplicate selected |
| `Ctrl+L` | Auto-layout graph |
| `Space` (drag) | Pan canvas |
| `Scroll` | Zoom |

<details>
<summary><strong>All shortcuts</strong></summary>

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New file |
| `Ctrl+O` | Open file |
| `Ctrl+Shift+S` | Save as |
| `Ctrl+E` | Export JSON |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+P` | Toggle preview |
| `Ctrl+G` | Toggle grid |
| `Ctrl+M` | Toggle minimap |
| `Escape` | Deselect / close dialog |

</details>

## Development

<details>
<summary><strong>Prerequisites & setup</strong></summary>

**Requirements:** [Node.js](https://nodejs.org/) 20+, [pnpm](https://pnpm.io/) 9+, [Rust](https://rustup.rs/) 1.77+

```bash
# Install frontend dependencies
pnpm install

# Launch in development mode (opens app window with hot reload)
pnpm tauri dev
```

### Build

```bash
# Build production bundle
pnpm tauri build
# Output: src-tauri/target/release/bundle/
```

</details>

## For Developers

<details>
<summary><strong>Project structure</strong></summary>

```
TerraNova/
├── src/                    # React frontend (TypeScript)
│   ├── components/         # UI components (editor, preview, layout, home)
│   ├── nodes/              # React Flow node components per V2 category
│   │   ├── density/        #   68 density function node types
│   │   ├── material/       #   14 material provider node types
│   │   ├── curves/         #   19 curve node types
│   │   └── shared/         #   BaseNode, handle registry, layout
│   ├── schema/             # TypeScript type definitions for V2 assets
│   ├── stores/             # Zustand state stores
│   ├── hooks/              # Custom React hooks
│   └── utils/              # Graph conversion, layout, colormaps
├── src-tauri/              # Tauri + Rust backend
│   └── src/
│       ├── commands/       #   Tauri command handlers
│       ├── schema/         #   Rust V2 schema types (serde)
│       ├── noise/          #   Density function evaluator
│       └── io/             #   Asset pack I/O and template system
├── templates/              # Bundled world generation templates
└── public/                 # Static assets
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Framework | [Tauri 2](https://tauri.app/) |
| Frontend | [React 19](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/) |
| Node Editor | [@xyflow/react](https://reactflow.dev/) (React Flow) |
| 3D Preview | [React Three Fiber](https://r3f.docs.pmnd.rs/) + [Three.js](https://threejs.org/) |
| State | [Zustand](https://zustand.docs.pmnd.rs/) |
| Styling | [Tailwind CSS](https://tailwindcss.com/) |
| Build | [Vite](https://vite.dev/) |
| Backend | [Rust](https://www.rust-lang.org/) |

</details>

## Links

- [Releases](https://github.com/HyperSystemsDev/TerraNova/releases) — Downloads
- [Discord](https://discord.gg/NHPzsjkQeu) — Support & community
- [Issues](https://github.com/HyperSystemsDev/TerraNova/issues) — Bug reports & features
- [HyperSystemsDev](https://github.com/HyperSystemsDev) — Organization

## License

[LGPL-2.1](LICENSE)

---

Part of the **HyperSystems** suite: [HyperPerms](https://github.com/HyperSystemsDev/HyperPerms) | [HyperHomes](https://github.com/HyperSystemsDev/HyperHomes) | [HyperFactions](https://github.com/HyperSystemsDev/HyperFactions) | [TerraNova](https://github.com/HyperSystemsDev/TerraNova)
