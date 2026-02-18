{
  inputs,
  pkgs,
  lib,
  ...
}: let
  rustToolchain = pkgs.rust-bin.stable.latest.default.override {
    extensions = ["rust-src" "rust-analyzer"];
  };

  darwinInputs = with pkgs;
    lib.optionals stdenv.isDarwin [
      darwin.apple_sdk.frameworks.AppKit
      darwin.apple_sdk.frameworks.WebKit
      darwin.apple_sdk.frameworks.Security
      darwin.apple_sdk.frameworks.CoreServices
      darwin.apple_sdk.frameworks.SystemConfiguration
    ];

  linuxInputs = with pkgs;
    lib.optionals stdenv.isLinux [
      glib-networking
      webkitgtk_4_1
      gtk3
      libsoup_3
      gsettings-desktop-schemas
    ];
in
  pkgs.mkShell {
    buildInputs = with pkgs;
      [
        # Rust toolchain
        rustToolchain
        cargo-tauri

        # Node.js and pnpm for frontend
        nodejs_22
        pnpm

        # Build dependencies
        pkg-config
        openssl

        # Additional development tools
        rust-analyzer
        jq
      ]
      ++ linuxInputs
      ++ darwinInputs
      # Ensure GSettings schemas are available
      ++ lib.optionals stdenv.isLinux [
        glib # for GSettings
      ];

    shellHook = ''
      echo "TerraNova development environment"
      echo "Rust version: $(rustc --version)"
      echo "Node version: $(node --version)"
      echo "pnpm version: $(pnpm --version)"
      echo ""
      echo "Available commands:"
      echo "  pnpm dev         - Start development server"
      echo "  pnpm build       - Build frontend"
      echo "  pnpm tauri dev   - Run Tauri in development mode"
      echo "  pnpm tauri build - Build Tauri application"
      echo ""
      echo "Nix flake commands:"
      echo "  nix build        - Build the complete application"
      echo "  nix run          - Run the application"
      echo "  nix build .#frontend - Build only the frontend"
    '';

    RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";

    # Set WebKit environment variable for Linux
    WEBKIT_DISABLE_DMABUF_RENDERER = pkgs.lib.optionalString pkgs.stdenv.isLinux "1";

    # Ensure GSettings schemas are found
    XDG_DATA_DIRS = pkgs.lib.optionalString pkgs.stdenv.isLinux "${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/${pkgs.gsettings-desktop-schemas.name}:${pkgs.gtk3}/share/gsettings-schemas/${pkgs.gtk3.name}:$XDG_DATA_DIRS";
  }
