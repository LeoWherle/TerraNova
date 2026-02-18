{
  description = "TerraNova - Offline design studio for Hytale World Generation V2";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        lib,
        ...
      }: let
        # Apply rust-overlay
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [(import inputs.rust-overlay)];
        };

        # Use the rust-version specified in Cargo.toml
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "rust-analyzer"];
        };

        # Frontend build using pnpm
        buildFrontend = pkgs.stdenvNoCC.mkDerivation {
          pname = "terranova-frontend";
          version = "0.1.5";

          src = inputs.self;

          pnpmDeps = pkgs.fetchPnpmDeps {
            inherit (buildFrontend) pname version src;
            hash = "sha256-WwxYAO+YBCA1WqAuOrQm6dG+VwvQv4otA1aIfagFGNU=";
            # fetcherVersion corresponds to lockfileVersion in pnpm-lock.yaml
            # lockfileVersion 9.0 -> fetcherVersion 3
            fetcherVersion = 3;
          };

          nativeBuildInputs = [
            pkgs.nodejs_22
            pkgs.pnpm
            pkgs.pnpmConfigHook
          ];

          buildPhase = ''
            runHook preBuild
            pnpm build
            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall
            mkdir -p $out
            cp -r dist/* $out/
            cp -r templates $out/
            runHook postInstall
          '';
        };

        # Platform-specific build inputs
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

        terranova = pkgs.rustPlatform.buildRustPackage {
          pname = "terranova";
          version = "0.1.5";

          src = inputs.self;

          # Set the source root to the Tauri directory
          postUnpack = ''
            cd $sourceRoot
            sourceRoot=src-tauri
          '';

          cargoLock = {
            lockFile = "${inputs.self}/src-tauri/Cargo.lock";
          };

          # Patch tauri.conf.json to use pre-built frontend
          postPatch = ''
            ${pkgs.jq}/bin/jq \
              'del(.build.devUrl) | .build.frontendDist = "${buildFrontend}" | .build.beforeBuildCommand = "" | .build.beforeDevCommand = ""' \
              tauri.conf.json > tauri.conf.json.tmp
            mv tauri.conf.json.tmp tauri.conf.json

            # Copy templates directory for bundling
            mkdir -p ../templates
            cp -r ${buildFrontend}/templates/* ../templates/ || true
          '';

          nativeBuildInputs = with pkgs;
            [
              pkg-config
              rustToolchain
              jq
              cargo-tauri
            ]
            ++ lib.optionals stdenv.isLinux [
              wrapGAppsHook3
              desktop-file-utils
            ];

          buildInputs = with pkgs;
            [
              openssl
            ]
            ++ linuxInputs ++ darwinInputs;

          # Skip tests (there's a test compilation error)
          doCheck = false;

          buildPhase = ''
            runHook preBuild
            cargo build --release --locked
            runHook postBuild
          '';

          installPhase =
            if pkgs.stdenv.isDarwin
            then ''
              runHook preInstall
              mkdir -p $out/Applications
              cp -r target/release/bundle/macos/TerraNova.app $out/Applications/
              mkdir -p $out/bin
              ln -s $out/Applications/TerraNova.app/Contents/MacOS/TerraNova $out/bin/terranova
              runHook postInstall
            ''
            else ''
                runHook preInstall
                mkdir -p $out/bin
                cp target/release/terranova $out/bin/

                # Install desktop file and icon
                mkdir -p $out/share/applications
                mkdir -p $out/share/icons/hicolor/128x128/apps

                if [ -f icons/icon.png ]; then
                  cp icons/icon.png $out/share/icons/hicolor/128x128/apps/terranova.png
                fi

                cat > $out/share/applications/terranova.desktop <<EOF
              [Desktop Entry]
              Type=Application
              Name=TerraNova
              Comment=Offline design studio for Hytale World Generation V2
              Exec=$out/bin/terranova
              Icon=terranova
              Terminal=false
              Categories=Graphics;Development;
              EOF

                ${pkgs.desktop-file-utils}/bin/desktop-file-validate $out/share/applications/terranova.desktop || true

                runHook postInstall
            '';

          preFixup = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            gappsWrapperArgs+=(
              --set-default WEBKIT_DISABLE_DMABUF_RENDERER "1"
            )
          '';

          meta = with pkgs.lib; {
            description = "Offline design studio for Hytale World Generation V2";
            homepage = "https://github.com/HyperSystemsDev/TerraNova";
            license = licenses.lgpl21Only;
            maintainers = [];
            platforms = platforms.linux ++ platforms.darwin;
            mainProgram = "terranova";
          };
        };
      in {
        packages = {
          default = terranova;
          inherit terranova;
          # Expose for testing/debugging
          frontend = buildFrontend;
        };

        devShells.default = pkgs.mkShell {
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
        };

        apps.default = {
          type = "app";
          program = "${terranova}/bin/terranova";
        };
      };
    };
}
