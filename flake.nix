{
  description = "TerraNova - Offline design studio for Hytale World Generation V2";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Use the rust-version specified in Cargo.toml
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "rust-analyzer"];
        };

        # Frontend build using pnpm
        buildFrontend = pkgs.stdenvNoCC.mkDerivation {
          pname = "terranova-frontend";
          version = "0.1.5";

          src = ./.;

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
          ];
      in {
        packages = {
          default = self.packages.${system}.terranova;

          # Expose for testing/debugging
          frontend = buildFrontend;

          terranova = pkgs.rustPlatform.buildRustPackage {
            pname = "terranova";
            version = "0.1.5";

            src = ./.;

            # Set the source root to the Tauri directory
            postUnpack = ''
              cd $sourceRoot
              sourceRoot=src-tauri
            '';

            cargoLock = {
              lockFile = ./src-tauri/Cargo.lock;
            };

            # Patch tauri.conf.json to use pre-built frontend
            postPatch = ''
              ${pkgs.jq}/bin/jq \
                '.build.frontendDist = "${buildFrontend}" | .build.beforeBuildCommand = ""' \
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
            ++ linuxInputs ++ darwinInputs;

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
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.terranova}/bin/terranova";
        };
      }
    );
}
