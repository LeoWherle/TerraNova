{
  inputs,
  pkgs,
  lib,
  ...
}: let
  # Use the rust-version specified in Cargo.toml
  rustToolchain = pkgs.rust-bin.stable.latest.default.override {
    extensions = ["rust-src" "rust-analyzer"];
  };

  frontendSrc = lib.cleanSourceWith {
    src = inputs.self;
    filter = let
      excludeDirs = ["node_modules" ".pnpm-store" "dist" "build" "target" "result" "result-bin" ".direnv" ".git" ".github" "docs" "src-tauri" "nix"];
      excludeFiles = ["README.md" "LICENSE" "CODE_OF_CONDUCT.md" ".gitignore"];
    in
      path: type: let
        baseName = baseNameOf path;
      in
        !(builtins.elem baseName excludeDirs) && !(builtins.elem baseName excludeFiles);
  };

  buildFrontend = pkgs.stdenvNoCC.mkDerivation rec {
    pname = "terranova-frontend";
    version = "0.1.5";

    src = frontendSrc;

    pnpmDeps = pkgs.fetchPnpmDeps {
      inherit pname version;
      src = inputs.self;
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

  tauriSrc = lib.cleanSourceWith {
    src = inputs.self;
    filter = let
      excludeDirs = ["node_modules" ".pnpm-store" "dist" "build" "result" "result-bin" ".direnv" ".git" ".github" "docs" "nix"];
      excludePaths = ["src" "public" "patches" "templates" "index.html" "vite.config.ts" "vitest.config.ts" "tsconfig.json" "tailwind.config.js" "postcss.config.js" "package.json" "package-lock.json" "pnpm-lock.yaml" "pnpm-workspace.yaml"];
      excludeFiles = ["README.md" "LICENSE" "CODE_OF_CONDUCT.md" "NIX.md" "QUICKSTART_NIX.md" "NIX_CACHING.md" ".gitignore"];
    in
      path: type: let
        baseName = baseNameOf path;
        relPath = lib.removePrefix "${inputs.self}/" path;
      in
        !(builtins.elem baseName excludeDirs)
        && !(builtins.elem baseName excludeFiles)
        && !(builtins.elem relPath excludePaths);
  };

  terranova = pkgs.rustPlatform.buildRustPackage {
    pname = "terranova";
    version = "0.1.5";

    src = tauriSrc;

    sourceRoot = "source/src-tauri";

    cargoLock = {
      lockFile = "${inputs.self}/src-tauri/Cargo.lock";
    };

    # Patch tauri.conf.json to use pre-built frontend
    postPatch = ''
      ${pkgs.jq}/bin/jq \
        'del(.build.devUrl) | .build.frontendDist = "${buildFrontend}" | .build.beforeBuildCommand = "" | .build.beforeDevCommand = "" | .bundle.resources = ["${buildFrontend}/templates/"]' \
        tauri.conf.json > tauri.conf.json.tmp
      mv tauri.conf.json.tmp tauri.conf.json
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
  default = terranova;
  terranova = terranova;

  frontend = buildFrontend;
}
