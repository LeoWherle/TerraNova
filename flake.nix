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
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [(import inputs.rust-overlay)];
        };
      in {
        packages = import ./nix/packages.nix {
          inherit inputs pkgs lib;
        };

        devShells.default = import ./nix/devShell.nix {
          inherit inputs pkgs lib;
        };

        apps.default = {
          type = "app";
          program = "${config.packages.terranova}/bin/terranova";
        };
      };
    };
}
