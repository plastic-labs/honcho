{
  description = "Honcho — Infrastructure for AI agents with memory and social cognition";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    let
      inherit (flake-utils.lib) eachDefaultSystem;

      # Shared nixosModule: injects pkgs.honcho via overlay, then imports ./nix/module.nix
      nixosModule = { config, lib, pkgs, ... }: {
        nixpkgs.overlays = [
          (final: prev: {
            honcho = self.packages.${pkgs.system}.honcho;
          })
        ];
        imports = [ ./nix/module.nix ];
      };
    in
    eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = false;
        };

        python = pkgs.python313;
        lib = pkgs.lib;

        honcho = import ./nix/build.nix {
          inherit lib python;
          src = lib.cleanSource ./.;
        };

      in
      {
        packages = {
          inherit honcho;
          default = honcho;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            python
            postgresql_16
            redis
          ];
          shellHook = ''
            echo "Honcho dev shell"
            echo "  uv:    $(uv --version)"
            echo "  nix:   $(nix --version)"
          '';
        };
      }
    ) // {
      nixosModules.default = nixosModule;
      nixosModules.honcho = nixosModule;
      overlays.default = final: prev: {
        honcho = self.packages.${final.stdenv.hostPlatform.system}.honcho;
      };
    };
}
