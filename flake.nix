{
  description = "Honcho — Infrastructure for AI agents with memory and social cognition";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    let
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
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = false;
        };

        lib = pkgs.lib;

        # ---- uv2nix: read uv.lock and build dependency overlay ----

        # Load the uv workspace (parses uv.lock + pyproject.toml for all members)
        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ./.;
        };

        # mkPyprojectOverlay pins every dependency from uv.lock at the exact version
        uvOverlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Create the base Python package set using pyproject-nix's build
        # infrastructure (not nixpkgs' buildPythonPackage).
        basePythonSet = pkgs.callPackage pyproject-nix.build.packages {
          python = pkgs.python313;
        };

        # Compose overlays:
        #  1. pyproject-build-systems — provides resolveBuildSystem hooks
        #  2. uvOverlay — pins all dependency versions from uv.lock
        pythonSet = basePythonSet.overrideScope (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            uvOverlay
          ]
        );

        # ---- honcho: the Python library (built via uv2nix) ----
        # pythonSet.honcho is the uv2nix-built package installed to
        # $out/lib/python3.13/site-packages/.
        honchoLib = pythonSet.honcho;

        # ---- honcho: deployable package ----
        # Wraps the uv2nix-built library with the required bin/ and scripts/
        # so the NixOS module can reference them.
        honcho = pkgs.stdenv.mkDerivation {
          pname = "honcho";
          version = "3.0.7";
          src = lib.cleanSource ./.;

          dontUnpack = true;
          dontBuild = true;
          dontFixup = true;

          nativeBuildInputs = [ pkgs.makeWrapper pkgs.python313 ];

          installPhase = ''
            mkdir -p $out/lib/python3.13/site-packages $out/bin $out/scripts

            # Copy the uv2nix-built library
            cp -r --no-preserve=mode ${honchoLib}/lib/python3.13/site-packages/* \
              $out/lib/python3.13/site-packages/

            # Copy scripts from source (provision_db.py)
            cp -r $src/scripts/* $out/scripts/

            # Wrap python with the right PYTHONPATH
            makeWrapper ${pkgs.python313}/bin/python $out/bin/python \
              --set PYTHONPATH $out/lib/python3.13/site-packages \
              --set-default HONCHO_CONFIG /etc/honcho/config.toml

            # Symlink fastapi CLI (comes from the fastapi[standard] dep)
            ln -s ${pythonSet.fastapi}/bin/fastapi $out/bin/fastapi

            # Wrap remaining entry points from site-packages bin dir
            if [ -d "${honchoLib}/bin" ]; then
              for f in ${honchoLib}/bin/*; do
                name=$(basename "$f")
                makeWrapper "$f" "$out/bin/$name" \
                  --set PYTHONPATH $out/lib/python3.13/site-packages
              done
            fi
          '';

          meta = {
            description = "Infrastructure for AI agents with memory and social cognition";
            homepage = "https://honcho.dev";
            license = lib.licenses.asl20;
            maintainers = with lib.maintainers; [ ];
            platforms = lib.platforms.linux;
          };
        };

      in
      {
        packages = {
          inherit honcho;
          default = honcho;
          # uv2nix-built library (for debugging/composition)
          lib = honchoLib;
          # Deps-only virtualenv for development
          deps = pythonSet.mkVirtualEnv "honcho-deps" workspace.deps.default;
        };

        devShells.default =
          let
            devEnv = pythonSet.mkVirtualEnv "honcho-dev" workspace.deps.all;
          in
          pkgs.mkShell {
            packages = with pkgs; [
              uv
              devEnv
              postgresql_16
              redis
            ];
            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON_DOWNLOADS = "never";
            };
            shellHook = ''
              unset PYTHONPATH
              echo "Honcho dev shell (uv2nix)"
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
