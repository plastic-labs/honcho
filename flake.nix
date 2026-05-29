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

        honcho = python.pkgs.buildPythonApplication rec {
          pname = "honcho";
          version = "3.0.7";
          pyproject = true;

          src = pkgs.lib.cleanSource ./.;

          # The root pyproject.toml lacks a [build-system] section (uv-managed),
          # so we inject one during the build.
          postPatch = ''
            if ! grep -q '\[build-system\]' pyproject.toml; then
              cat >> pyproject.toml << 'PYEOF'

[build-system]
requires = ["setuptools>=75.0", "wheel"]
build-backend = "setuptools.build_meta"
PYEOF
            fi
            # Remove workspace members from pyproject.toml to avoid
            # setuptools trying to discover them as sub-packages
          '';

          nativeBuildInputs = with python.pkgs; [
            setuptools
            wheel
          ];

          propagatedBuildInputs = with python.pkgs; [
            alembic
            cashews
            cloudevents
            fastapi
            fastapi-pagination
            google-genai
            greenlet
            httpx
            json-repair
            langfuse
            lancedb
            nanoid
            openai
            pdfplumber
            pgvector
            prometheus-client
            psycopg
            pyarrow
            pydantic
            pydantic-settings
            pyjwt
            python-dotenv
            redis
            rich
            scikit-learn
            sentry-sdk
            sqlalchemy
            tenacity
            tiktoken
            typing-extensions
          ] ++ lib.optional (builtins.hasAttr "turbopuffer" python.pkgs) python.pkgs.turbopuffer;

          pythonImportsCheck = [ ];

          # Tests require a running PostgreSQL + Redis + LLM API keys
          doCheck = false;

          # turbopuffer is listed in pyproject.toml but only conditionally
          # available in nixpkgs. Skip the runtime deps check.
          dontCheckRuntimeDeps = true;

          meta = {
            description = "Infrastructure for AI agents with memory and social cognition";
            homepage = "https://honcho.dev";
            license = pkgs.lib.licenses.asl20;
            maintainers = with pkgs.lib.maintainers; [ ];
            platforms = pkgs.lib.platforms.linux;
            mainProgram = "honcho";
          };
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
