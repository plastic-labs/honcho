{ lib, python, src }:

python.pkgs.buildPythonApplication rec {
  pname = "honcho";
  version = "3.0.7";
  pyproject = true;

  inherit src;

  # The root pyproject.toml lacks a [build-system] section (uv-managed),
  # so we inject one during the build.
  postPatch = ''
    if ! grep -q '\\[build-system\\]' pyproject.toml; then
      cat >> pyproject.toml << 'PYEOF'

[build-system]
requires = ["setuptools>=75.0", "wheel"]
build-backend = "setuptools.build_meta"
PYEOF
    fi
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
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    platforms = lib.platforms.linux;
    mainProgram = "honcho";
  };
}
