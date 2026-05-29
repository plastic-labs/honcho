{ config, lib, pkgs, ... }:

let
  cfg = config.services.honcho;
  inherit (lib)
    mkEnableOption
    mkOption
    mkIf
    types
    literalExpression
    escapeURL
    hasPrefix
    optionals
    ;

  tomlFormat = pkgs.formats.toml { };

  # URL-encode user and password for safe URI interpolation
  encodedUser = escapeURL cfg.database.user;
  encodedPassword = escapeURL cfg.database.password;

  # Detect Unix socket path (starts with "/")
  isSocket = hasPrefix "/" cfg.database.host;

  # Build the connection URI from components
  dbConnectionURI =
    if cfg.database.password != "" then
      if isSocket then
        "postgresql+psycopg://${encodedUser}:${encodedPassword}@/${cfg.database.name}?host=${cfg.database.host}"
      else
        "postgresql+psycopg://${encodedUser}:${encodedPassword}@${cfg.database.host}:${toString cfg.database.port}/${cfg.database.name}"
    else
      "postgresql+psycopg:///${cfg.database.name}?host=${cfg.database.host}";

  cacheURL = "redis://${cfg.cache.host}:${toString cfg.cache.port}/${toString cfg.cache.db}?suppress=true";

  # Write out the merged config.toml
  configFile = tomlFormat.generate "honcho-config.toml" cfg.settings;

  # Only non-secret env vars go in baseEnv — secrets leak into the world-readable
  # Nix store via systemd unit derivations.  DB_CONNECTION_URI embeds
  # cfg.database.password when one is set, so we exclude it here and expect the
  # user to provide it through environmentFile (see services.honcho.environmentFile).
  baseEnv = {
    CACHE_URL = cacheURL;
    CACHE_ENABLED = if cfg.cache.enable then "true" else "false";
    HONCHO_CONFIG = "/etc/honcho/config.toml";
  }
  // (if cfg.database.password == "" then { DB_CONNECTION_URI = dbConnectionURI; } else { })
  // cfg.environment;
in
{
  options.services.honcho = {
    enable = mkEnableOption "Honcho — AI agent memory platform";

    package = mkOption {
      type = types.package;
      default = pkgs.honcho;
      defaultText = literalExpression "pkgs.honcho";
      description = "The Honcho package to use.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      example = {
        OPENAI_API_KEY = "sk-...";
        ANTHROPIC_API_KEY = "sk-ant-...";
        LANGFUSE_PUBLIC_KEY = "pk-...";
        LANGFUSE_SECRET_KEY = "sk-...";
      };
      description = ''
        Extra environment variables passed to both honcho services.

        WARNING: these values are rendered into the world-readable Nix store
        via the systemd unit derivation. Do NOT put secrets here.  Use
        services.honcho.environmentFile for API keys, database passwords,
        JWT secrets, and other sensitive values.
      '';
    };

    settings = mkOption {
      type = tomlFormat.type;
      default = { };
      example = {
        app = {
          LOG_LEVEL = "DEBUG";
          NAMESPACE = "honcho-prod";
        };
        auth = {
          USE_AUTH = true;
          JWT_SECRET = "change-me";
        };
        db = {
          POOL_SIZE = 20;
          MAX_OVERFLOW = 40;
        };
      };
      description = ''
        Honcho configuration settings, serialised as config.toml.

        WARNING: the generated file is a world-readable Nix store path. Do
        NOT put secrets here — JWTs, API keys, and passwords are visible to
        every user on the system.  Use services.honcho.environmentFile for
        sensitive values instead.

        See config.toml.example in the source tree for all available options.
      '';
    };

    database = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Automatically configure a local PostgreSQL database with pgvector.";
      };

      host = mkOption {
        type = types.str;
        default = "/run/postgresql";
        description = "PostgreSQL host (Unix socket when starting with /).";
      };

      port = mkOption {
        type = types.port;
        default = 5432;
        description = "PostgreSQL port.";
      };

      name = mkOption {
        type = types.str;
        default = "honcho";
        description = "Database name.";
      };

      user = mkOption {
        type = types.str;
        default = "honcho";
        description = "Database user.";
      };

      password = mkOption {
        type = types.str;
        default = "";
        description = ''
          Database password. Empty means peer auth via Unix socket.
          When set, the connection URL uses password auth over TCP.
        '';
      };

      poolSize = mkOption {
        type = types.int;
        default = 10;
        description = "SQLAlchemy pool_size.";
      };

      maxOverflow = mkOption {
        type = types.int;
        default = 20;
        description = "SQLAlchemy max_overflow.";
      };
    };

    cache = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Automatically configure a local Redis instance for caching.";
      };

      host = mkOption {
        type = types.str;
        default = "127.0.0.1";
        description = "Redis host.";
      };

      port = mkOption {
        type = types.port;
        default = 6379;
        description = "Redis port.";
      };

      db = mkOption {
        type = types.int;
        default = 0;
        description = "Redis database number.";
      };
    };

    api = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Run the Honcho API server.";
      };

      host = mkOption {
        type = types.str;
        default = "127.0.0.1";
        description = "Address to bind the API server to.";
      };

      port = mkOption {
        type = types.port;
        default = 8000;
        description = "Port to bind the API server to.";
      };
    };

    deriver = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Run the Honcho deriver worker (background queue consumer).";
      };

      workers = mkOption {
        type = types.int;
        default = 1;
        description = "Number of deriver worker processes (DERIVER_WORKERS).";
      };
    };

    environmentFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      example = "/run/secrets/honcho.env";
      description = ''
        Path to a runtime environment file loaded by both systemd services via
        EnvironmentFile.  Use this to provide secrets (database passwords, API
        keys, JWT secrets) without embedding them in the world-readable Nix
        store systemd unit derivations.

        The file contains KEY=VALUE lines, one per secret:
          DB_CONNECTION_URI=postgresql+psycopg://user:pass@host:5432/honcho
          OPENAI_API_KEY=sk-...
          JWT_SECRET=...

        This path is typically managed by sops-nix, agenix, or similar secret
        deployment tools and should have restricted permissions (mode 0600).

        When database.password is non-empty, DB_CONNECTION_URI must be
        provided through this file — it is intentionally excluded from the
        systemd unit environment to avoid leaking the password into the Nix
        store.
      '';
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.api.enable || cfg.deriver.enable;
        message = "Honcho is enabled but neither api.enable nor deriver.enable is set.";
      }
      {
        assertion = !cfg.api.enable || cfg.database.enable;
        message = "Honcho API requires a database. Set services.honcho.database.enable = true or disable the API.";
      }
      {
        assertion = !cfg.deriver.enable || cfg.database.enable;
        message = "Honcho deriver requires a database. Set services.honcho.database.enable = true or disable the deriver.";
      }
      {
        assertion = cfg.database.password == "" || cfg.environmentFile != null;
        message = ''
          Honcho has database.password set but no environmentFile.
          DB_CONNECTION_URI would leak the password into the world-readable Nix store.
          Either:
          - Set services.honcho.environmentFile to a runtime secrets file
            containing DB_CONNECTION_URI=postgresql+psycopg://..., or
          - Leave database.password empty to use peer auth via Unix socket.
        '';
      }
    ];

    # ---------- infrastructure ----------

    services.postgresql = mkIf cfg.database.enable {
      enable = true;
      package = pkgs.postgresql_16.withPackages (ps: [ ps.pgvector ]);
      ensureDatabases = [ cfg.database.name ];
      ensureUsers = [
        ({ name = cfg.database.user; ensureDBOwnership = true; }
          // (if cfg.database.password != "" then {
            ensureClauses = {
              login = true;
              password = cfg.database.password;
            };
          } else { })
        )
      ];
      initialScript = pkgs.writeText "honcho-init-vector.sql" ''
        CREATE EXTENSION IF NOT EXISTS vector;
      '';
      settings = {
        max_connections = cfg.database.poolSize + cfg.database.maxOverflow + 20;
      };
      authentication = pkgs.lib.mkIf (cfg.database.password != "") (
        pkgs.lib.mkForce ''
          #allov all all peer
          local all all peer
          host  all all 127.0.0.1/32 scram-sha-256
          host  all all ::1/128      scram-sha-256
        ''
      );
    };

    services.redis.servers."" = {
      enable = cfg.cache.enable;
      bind = cfg.cache.host;
      port = cfg.cache.port;
      settings.save = "";
    };

    # ---------- DB provisioning (shared oneshot) ----------

    systemd.services.honcho-db-provision = mkIf cfg.database.enable {
      description = "Honcho Database Provisioning";
      after = [ "postgresql.service" ];
      requires = [ "postgresql.service" ];
      before = [ "honcho-api.service" "honcho-deriver.service" ];

      serviceConfig = {
        Type = "oneshot";
        User = "honcho";
        Group = "honcho";
        StateDirectory = "honcho";
        WorkingDirectory = "${cfg.package}";
        ExecStart = "${cfg.package}/bin/python ${cfg.package}/scripts/provision_db.py";
        EnvironmentFile = mkIf (cfg.environmentFile != null) cfg.environmentFile;
      };

      environment = baseEnv;
    };

    # ---------- configuration file ----------

    environment.etc."honcho/config.toml".source = configFile;

    # ---------- systemd services ----------

    systemd.services.honcho-api = mkIf cfg.api.enable {
      description = "Honcho API Server";
      after = [ "postgresql.service" "network.target" "honcho-db-provision.service" ] ++ optionals cfg.cache.enable [ "redis.service" ];
      requires = [ "postgresql.service" "honcho-db-provision.service" ] ++ optionals cfg.cache.enable [ "redis.service" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        User = "honcho";
        Group = "honcho";
        StateDirectory = "honcho";
        WorkingDirectory = "/var/lib/honcho";
        ExecStart = "${cfg.package}/bin/python -m uvicorn src.main:app --host ${cfg.api.host} --port ${toString cfg.api.port}";
        EnvironmentFile = mkIf (cfg.environmentFile != null) cfg.environmentFile;
        Restart = "on-failure";
        RestartSec = "5s";
        TimeoutStartSec = "60s";
      };

      environment = baseEnv;

      unitConfig.StartLimitIntervalSec = "30s";
    };

    systemd.services.honcho-deriver = mkIf cfg.deriver.enable {
      description = "Honcho Deriver Worker (background queue consumer)";
      after = [ "postgresql.service" "honcho-db-provision.service" ] ++ optionals cfg.cache.enable [ "redis.service" ];
      requires = [ "postgresql.service" "honcho-db-provision.service" ] ++ optionals cfg.cache.enable [ "redis.service" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        User = "honcho";
        Group = "honcho";
        StateDirectory = "honcho";
        WorkingDirectory = "/var/lib/honcho";
        ExecStart = "${cfg.package}/bin/python -m src.deriver";
        EnvironmentFile = mkIf (cfg.environmentFile != null) cfg.environmentFile;
        Restart = "on-failure";
        RestartSec = "5s";
        TimeoutStartSec = "60s";
      };

      environment = baseEnv // {
        DERIVER_WORKERS = toString cfg.deriver.workers;
      };

      unitConfig.StartLimitIntervalSec = "30s";
    };

    # ---------- users ----------

    users.users.honcho = {
      isSystemUser = true;
      group = "honcho";
      description = "Honcho daemon user";
    };

    users.groups.honcho = { };
  };
}
