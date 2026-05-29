{ config, lib, pkgs, ... }:

let
  cfg = config.services.honcho;
  inherit (lib)
    mkEnableOption
    mkOption
    mkIf
    types
    literalExpression
    ;

  tomlFormat = pkgs.formats.toml { };

  # Build the connection URI from components
  dbConnectionURI =
    if cfg.database.password != "" then
      "postgresql+psycopg://${cfg.database.user}:${cfg.database.password}@${cfg.database.host}:${toString cfg.database.port}/${cfg.database.name}"
    else
      "postgresql+psycopg:///${cfg.database.name}?host=${cfg.database.host}";

  cacheURL = "redis://${cfg.cache.host}:${toString cfg.cache.port}/${toString cfg.cache.db}?suppress=true";

  # Write out the merged config.toml
  configFile = tomlFormat.generate "honcho-config.toml" cfg.settings;

  # Environment shared across all honcho services
  baseEnv = {
    DB_CONNECTION_URI = dbConnectionURI;
    CACHE_URL = cacheURL;
    CACHE_ENABLED = if cfg.cache.enable then "true" else "false";
    HONCHO_CONFIG = "/etc/honcho/config.toml";
  } // cfg.environment;
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
        Use this to set LLM provider API keys and other secrets.
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
    ];

    # ---------- infrastructure ----------

    services.postgresql = mkIf cfg.database.enable {
      enable = true;
      package = pkgs.postgresql_16.withPackages (ps: [ ps.pgvector ]);
      ensureDatabases = [ cfg.database.name ];
      ensureUsers = [
        {
          name = cfg.database.user;
          ensureDBOwnership = true;
        }
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

    services.redis = mkIf cfg.cache.enable {
      enable = true;
      bind = cfg.cache.host;
      port = cfg.cache.port;
      settings.save = "";
    };

    # ---------- configuration file ----------

    environment.etc."honcho/config.toml".source = configFile;

    # ---------- systemd services ----------

    systemd.services.honcho-api = mkIf cfg.api.enable {
      description = "Honcho API Server";
      after = [ "postgresql.service" "redis.service" "network.target" ];
      requires = [ "postgresql.service" "redis.service" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        User = "honcho";
        Group = "honcho";
        StateDirectory = "honcho";
        WorkingDirectory = "/var/lib/honcho";
        ExecStartPre = "${cfg.package}/bin/python ${cfg.package}/scripts/provision_db.py ${cfg.database.name}";
        ExecStart = "${cfg.package}/bin/fastapi run --host ${cfg.api.host} --port ${toString cfg.api.port} src.main:app";
        Restart = "on-failure";
        RestartSec = "5s";
        TimeoutStartSec = "60s";
      };

      environment = baseEnv;

      unitConfig.StartLimitIntervalSec = "30s";
    };

    systemd.services.honcho-deriver = mkIf cfg.deriver.enable {
      description = "Honcho Deriver Worker (background queue consumer)";
      after = [ "postgresql.service" "redis.service" "honcho-api.service" ];
      requires = [ "postgresql.service" "redis.service" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        Type = "simple";
        User = "honcho";
        Group = "honcho";
        StateDirectory = "honcho";
        WorkingDirectory = "/var/lib/honcho";
        ExecStart = "${cfg.package}/bin/python -m src.deriver";
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
