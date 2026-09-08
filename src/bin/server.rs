//! AletheiaDB HTTP Server Binary
//!
//! This binary launches the AletheiaDB HTTP server, providing a REST API
//! for interacting with the database.
//!
//! # Usage
//!
//! ```bash
//! # Run with default settings (port 1963, restrictive CORS)
//! cargo run --bin aletheia-server --features http-server
//!
//! # Run with custom port
//! ALETHEIADB_PORT=3000 cargo run --bin aletheia-server --features http-server
//!
//! # Run with permissive CORS (development only!)
//! ALETHEIADB_CORS_PERMISSIVE=true cargo run --bin aletheia-server --features http-server
//!
//! # Health check (auth is required by default: pass a valid API key)
//! curl -H "x-api-key: $ALETHEIADB_BOOTSTRAP_ADMIN_KEY" http://localhost:1963/status
//! ```
//!
//! # Environment Variables
//!
//! - `ALETHEIADB_PORT`: Port to listen on (default: 1963)
//! - `ALETHEIADB_HOST`: Host to bind to (default: 0.0.0.0)
//! - `ALETHEIADB_CORS_PERMISSIVE`: Set to "true" to allow any origin (default: false)
//! - `ALETHEIADB_CORS_ORIGINS`: Comma-separated list of allowed origins
//! - `ALETHEIADB_CONFIG`: Path to a TOML config file consumed by
//!   `AletheiaDBConfig::from_toml_file`. Takes precedence over
//!   `ALETHEIADB_DATA_DIR`.
//! - `ALETHEIADB_DATA_DIR`: Directory for WAL + index persistence. When set,
//!   the server persists state across restarts. When unset (and `ALETHEIADB_CONFIG`
//!   is also unset), runs in-memory (ephemeral — useful for demos, useless
//!   for production).
//! - `ALETHEIADB_AUTH_MODE`: `required` (default) or `anonymous`. In
//!   `required` mode every request must present a valid API key
//!   (`Authorization: Bearer <key>` or `x-api-key`), and the server refuses
//!   to start with zero credentials. `anonymous` disables authentication
//!   entirely (explicit opt-in; a prominent warning is logged). Invalid
//!   values fall back to `required` (fail-closed) with a warning.
//! - `ALETHEIADB_BOOTSTRAP_ADMIN_KEY`: Plaintext admin key installed at
//!   startup as principal `bootstrap-admin` (role `admin`). Memory-only —
//!   re-supply it on every start, or use it once to mint persisted keys via
//!   `POST /admin/keys` (persisted under `{data_dir}/auth/keys.json` when a
//!   data dir is configured).
//!
//! # Endpoints
//!
//! All AletheiaDB endpoints require a credential in `required` mode
//! (`Authorization: Bearer <key>` or `x-api-key`); see
//! `docs/guides/access-control-matrix.md` for role classes.
//!
//! - `GET /status` - Health check endpoint (metrics class), returns
//!   `{"status": "healthy"}`
//! - `POST /query` - JSON query API (`find_node`, `create_node`, bulk ops,
//!   `execute_query`, ...); read or write class per operation
//! - `POST /admin/keys` - Mint an API key (admin class)
//! - `GET /admin/keys` - List principals, masked (admin class)
//! - `POST /admin/keys/revoke` - Revoke a key by principal id (admin class)
//!
//! The autumn-web framework additionally serves unauthenticated
//! health/metadata routes (`/health`, `/live`, `/ready`, `/startup`,
//! `/actuator/health|info|metrics|a11y|ui`); its sensitive actuator group is
//! force-disabled in every profile. See
//! `docs/guides/security-quickstart.md`.
//!
//! # Security Notes
//!
//! - By default, CORS is restrictive (only localhost:3000 allowed)
//! - Set `ALETHEIADB_CORS_ORIGINS` to configure allowed origins for production
//! - Only use `ALETHEIADB_CORS_PERMISSIVE=true` in development environments
//!
//! # Graceful Shutdown
//!
//! The server handles SIGTERM and SIGINT signals for graceful shutdown,
//! allowing in-flight requests to complete before terminating.

use aletheiadb::auth::{SecretString, auth_mode_from_env};
use aletheiadb::http::{CorsConfig, ServerConfig, run_server};
use std::env;
use std::path::PathBuf;

/// Parse port from environment variable with warning on invalid values.
fn parse_port() -> u16 {
    match env::var("ALETHEIADB_PORT") {
        Ok(port_str) => match port_str.parse::<u16>() {
            Ok(port) => port,
            Err(e) => {
                eprintln!(
                    "WARNING: Invalid ALETHEIADB_PORT '{}': {}. Using default port 1963.",
                    port_str, e
                );
                1963
            }
        },
        Err(_) => 1963,
    }
}

/// Parse host from environment variable.
fn parse_host() -> String {
    env::var("ALETHEIADB_HOST").unwrap_or_else(|_| "0.0.0.0".to_string())
}

/// Parse CORS configuration from environment variables.
fn parse_cors_config() -> CorsConfig {
    // Check if permissive mode is enabled
    let is_permissive = env::var("ALETHEIADB_CORS_PERMISSIVE")
        .map(|v| v.to_lowercase() == "true" || v == "1")
        .unwrap_or(false);

    if is_permissive {
        return CorsConfig::permissive();
    }

    // Parse allowed origins from comma-separated list
    match env::var("ALETHEIADB_CORS_ORIGINS") {
        Ok(origins_str) => {
            let mut config = CorsConfig::restrictive();
            for origin in origins_str
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                config = config.allow_origin(origin);
            }
            config
        }
        Err(_) => CorsConfig::restrictive(),
    }
}

/// Parse the data directory env var into an optional `PathBuf`.
///
/// Unset or empty → `None` (in-memory mode). Any non-empty value is taken
/// as a path literal; `AletheiaDB::with_unified_config` creates the directory
/// structure on startup. Delegates to the shared helper so every exposed
/// binary reads the same env var with the same semantics.
fn parse_data_dir() -> Option<PathBuf> {
    aletheiadb::config::data_dir_from_env()
}

/// Parse `ALETHEIADB_BOOTSTRAP_ADMIN_KEY` into a redacted secret, if set.
///
/// The value is trimmed: the HTTP extractor trims presented tokens, so an
/// env value with stray whitespace (e.g. from `echo` without `-n` piped into
/// the environment) could otherwise never verify.
fn parse_bootstrap_admin_key() -> Option<SecretString> {
    match env::var("ALETHEIADB_BOOTSTRAP_ADMIN_KEY") {
        Ok(raw) if !raw.trim().is_empty() => Some(SecretString::new(raw.trim())),
        _ => None,
    }
}

#[autumn_web::main]
async fn main() {
    let port = parse_port();
    let host = parse_host();
    let cors = parse_cors_config();
    let data_dir = parse_data_dir();

    // The legacy HTTP server owns storage — refuse a data directory a live
    // daemon already owns (Issue #2905), rather than becoming a second writer.
    if let Some(dir) = data_dir.as_deref()
        && let Some(pid) = aletheiadb::daemon_lock::live_owner(dir)
    {
        eprintln!(
            "aletheia-server refusing to start: {}",
            aletheiadb::daemon_lock::already_owned_message(dir, pid)
        );
        std::process::exit(1);
    }

    let mut builder = ServerConfig::builder()
        .port(port)
        .host(host)
        .cors(cors)
        .auth_mode(auth_mode_from_env());
    if let Some(path) = data_dir {
        builder = builder.data_dir(path);
    }
    if let Some(key) = parse_bootstrap_admin_key() {
        builder = builder.bootstrap_admin_key(key);
    }
    let config = builder.build();

    if let Err(e) = run_server(config).await {
        eprintln!("aletheia-server failed: {e}");
        std::process::exit(1);
    }
}
