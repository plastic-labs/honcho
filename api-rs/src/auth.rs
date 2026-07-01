use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chrono::{DateTime, NaiveDateTime, SecondsFormat, Utc};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::Sha256;
use thiserror::Error;

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthConfig {
    pub use_auth: bool,
    pub jwt_secret: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct JwtParams {
    #[serde(default, rename = "t")]
    pub timestamp: String,
    #[serde(default)]
    pub exp: Option<String>,
    #[serde(default, rename = "ad")]
    pub admin: Option<bool>,
    #[serde(default, rename = "w")]
    pub workspace: Option<String>,
    #[serde(default, rename = "p")]
    pub peer: Option<String>,
    #[serde(default, rename = "s")]
    pub session: Option<String>,
}

impl JwtParams {
    pub fn admin() -> Self {
        Self {
            timestamp: String::new(),
            exp: None,
            admin: Some(true),
            workspace: None,
            peer: None,
            session: None,
        }
    }

    fn is_admin(&self) -> bool {
        self.admin.unwrap_or(false)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AuthError {
    #[error("No access token provided")]
    MissingToken,
    #[error("Invalid JWT")]
    InvalidJwt,
    #[error("Invalid JWT scope: peer/session token missing workspace")]
    InvalidScope,
    #[error("JWT expired")]
    Expired,
    #[error("Resource requires admin privileges")]
    AdminRequired,
    #[error("JWT not permissioned for this resource")]
    PermissionDenied,
}

/// Whether a peer- or session-scoped claim lacks its parent workspace.
///
/// A peer or session scope is meaningless without a workspace: the route-level
/// check cannot rule out cross-workspace use (a `{p: "alice"}` token would match
/// `alice` in any workspace). Truthiness-based — callers pass already-normalized
/// `Option`s so empty-string claims count as absent. Shared by token
/// verification (the token-shape invariant) and the keys API (the creation-time
/// guard) so the two rules cannot drift apart.
pub fn scope_requires_workspace(
    peer: Option<&str>,
    session: Option<&str>,
    workspace: Option<&str>,
) -> bool {
    (peer.is_some() || session.is_some()) && workspace.is_none()
}

/// The outcome of a pure scope check (no token parsing, no I/O).
///
/// `NeedsMembership` defers a peer-scoped-key session-membership lookup to the
/// caller, which has the DB pool — `authorize_scope` itself stays synchronous.
#[derive(Debug, PartialEq, Eq)]
pub enum ScopeOutcome {
    Grant,
    Deny(AuthError),
    NeedsMembership {
        peer: String,
        session: String,
        workspace: String,
    },
}

/// Parse the `Authorization` header and verify the token, returning the decoded
/// claims. In no-auth mode every request is admin. This is the shared preamble
/// for both the synchronous [`authorize`] and the async member-read path in the
/// router — the scope decision ([`authorize_scope`]) runs on the result.
pub fn verify_request(
    config: &AuthConfig,
    authorization_header: Option<&str>,
) -> Result<JwtParams, AuthError> {
    if !config.use_auth {
        return Ok(JwtParams::admin());
    }

    let header = authorization_header
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(AuthError::MissingToken)?;
    // Mirror FastAPI's HTTPBearer(auto_error=False): split on the first
    // whitespace, match the scheme case-insensitively, and treat the remainder
    // as the credentials. A non-"bearer" scheme or empty credentials behaves
    // exactly like "no token provided" (HTTPBearer returns None there).
    let token = header
        .split_once(char::is_whitespace)
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, rest)| rest.trim())
        .filter(|rest| !rest.is_empty())
        .ok_or(AuthError::MissingToken)?;
    verify_hs256_token(
        token,
        config.jwt_secret.as_deref().ok_or(AuthError::InvalidJwt)?,
    )
}

/// Authorize verified claims against a route's declared scope, by the token's
/// *narrowest* scope (port of `src/security.py::auth`, #679).
///
/// A narrower-than-workspace token must NOT fall back to workspace access:
/// `{w: ws, p: alice}` may only act on `alice`, never on a sibling peer. When
/// `allow_member_read` is set and a peer-scoped token hits a session route it is
/// not directly scoped to, the decision is deferred via `NeedsMembership` so the
/// caller can check `is_peer_in_session`.
pub fn authorize_scope(
    params: &JwtParams,
    admin_required: bool,
    workspace_name: Option<&str>,
    peer_name: Option<&str>,
    session_name: Option<&str>,
    allow_member_read: bool,
) -> ScopeOutcome {
    if params.is_admin() {
        return ScopeOutcome::Grant;
    }
    if admin_required {
        return ScopeOutcome::Deny(AuthError::AdminRequired);
    }

    // Self-authorizing routes declare no scope: decode the token here and let
    // the handler compare the claims against body/path data itself (needed for
    // routes whose resource identifier is not available to the route guard).
    if workspace_name.is_none() && peer_name.is_none() && session_name.is_none() {
        return ScopeOutcome::Grant;
    }

    // Every scoped, non-admin path requires the token's workspace to match the
    // route's. Check it once here so no branch below can forget it and silently
    // re-open cross-workspace access (the bug this rewrite fixes).
    if let Some(workspace_name) = workspace_name {
        if params.workspace.as_deref() != Some(workspace_name) {
            return ScopeOutcome::Deny(AuthError::PermissionDenied);
        }
    }

    if params.session.is_some() {
        // Session-scoped token: confined to its own session. No cross-scope
        // access to peer routes.
        if session_name.is_none() || params.session.as_deref() != session_name {
            return ScopeOutcome::Deny(AuthError::PermissionDenied);
        }
        return ScopeOutcome::Grant;
    }

    if let Some(token_peer) = params.peer.as_deref() {
        // Peer-scoped token: its own peer routes...
        if peer_name == Some(token_peer) {
            return ScopeOutcome::Grant;
        }
        // ...plus read-only access to the sessions the peer is a member of.
        // Gated on `allow_member_read` so only read routes opt in; writes stay
        // denied. Requires the route's workspace (already matched above) so the
        // membership lookup is scoped.
        if allow_member_read {
            if let (Some(session_name), Some(workspace_name)) = (session_name, workspace_name) {
                return ScopeOutcome::NeedsMembership {
                    peer: token_peer.to_string(),
                    session: session_name.to_string(),
                    workspace: workspace_name.to_string(),
                };
            }
        }
        return ScopeOutcome::Deny(AuthError::PermissionDenied);
    }

    if params.workspace.is_some() {
        // Workspace tokens reach any route inside their workspace (the workspace
        // match was verified above).
        return ScopeOutcome::Grant;
    }

    ScopeOutcome::Deny(AuthError::PermissionDenied)
}

/// Verify the request and authorize it against a route scope. Used by every
/// route that is not a session member-read route (those defer to
/// [`authorize_scope`] with `allow_member_read = true` plus a DB membership
/// check). `allow_member_read` is `false` here, so `NeedsMembership` never
/// occurs; it is treated as a denial to fail closed.
pub fn authorize(
    config: &AuthConfig,
    authorization_header: Option<&str>,
    admin_required: bool,
    workspace_name: Option<&str>,
    peer_name: Option<&str>,
    session_name: Option<&str>,
) -> Result<JwtParams, AuthError> {
    let params = verify_request(config, authorization_header)?;
    match authorize_scope(
        &params,
        admin_required,
        workspace_name,
        peer_name,
        session_name,
        false,
    ) {
        ScopeOutcome::Grant => Ok(params),
        ScopeOutcome::Deny(error) => Err(error),
        ScopeOutcome::NeedsMembership { .. } => Err(AuthError::PermissionDenied),
    }
}

fn verify_hs256_token(token: &str, secret: &str) -> Result<JwtParams, AuthError> {
    let parts = token.split('.').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(AuthError::InvalidJwt);
    }

    let actual = URL_SAFE_NO_PAD
        .decode(parts[2])
        .map_err(|_| AuthError::InvalidJwt)?;
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts keys of any size");
    mac.update(format!("{}.{}", parts[0], parts[1]).as_bytes());
    mac.verify_slice(&actual)
        .map_err(|_| AuthError::InvalidJwt)?;

    let payload = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|_| AuthError::InvalidJwt)?;
    let mut params =
        serde_json::from_slice::<JwtParams>(&payload).map_err(|_| AuthError::InvalidJwt)?;
    // Normalize empty-string scope claims to None so a blank `w`/`p`/`s` cannot
    // masquerade as a present claim in the checks below.
    params.workspace = params.workspace.filter(|value| !value.is_empty());
    params.peer = params.peer.filter(|value| !value.is_empty());
    params.session = params.session.filter(|value| !value.is_empty());
    // Token-shape invariant: a peer- or session-scoped token MUST also carry its
    // parent workspace, otherwise the route-level check cannot rule out
    // cross-workspace use.
    if scope_requires_workspace(
        params.peer.as_deref(),
        params.session.as_deref(),
        params.workspace.as_deref(),
    ) {
        return Err(AuthError::InvalidScope);
    }
    if let Some(exp) = params.exp.as_deref() {
        let exp_time = parse_iso_datetime_lenient(exp)?;
        if exp_time < Utc::now() {
            return Err(AuthError::Expired);
        }
    }

    Ok(params)
}

/// Parse an ISO 8601 timestamp the way Python's
/// `src/utils/formatting.py:parse_datetime_iso` does: reject null bytes /
/// line breaks / non-printable characters, accept a `Z`/`z` suffix and explicit
/// offsets, and fall back to treating a naive (timezone-less) timestamp as UTC.
/// This keeps `exp` validation byte-for-byte compatible with the Python issuer,
/// which stores `exp` as an ISO string rather than a numeric epoch.
fn parse_iso_datetime_lenient(value: &str) -> Result<DateTime<Utc>, AuthError> {
    if value.contains('\0') || value.contains('\r') || value.contains('\n') {
        return Err(AuthError::InvalidJwt);
    }
    if value.chars().any(|c| (c as u32) < 32 && c != '\t') {
        return Err(AuthError::InvalidJwt);
    }
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(AuthError::InvalidJwt);
    }
    let normalized = match trimmed.strip_suffix(['Z', 'z']) {
        Some(prefix) => format!("{prefix}+00:00"),
        None => trimmed.to_string(),
    };
    if let Ok(parsed) = DateTime::parse_from_rfc3339(&normalized) {
        return Ok(parsed.with_timezone(&Utc));
    }
    // No timezone present: assume UTC, matching Python's `fromisoformat` +
    // "if result.tzinfo is None: replace(tzinfo=utc)" fallback.
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(DateTime::from_naive_utc_and_offset(naive, Utc));
        }
    }
    Err(AuthError::InvalidJwt)
}

fn sign_hs256(input: &[u8], secret: &str) -> Vec<u8> {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts keys of any size");
    mac.update(input);
    mac.finalize().into_bytes().to_vec()
}

pub fn create_scoped_key(mut params: JwtParams, secret: &str) -> Result<String, AuthError> {
    if params.timestamp.is_empty() {
        params.timestamp = format_datetime_utc(Utc::now());
    }
    if let Some(exp) = params.exp.as_deref() {
        let exp_time = parse_iso_datetime_lenient(exp)?;
        params.exp = Some(format_datetime_utc(exp_time));
    }
    let mut payload = serde_json::Map::new();
    payload.insert("t".to_string(), Value::String(params.timestamp));
    if let Some(exp) = params.exp {
        payload.insert("exp".to_string(), Value::String(exp));
    }
    if let Some(workspace) = params.workspace {
        payload.insert("w".to_string(), Value::String(workspace));
    }
    if let Some(peer) = params.peer {
        payload.insert("p".to_string(), Value::String(peer));
    }
    if let Some(session) = params.session {
        payload.insert("s".to_string(), Value::String(session));
    }
    Ok(create_hs256_token(&Value::Object(payload), secret))
}

fn format_datetime_utc(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn create_hs256_token(payload: &Value, secret: &str) -> String {
    let header = json!({"alg": "HS256", "typ": "JWT"});
    let header = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
    let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(payload).unwrap());
    let signing_input = format!("{header}.{payload}");
    let signature = URL_SAFE_NO_PAD.encode(sign_hs256(signing_input.as_bytes(), secret));
    format!("{signing_input}.{signature}")
}

pub fn create_hs256_token_for_test(payload: &Value, secret: &str) -> String {
    create_hs256_token(payload, secret)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AuthConfig {
        AuthConfig {
            use_auth: true,
            jwt_secret: Some("secret".to_string()),
        }
    }

    fn workspace_token(extra: Value) -> String {
        let mut payload = serde_json::Map::new();
        payload.insert("t".to_string(), Value::String(String::new()));
        payload.insert("w".to_string(), Value::String("ws".to_string()));
        if let Value::Object(map) = extra {
            payload.extend(map);
        }
        create_hs256_token(&Value::Object(payload), "secret")
    }

    fn authorize_ws(header: &str) -> Result<JwtParams, AuthError> {
        authorize(&config(), Some(header), false, Some("ws"), None, None)
    }

    #[test]
    fn bearer_scheme_is_case_insensitive() {
        let token = workspace_token(json!({}));
        for scheme in ["Bearer", "bearer", "BEARER", "BeArEr"] {
            let header = format!("{scheme} {token}");
            let params = authorize_ws(&header).expect("scheme must be accepted case-insensitively");
            assert_eq!(params.workspace.as_deref(), Some("ws"));
        }
    }

    #[test]
    fn non_bearer_scheme_is_missing_token() {
        let header = format!("Basic {}", workspace_token(json!({})));
        assert_eq!(authorize_ws(&header), Err(AuthError::MissingToken));
    }

    #[test]
    fn empty_or_schemeless_credentials_are_missing_token() {
        assert_eq!(authorize_ws("Bearer   "), Err(AuthError::MissingToken));
        assert_eq!(authorize_ws("Bearer"), Err(AuthError::MissingToken));
    }

    #[test]
    fn exp_with_z_suffix_is_enforced() {
        let valid = format!("Bearer {}", workspace_token(json!({"exp": "2999-01-01T00:00:00Z"})));
        assert!(authorize_ws(&valid).is_ok());
        let expired = format!("Bearer {}", workspace_token(json!({"exp": "2000-01-01T00:00:00Z"})));
        assert_eq!(authorize_ws(&expired), Err(AuthError::Expired));
    }

    #[test]
    fn exp_naive_timestamp_is_treated_as_utc() {
        let valid = format!("Bearer {}", workspace_token(json!({"exp": "2999-01-01T00:00:00"})));
        assert!(authorize_ws(&valid).is_ok());
        let expired = format!("Bearer {}", workspace_token(json!({"exp": "2000-01-01 00:00:00"})));
        assert_eq!(authorize_ws(&expired), Err(AuthError::Expired));
    }

    #[test]
    fn no_auth_mode_yields_admin() {
        let config = AuthConfig {
            use_auth: false,
            jwt_secret: None,
        };
        let params =
            authorize(&config, None, true, None, None, None).expect("no-auth must yield admin");
        assert!(params.is_admin());
    }

    fn params(workspace: Option<&str>, peer: Option<&str>, session: Option<&str>) -> JwtParams {
        JwtParams {
            workspace: workspace.map(str::to_string),
            peer: peer.map(str::to_string),
            session: session.map(str::to_string),
            ..JwtParams::default()
        }
    }

    #[test]
    fn peer_scoped_token_denied_on_sibling_peer_route() {
        // The core of #679: `{w: ws, p: alice}` must NOT act on peer `bob`. The
        // old logic fell through to the workspace match and granted it.
        let alice = params(Some("ws"), Some("alice"), None);
        assert_eq!(
            authorize_scope(&alice, false, Some("ws"), Some("bob"), None, false),
            ScopeOutcome::Deny(AuthError::PermissionDenied)
        );
        // ...but it may act on its own peer route.
        assert_eq!(
            authorize_scope(&alice, false, Some("ws"), Some("alice"), None, false),
            ScopeOutcome::Grant
        );
    }

    #[test]
    fn session_scoped_token_confined_to_its_session() {
        let s1 = params(Some("ws"), None, Some("s1"));
        assert_eq!(
            authorize_scope(&s1, false, Some("ws"), None, Some("s1"), false),
            ScopeOutcome::Grant
        );
        // Wrong session denied.
        assert_eq!(
            authorize_scope(&s1, false, Some("ws"), None, Some("s2"), false),
            ScopeOutcome::Deny(AuthError::PermissionDenied)
        );
        // No cross-scope to peer routes.
        assert_eq!(
            authorize_scope(&s1, false, Some("ws"), Some("alice"), None, false),
            ScopeOutcome::Deny(AuthError::PermissionDenied)
        );
    }

    #[test]
    fn workspace_token_spans_workspace_but_not_across_workspaces() {
        let ws = params(Some("ws"), None, None);
        assert_eq!(
            authorize_scope(&ws, false, Some("ws"), Some("bob"), None, false),
            ScopeOutcome::Grant
        );
        assert_eq!(
            authorize_scope(&ws, false, Some("other"), None, None, false),
            ScopeOutcome::Deny(AuthError::PermissionDenied)
        );
    }

    #[test]
    fn peer_member_read_defers_to_membership_only_when_opted_in() {
        let alice = params(Some("ws"), Some("alice"), None);
        // Read route opts into member-read: defer to is_peer_in_session.
        assert_eq!(
            authorize_scope(&alice, false, Some("ws"), None, Some("s1"), true),
            ScopeOutcome::NeedsMembership {
                peer: "alice".to_string(),
                session: "s1".to_string(),
                workspace: "ws".to_string(),
            }
        );
        // Same route without member-read (a write route) denies outright.
        assert_eq!(
            authorize_scope(&alice, false, Some("ws"), None, Some("s1"), false),
            ScopeOutcome::Deny(AuthError::PermissionDenied)
        );
    }

    #[test]
    fn admin_grants_anywhere_and_admin_required_denies_non_admin() {
        assert_eq!(
            authorize_scope(&JwtParams::admin(), false, Some("ws"), Some("bob"), None, false),
            ScopeOutcome::Grant
        );
        assert_eq!(
            authorize_scope(&params(Some("ws"), None, None), true, None, None, None, false),
            ScopeOutcome::Deny(AuthError::AdminRequired)
        );
    }

    #[test]
    fn self_authorizing_route_grants_decode_only() {
        // No scope declared: the handler compares claims itself.
        assert_eq!(
            authorize_scope(&params(Some("ws"), Some("alice"), None), false, None, None, None, false),
            ScopeOutcome::Grant
        );
    }

    #[test]
    fn verify_rejects_scoped_token_missing_workspace() {
        let bearer = |claims: Value| format!("Bearer {}", create_hs256_token(&claims, "secret"));
        // Peer token with no workspace.
        assert_eq!(
            authorize(
                &config(),
                Some(&bearer(json!({"t": "", "p": "alice"}))),
                false,
                Some("ws"),
                Some("alice"),
                None,
            ),
            Err(AuthError::InvalidScope)
        );
        // Session token with no workspace.
        assert_eq!(
            authorize(
                &config(),
                Some(&bearer(json!({"t": "", "s": "s1"}))),
                false,
                Some("ws"),
                None,
                Some("s1"),
            ),
            Err(AuthError::InvalidScope)
        );
        // Empty-string workspace is normalized to absent → still rejected.
        assert_eq!(
            authorize(
                &config(),
                Some(&bearer(json!({"t": "", "w": "", "p": "alice"}))),
                false,
                Some("ws"),
                Some("alice"),
                None,
            ),
            Err(AuthError::InvalidScope)
        );
    }

    #[test]
    fn scope_requires_workspace_predicate() {
        assert!(scope_requires_workspace(Some("alice"), None, None));
        assert!(scope_requires_workspace(None, Some("s1"), None));
        assert!(!scope_requires_workspace(Some("alice"), None, Some("ws")));
        assert!(!scope_requires_workspace(None, None, None));
    }
}
