use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use chrono::{DateTime, SecondsFormat, Utc};
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
    #[error("JWT expired")]
    Expired,
    #[error("Resource requires admin privileges")]
    AdminRequired,
    #[error("JWT not permissioned for this resource")]
    PermissionDenied,
}

pub fn authorize(
    config: &AuthConfig,
    authorization_header: Option<&str>,
    admin_required: bool,
    workspace_name: Option<&str>,
    peer_name: Option<&str>,
    session_name: Option<&str>,
) -> Result<JwtParams, AuthError> {
    if !config.use_auth {
        return Ok(JwtParams::admin());
    }

    let header = authorization_header
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(AuthError::MissingToken)?;
    let token = header
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(AuthError::InvalidJwt)?;
    let params = verify_hs256_token(
        token,
        config.jwt_secret.as_deref().ok_or(AuthError::InvalidJwt)?,
    )?;

    if params.is_admin() {
        return Ok(params);
    }
    if admin_required {
        return Err(AuthError::AdminRequired);
    }
    if let Some(session_name) = session_name {
        if params.session.as_deref() == Some(session_name) {
            if let Some(workspace_name) = workspace_name {
                if params.workspace.as_deref() != Some(workspace_name) {
                    return Err(AuthError::PermissionDenied);
                }
            }
            return Ok(params);
        }
    }
    if let Some(peer_name) = peer_name {
        if params.peer.as_deref() == Some(peer_name) {
            if let Some(workspace_name) = workspace_name {
                if params.workspace.as_deref() != Some(workspace_name) {
                    return Err(AuthError::PermissionDenied);
                }
            }
            return Ok(params);
        }
    }
    if let Some(workspace_name) = workspace_name {
        if params.workspace.as_deref() == Some(workspace_name) {
            return Ok(params);
        }
    }

    if workspace_name.is_some() || peer_name.is_some() || session_name.is_some() {
        return Err(AuthError::PermissionDenied);
    }

    Ok(params)
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
    let params =
        serde_json::from_slice::<JwtParams>(&payload).map_err(|_| AuthError::InvalidJwt)?;
    if let Some(exp) = params.exp.as_deref() {
        let exp_time = DateTime::parse_from_rfc3339(exp)
            .map_err(|_| AuthError::InvalidJwt)?
            .with_timezone(&Utc);
        if exp_time < Utc::now() {
            return Err(AuthError::Expired);
        }
    }

    Ok(params)
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
        let exp_time = DateTime::parse_from_rfc3339(exp)
            .map_err(|_| AuthError::InvalidJwt)?
            .with_timezone(&Utc);
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
