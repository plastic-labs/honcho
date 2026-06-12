use http::{HeaderMap, HeaderName};
use thiserror::Error;

pub const DEFAULT_ASSISTANT_NAME: &str = "Assistant";
pub const DEFAULT_WORKSPACE_ID: &str = "default";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HonchoConfig {
    pub api_key: String,
    pub authorization: String,
    pub user_name: String,
    pub assistant_name: String,
    pub workspace_id: String,
    pub base_url: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("Missing Authorization header")]
    MissingAuthorization,
    #[error("Authorization header must use Bearer authentication")]
    InvalidAuthorization,
    #[error("Missing X-Honcho-User-Name header")]
    MissingUserName,
    #[error("Invalid header value for {0}")]
    InvalidHeader(&'static str),
}

pub fn parse_config_from_headers(
    headers: &HeaderMap,
    base_url: &str,
) -> Result<HonchoConfig, ConfigError> {
    let authorization =
        header_value(headers, "authorization")?.ok_or(ConfigError::MissingAuthorization)?;
    parse_values(
        authorization,
        header_value(headers, "x-honcho-user-name")?,
        header_value(headers, "x-honcho-assistant-name")?,
        header_value(headers, "x-honcho-workspace-id")?,
        base_url,
    )
}

pub fn parse_config_from_pairs(
    pairs: &[(&str, &str)],
    base_url: &str,
) -> Result<HonchoConfig, ConfigError> {
    let get = |name: &str| {
        pairs
            .iter()
            .rev()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.trim().to_string())
            .filter(|value| !value.is_empty())
    };

    let authorization = get("authorization").ok_or(ConfigError::MissingAuthorization)?;
    parse_values(
        authorization,
        get("x-honcho-user-name"),
        get("x-honcho-assistant-name"),
        get("x-honcho-workspace-id"),
        base_url,
    )
}

fn parse_values(
    authorization: String,
    user_name: Option<String>,
    assistant_name: Option<String>,
    workspace_id: Option<String>,
    base_url: &str,
) -> Result<HonchoConfig, ConfigError> {
    let api_key = authorization
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or(ConfigError::InvalidAuthorization)?
        .to_string();
    let user_name = user_name.ok_or(ConfigError::MissingUserName)?;

    Ok(HonchoConfig {
        api_key,
        authorization,
        user_name,
        assistant_name: assistant_name.unwrap_or_else(|| DEFAULT_ASSISTANT_NAME.to_string()),
        workspace_id: workspace_id.unwrap_or_else(|| DEFAULT_WORKSPACE_ID.to_string()),
        base_url: base_url.trim_end_matches('/').to_string(),
    })
}

fn header_value(headers: &HeaderMap, name: &'static str) -> Result<Option<String>, ConfigError> {
    let header_name = HeaderName::from_static(name);
    headers
        .get(&header_name)
        .map(|value| {
            value
                .to_str()
                .map(str::trim)
                .map(str::to_string)
                .map_err(|_| ConfigError::InvalidHeader(name))
        })
        .transpose()
        .map(|value| value.filter(|value| !value.is_empty()))
}
