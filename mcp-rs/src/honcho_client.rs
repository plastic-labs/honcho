use reqwest::{Client, Method};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct HonchoClient {
    base_url: String,
    authorization: String,
    http: Client,
}

#[derive(Debug, Error)]
pub enum HonchoApiError {
    #[error("Honcho API request failed ({method} {url}): status {status}: {body}")]
    Status {
        method: Method,
        url: String,
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("Honcho API request failed ({method} {url}): {source}")]
    Transport {
        method: Method,
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("Honcho API response could not be decoded ({method} {url}): {source}")]
    Decode {
        method: Method,
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("Honcho API response could not be decoded ({method} {url}): {source}")]
    DecodeJson {
        method: Method,
        url: String,
        #[source]
        source: serde_json::Error,
    },
}

impl HonchoClient {
    pub fn new(
        base_url: impl Into<String>,
        authorization: impl Into<String>,
        http: Client,
    ) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            authorization: authorization.into(),
            http,
        }
    }

    pub fn new_for_test(base_url: &str, authorization: &str) -> Self {
        Self::new(base_url, authorization, Client::new())
    }

    pub fn endpoint(&self, segments: &[&str]) -> String {
        let path = segments
            .iter()
            .map(|segment| encode_path_segment(segment))
            .collect::<Vec<_>>()
            .join("/");
        format!("{}/v3/{}", self.base_url, path)
    }

    pub fn endpoint_for_test(&self, segments: &[&str]) -> String {
        self.endpoint(segments)
    }

    pub async fn get_json<T: DeserializeOwned>(
        &self,
        segments: &[&str],
        query: &[(&str, String)],
    ) -> Result<T, HonchoApiError> {
        self.request_json(Method::GET, segments, query, None).await
    }

    pub async fn post_json<T: DeserializeOwned>(
        &self,
        segments: &[&str],
        body: Value,
    ) -> Result<T, HonchoApiError> {
        self.request_json(Method::POST, segments, &[], Some(body))
            .await
    }

    pub async fn post_json_with_query<T: DeserializeOwned>(
        &self,
        segments: &[&str],
        query: &[(&str, String)],
        body: Value,
    ) -> Result<T, HonchoApiError> {
        self.request_json(Method::POST, segments, query, Some(body))
            .await
    }

    pub async fn put_json<T: DeserializeOwned>(
        &self,
        segments: &[&str],
        body: Value,
    ) -> Result<T, HonchoApiError> {
        self.request_json(Method::PUT, segments, &[], Some(body))
            .await
    }

    pub async fn put_json_with_query<T: DeserializeOwned>(
        &self,
        segments: &[&str],
        query: &[(&str, String)],
        body: Value,
    ) -> Result<T, HonchoApiError> {
        self.request_json(Method::PUT, segments, query, Some(body))
            .await
    }

    pub async fn delete_json<T: DeserializeOwned>(
        &self,
        segments: &[&str],
        query: &[(&str, String)],
        body: Option<Value>,
    ) -> Result<T, HonchoApiError> {
        self.request_json(Method::DELETE, segments, query, body)
            .await
    }

    async fn request_json<T: DeserializeOwned>(
        &self,
        method: Method,
        segments: &[&str],
        query: &[(&str, String)],
        body: Option<Value>,
    ) -> Result<T, HonchoApiError> {
        let url = self.endpoint_with_query(segments, query);
        let mut request = self
            .http
            .request(method.clone(), &url)
            .bearer_auth(self.authorization.trim_start_matches("Bearer ").trim());

        if let Some(body) = body {
            request = request.json(&body);
        }

        let response = request
            .send()
            .await
            .map_err(|source| HonchoApiError::Transport {
                method: method.clone(),
                url: url.clone(),
                source,
            })?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(HonchoApiError::Status {
                method,
                url,
                status,
                body,
            });
        }

        if status == reqwest::StatusCode::NO_CONTENT {
            return serde_json::from_value(json!(null)).map_err(|source| {
                HonchoApiError::DecodeJson {
                    method,
                    url,
                    source,
                }
            });
        }

        response
            .json::<T>()
            .await
            .map_err(|source| HonchoApiError::Decode {
                method,
                url,
                source,
            })
    }
}

fn encode_path_segment(segment: &str) -> String {
    let mut encoded = String::new();
    for byte in segment.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(*byte as char)
            }
            byte => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

impl HonchoClient {
    fn endpoint_with_query(&self, segments: &[&str], query: &[(&str, String)]) -> String {
        let mut url = self.endpoint(segments);
        if !query.is_empty() {
            let params = query
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}={}",
                        encode_path_segment(key),
                        encode_path_segment(value)
                    )
                })
                .collect::<Vec<_>>()
                .join("&");
            url.push('?');
            url.push_str(&params);
        }
        url
    }
}
