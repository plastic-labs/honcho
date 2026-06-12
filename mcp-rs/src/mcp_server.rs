use crate::config::parse_config_from_headers;
use crate::honcho_client::HonchoClient;
use crate::tools;
use rmcp::ErrorData as McpError;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, Implementation, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use serde_json::{Map, Value};

#[derive(Debug, Clone)]
pub struct HonchoMcp {
    base_url: String,
    http: reqwest::Client,
}

impl HonchoMcp {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        }
    }
}

impl ServerHandler for HonchoMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_server_info(
            Implementation::new("honcho-mcp-rs", env!("CARGO_PKG_VERSION")),
        )
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tools::get_tool(name)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: tools::all_tools(),
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let Some(parts) = context.extensions.get::<http::request::Parts>() else {
            return Ok(error_result("Missing HTTP request context"));
        };
        let config = match parse_config_from_headers(&parts.headers, &self.base_url) {
            Ok(config) => config,
            Err(error) => return Ok(error_result(error.to_string())),
        };
        let client = HonchoClient::new(&config.base_url, &config.authorization, self.http.clone());
        let args = request.arguments.unwrap_or_default();

        Ok(
            match tools::dispatch(&client, &config, request.name.as_ref(), args).await {
                Ok(value) => text_result(value),
                Err(error) => error_result(error.to_string()),
            },
        )
    }
}

pub fn text_result(value: Value) -> CallToolResult {
    let text = match value {
        Value::String(text) => text,
        value => serde_json::to_string(&value).unwrap_or_else(|_| "null".to_string()),
    };
    CallToolResult::success(vec![Content::text(text)])
}

pub fn error_result(error: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![Content::text(error.into())])
}

pub fn empty_args(args: Option<Map<String, Value>>) -> Map<String, Value> {
    args.unwrap_or_default()
}
