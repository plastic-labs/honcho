use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterTarget {
    Workspace,
    Peer,
    Session,
    Message,
    Conclusion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterClause {
    pub sql: String,
    pub bindings: Vec<Value>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FilterError {
    #[error("filters must be a JSON object")]
    InvalidRoot,
    #[error("Column '{0}' is not allowed to be filtered on")]
    UnsupportedColumn(String),
    #[error("Column '{0}' is not allowed to be filtered on or does not exist on Message")]
    UnsupportedMessageColumn(String),
    #[error("Unsupported comparison operator: {0}")]
    UnsupportedOperator(String),
    #[error("{0}")]
    InvalidValue(String),
}

pub fn build_filter_clause(
    target: FilterTarget,
    filters: Option<&Value>,
) -> Result<FilterClause, FilterError> {
    let Some(filters) = filters else {
        return Ok(FilterClause {
            sql: String::new(),
            bindings: Vec::new(),
        });
    };
    if filters.is_null() {
        return Ok(FilterClause {
            sql: String::new(),
            bindings: Vec::new(),
        });
    }
    let object = filters.as_object().ok_or(FilterError::InvalidRoot)?;
    if object.is_empty() {
        return Ok(FilterClause {
            sql: String::new(),
            bindings: Vec::new(),
        });
    }

    let mut builder = FilterBuilder {
        target,
        bindings: Vec::new(),
    };
    let conditions = builder.object_conditions(object)?;
    let sql = if conditions.is_empty() {
        String::new()
    } else {
        format!(" AND {}", conditions.join(" AND "))
    };

    Ok(FilterClause {
        sql,
        bindings: builder.bindings,
    })
}

struct FilterBuilder {
    target: FilterTarget,
    bindings: Vec<Value>,
}

impl FilterBuilder {
    fn object_conditions(
        &mut self,
        object: &serde_json::Map<String, Value>,
    ) -> Result<Vec<String>, FilterError> {
        let mut conditions = Vec::new();

        for (key, value) in object {
            if value == "*" {
                continue;
            }

            if key == "AND" || key == "OR" || key == "NOT" {
                if let Some(condition) = self.logical_condition(key, value)? {
                    conditions.push(condition);
                }
                continue;
            }

            if let Some(condition) = self.field_condition(key, value)? {
                conditions.push(condition);
            }
        }

        Ok(conditions)
    }

    fn logical_condition(
        &mut self,
        key: &str,
        value: &Value,
    ) -> Result<Option<String>, FilterError> {
        let Some(items) = value.as_array() else {
            return Err(FilterError::InvalidRoot);
        };
        let mut parts = Vec::new();
        for item in items {
            let object = item.as_object().ok_or(FilterError::InvalidRoot)?;
            let nested = self.object_conditions(object)?;
            if !nested.is_empty() {
                parts.push(format!("({})", nested.join(" AND ")));
            }
        }

        if parts.is_empty() {
            return Ok(None);
        }

        let joined = match key {
            "AND" => parts.join(" AND "),
            "OR" => parts.join(" OR "),
            "NOT" => parts
                .into_iter()
                .map(|part| format!("NOT {part}"))
                .collect::<Vec<_>>()
                .join(" AND "),
            _ => unreachable!("validated logical key"),
        };
        Ok(Some(format!("({joined})")))
    }

    fn field_condition(&mut self, key: &str, value: &Value) -> Result<Option<String>, FilterError> {
        let column = self.column_name(key)?;
        if column == "metadata" || column == "internal_metadata" {
            return self.metadata_condition(column, value);
        }

        if let Some(comparisons) = comparison_object(value) {
            return self.comparison_conditions(column, comparisons);
        }

        let placeholder = self.push_binding(value.clone());
        Ok(Some(format!("{column} = {placeholder}")))
    }

    fn metadata_condition(
        &mut self,
        column: &'static str,
        value: &Value,
    ) -> Result<Option<String>, FilterError> {
        let object = value.as_object().ok_or(FilterError::InvalidRoot)?;
        let mut direct = serde_json::Map::new();
        let mut comparisons = Vec::new();

        for (key, value) in object {
            if value == "*" {
                continue;
            }
            if let Some(comparison) = comparison_object(value) {
                for (operator, op_value) in comparison {
                    let accessor = format!("{column}->>'{}'", escape_literal(key));
                    let condition = self.metadata_operator_sql(&accessor, operator, op_value)?;
                    if !condition.is_empty() {
                        comparisons.push(condition);
                    }
                }
            } else {
                direct.insert(key.clone(), value.clone());
            }
        }

        let mut conditions = Vec::new();
        if !direct.is_empty() {
            let placeholder = self.push_binding(Value::Object(direct));
            conditions.push(format!("{column} @> {placeholder}::jsonb"));
        }
        conditions.extend(comparisons);

        if conditions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(conditions.join(" AND ")))
        }
    }

    fn comparison_conditions(
        &mut self,
        column: &'static str,
        comparisons: &serde_json::Map<String, Value>,
    ) -> Result<Option<String>, FilterError> {
        let mut conditions = Vec::new();
        for (operator, value) in comparisons {
            if value == "*" {
                continue;
            }
            if operator == "in" {
                let values = value
                    .as_array()
                    .ok_or_else(|| FilterError::UnsupportedOperator(operator.to_string()))?;
                if values.iter().any(|value| value == "*") {
                    continue;
                }
                if values.is_empty() {
                    if column == "created_at" {
                        continue;
                    }
                    conditions.push("FALSE".to_string());
                    continue;
                }
                let placeholders = if column == "created_at" {
                    values
                        .iter()
                        .map(|value| {
                            let placeholder = self.push_binding(normalize_datetime_value(value)?);
                            Ok(format!("{placeholder}::timestamptz"))
                        })
                        .collect::<Result<Vec<_>, FilterError>>()?
                        .join(", ")
                } else {
                    values
                        .iter()
                        .map(|value| self.push_binding(value.clone()))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                conditions.push(format!("{column} IN ({placeholders})"));
                continue;
            }
            conditions.push(self.column_operator_sql(column, operator, value)?);
        }

        if conditions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(conditions.join(" AND ")))
        }
    }

    fn push_binding(&mut self, value: Value) -> String {
        self.bindings.push(value);
        format!("${}", self.bindings.len())
    }

    fn column_name(&self, key: &str) -> Result<&'static str, FilterError> {
        match (self.target, key) {
            (FilterTarget::Message, "peer_id") => Ok("peer_name"),
            (FilterTarget::Message, "session_id") => Ok("session_name"),
            (FilterTarget::Message, "workspace_id") => Ok("workspace_name"),
            (FilterTarget::Message, "token_count") => Ok("token_count"),
            (FilterTarget::Message, "created_at") => Ok("created_at"),
            (FilterTarget::Message, "metadata") => Ok("metadata"),
            (FilterTarget::Message, _) => {
                Err(FilterError::UnsupportedMessageColumn(key.to_string()))
            }
            (FilterTarget::Conclusion, "id") => Ok("id"),
            (FilterTarget::Conclusion, "created_at") => Ok("created_at"),
            (FilterTarget::Conclusion, "session_id" | "session_name") => Ok("session_name"),
            (FilterTarget::Conclusion, "workspace_id" | "workspace_name") => Ok("workspace_name"),
            (FilterTarget::Conclusion, "observer_id" | "observer") => Ok("observer"),
            (FilterTarget::Conclusion, "observed_id" | "observed") => Ok("observed"),
            (FilterTarget::Conclusion, "metadata") => Ok("internal_metadata"),
            (_, "id") => Ok("name"),
            (_, "created_at") => Ok("created_at"),
            (_, "workspace_id") => Ok("workspace_name"),
            (FilterTarget::Session, "is_active") => Ok("is_active"),
            (_, "metadata") => Ok("metadata"),
            _ => Err(FilterError::UnsupportedColumn(key.to_string())),
        }
    }

    fn column_operator_sql(
        &mut self,
        column: &'static str,
        operator: &str,
        value: &Value,
    ) -> Result<String, FilterError> {
        if column == "created_at" {
            let normalized = normalize_datetime_value(value)?;
            let placeholder = self.push_binding(normalized);
            return operator_sql(
                column,
                operator,
                format!("{placeholder}::timestamptz"),
                false,
            );
        }

        let binding = if matches!(operator, "contains" | "icontains") {
            Value::String(escape_ilike_pattern(&value_to_filter_string(value)))
        } else {
            value.clone()
        };
        let placeholder = self.push_binding(binding);
        operator_sql(
            column,
            operator,
            placeholder,
            matches!(operator, "contains" | "icontains"),
        )
    }

    fn metadata_operator_sql(
        &mut self,
        accessor: &str,
        operator: &str,
        value: &Value,
    ) -> Result<String, FilterError> {
        if operator == "in" {
            let values = value
                .as_array()
                .ok_or_else(|| FilterError::UnsupportedOperator(operator.to_string()))?;
            if values.iter().any(|value| value == "*") {
                return Ok(String::new());
            }
            if values.is_empty() {
                return Ok("FALSE".to_string());
            }
            let placeholders = values
                .iter()
                .map(|value| self.push_binding(Value::String(value_to_python_filter_string(value))))
                .collect::<Vec<_>>()
                .join(", ");
            return Ok(format!("{accessor} IN ({placeholders})"));
        }

        if matches!(operator, "gte" | "lte" | "gt" | "lt" | "ne") {
            match metadata_numeric_comparison_value(value)? {
                MetadataComparisonValue::Numeric(binding) => {
                    let placeholder = self.push_binding(binding);
                    return operator_sql(
                        &safe_numeric_accessor(accessor),
                        operator,
                        format!("{placeholder}::numeric"),
                        false,
                    );
                }
                MetadataComparisonValue::String(binding) => {
                    let placeholder = self.push_binding(Value::String(binding));
                    return operator_sql(accessor, operator, placeholder, false);
                }
            }
        }

        let binding = if matches!(operator, "contains" | "icontains") {
            Value::String(escape_ilike_pattern(&value_to_filter_string(value)))
        } else {
            value.clone()
        };
        let placeholder = self.push_binding(binding);
        operator_sql(
            accessor,
            operator,
            placeholder,
            matches!(operator, "contains" | "icontains"),
        )
    }
}

fn comparison_object(value: &Value) -> Option<&serde_json::Map<String, Value>> {
    const OPERATORS: &[&str] = &[
        "gte",
        "lte",
        "gt",
        "lt",
        "ne",
        "in",
        "contains",
        "icontains",
    ];

    value
        .as_object()
        .filter(|object| object.keys().any(|key| OPERATORS.contains(&key.as_str())))
}

fn operator_sql(
    column: &str,
    operator: &str,
    placeholder: String,
    escape_like: bool,
) -> Result<String, FilterError> {
    match operator {
        "gte" => Ok(format!("{column} >= {placeholder}")),
        "lte" => Ok(format!("{column} <= {placeholder}")),
        "gt" => Ok(format!("{column} > {placeholder}")),
        "lt" => Ok(format!("{column} < {placeholder}")),
        "ne" => Ok(format!("{column} != {placeholder}")),
        "contains" | "icontains" => {
            let mut sql = format!("{column} ILIKE '%' || {placeholder} || '%'");
            if escape_like {
                sql.push_str(" ESCAPE '\\'");
            }
            Ok(sql)
        }
        value => Err(FilterError::UnsupportedOperator(value.to_string())),
    }
}

fn escape_literal(value: &str) -> String {
    value.replace('\'', "''")
}

fn escape_ilike_pattern(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn value_to_filter_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn value_to_python_filter_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

enum MetadataComparisonValue {
    Numeric(Value),
    String(String),
}

fn metadata_numeric_comparison_value(
    value: &Value,
) -> Result<MetadataComparisonValue, FilterError> {
    match value {
        Value::Bool(value) => Ok(MetadataComparisonValue::String(value.to_string())),
        Value::Number(_) => Ok(MetadataComparisonValue::Numeric(value.clone())),
        Value::String(value) => {
            if value.parse::<f64>().is_ok() {
                Ok(MetadataComparisonValue::Numeric(Value::String(
                    value.clone(),
                )))
            } else {
                Ok(MetadataComparisonValue::String(value.clone()))
            }
        }
        _ => Err(FilterError::InvalidValue(format!(
            "Invalid value for numeric operator: {value}. Expected a number, got {}",
            json_type_name(value)
        ))),
    }
}

fn safe_numeric_accessor(accessor: &str) -> String {
    format!(
        "CASE WHEN {accessor} = '' THEN NULL WHEN {accessor} IS NULL THEN NULL WHEN {accessor} ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN ({accessor})::numeric ELSE NULL END"
    )
}

fn normalize_datetime_value(value: &Value) -> Result<Value, FilterError> {
    let Value::String(value) = value else {
        return Err(FilterError::InvalidValue(format!(
            "Invalid datetime value: {value}"
        )));
    };
    let parsed = parse_datetime(value)
        .ok_or_else(|| FilterError::InvalidValue(format!("Invalid datetime value: {value}")))?;
    Ok(Value::String(parsed.to_rfc3339()))
}

fn parse_datetime(value: &str) -> Option<DateTime<Utc>> {
    let value = value.trim();
    if value.is_empty()
        || value.contains('\0')
        || value.contains('\r')
        || value.contains('\n')
        || value
            .chars()
            .any(|value| value.is_control() && value != '\t')
    {
        return None;
    }

    if let Ok(parsed) = DateTime::parse_from_rfc3339(&value.replace(['z', 'Z'], "+00:00")) {
        return Some(parsed.with_timezone(&Utc));
    }

    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(parsed) = NaiveDateTime::parse_from_str(value, format) {
            return Some(parsed.and_utc());
        }
    }

    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .map(|datetime| datetime.and_utc())
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
