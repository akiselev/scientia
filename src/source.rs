use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSpan {
    pub start: usize,
    pub end: usize,
}

impl SourceSpan {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceSeverity {
    Note,
    Warning,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelatedSpan {
    pub span: SourceSpan,
    pub message: String,
}

/// Stable, structured diagnostic intended for humans, agents, CI and editor/MCP adapters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDiagnostic {
    pub code: String,
    pub severity: SourceSeverity,
    pub message: String,
    pub span: SourceSpan,
    #[serde(default)]
    pub related: Vec<RelatedSpan>,
    #[serde(default)]
    pub hints: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

impl SourceDiagnostic {
    pub fn error(code: impl Into<String>, message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            code: code.into(),
            severity: SourceSeverity::Error,
            message: message.into(),
            span,
            related: vec![],
            hints: vec![],
            phase: None,
        }
    }
    pub fn warning(code: impl Into<String>, message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            code: code.into(),
            severity: SourceSeverity::Warning,
            message: message.into(),
            span,
            related: vec![],
            hints: vec![],
            phase: None,
        }
    }
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hints.push(hint.into());
        self
    }
    pub fn phase(mut self, phase: impl Into<String>) -> Self {
        self.phase = Some(phase.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spanned<T> {
    pub value: T,
    pub span: SourceSpan,
}
