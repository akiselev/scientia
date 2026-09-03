//! Scientific authoring and semantic infrastructure.
//!
//! This module is deliberately compiler-oriented: source declarations become typed data,
//! expressions remain structured, coupling is derived from use, and solver strategy never
//! enters the model semantics.

use crate::source::{SourceDiagnostic, SourceSpan, Spanned};
use quantitas::{Dimension, Quantity, QuantityKindId, QuantityLiteral, UnitId, UnitRegistry};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const SCIENTIFIC_SCHEMA: &str = "scientia-scientific/1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScientificModule {
    pub schema: String,
    pub name: String,
    pub imports: Vec<Spanned<String>>,
    pub models: Vec<ScientificModel>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScientificModel {
    pub name: String,
    pub domains: Vec<DomainDecl>,
    pub fields: Vec<FieldDecl>,
    pub parameters: Vec<ValueDecl>,
    pub constants: Vec<ValueDecl>,
    pub sources: Vec<ValueDecl>,
    /// `input field` / `input value` declarations (SC, `sinbad/ARCHITECTURE.md` §3.3). Skipped
    /// from the digest projection when empty so every pre-SC module keeps its digest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<InputDecl>,
    pub providers: Vec<ProviderDecl>,
    pub properties: Vec<PropertyBinding>,
    pub constitutive_laws: Vec<ConstitutiveBinding>,
    pub equations: Vec<EquationDecl>,
    pub forms: Vec<FormDecl>,
    pub initial_conditions: Vec<ConditionDecl>,
    pub boundary_conditions: Vec<BoundaryConditionDecl>,
    pub interface_conditions: Vec<BoundaryConditionDecl>,
    pub observables: Vec<ObservableDecl>,
    pub invariants: Vec<ObservableDecl>,
    /// `objective NAME { minimize|maximize|measure EXPR; }` (SV1-A). Skipped from the digest
    /// projection when empty so every pre-SV1 module keeps its digest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objectives: Vec<ObjectiveDecl>,
    pub verifications: Vec<VerificationAnnotation>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainDecl {
    pub name: String,
    pub dimension: u8,
    pub coordinates: CoordinateSystem,
    pub coordinates_span: SourceSpan,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateSystem {
    Cartesian,
    Cylindrical,
    Spherical,
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldRole {
    State,
    Unknown,
    Test,
    Trial,
    Coefficient,
    Parameter,
    Derived,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueShape {
    Scalar,
    Vector(u8),
    Tensor { rows: u8, cols: u8 },
    SymmetricTensor(u8),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceSpec {
    pub family: SpaceFamily,
    pub order: u8,
    pub continuity: Continuity,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceFamily {
    H1,
    L2,
    HCurl,
    HDiv,
    Dg,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Continuity {
    Continuous,
    Discontinuous,
    Tangential,
    Normal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldDecl {
    pub name: String,
    pub role: FieldRole,
    pub shape: ValueShape,
    pub space: SpaceSpec,
    pub domain: String,
    pub domain_span: SourceSpan,
    pub quantity_kind: Option<QuantityKindId>,
    pub quantity_kind_span: Option<SourceSpan>,
    pub unit: Option<UnitId>,
    pub unit_span: Option<SourceSpan>,
    pub nominal: Option<QuantityLiteral>,
    pub physical_min: Option<QuantityLiteral>,
    pub physical_max: Option<QuantityLiteral>,
    pub time_role: Option<TimeRole>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueDecl {
    pub name: String,
    pub quantity_kind: Option<QuantityKindId>,
    pub quantity_kind_span: Option<SourceSpan>,
    pub unit: Option<UnitId>,
    pub unit_span: Option<SourceSpan>,
    pub value: Option<Expr>,
    pub span: SourceSpan,
}

/// `input field NAME: Kind on Domain;` or `input value NAME: Kind;` (SC §3.3): a binding slot
/// a system `bind` or a case closes. An input field is externally supplied field data on one
/// domain; an input value is one spatially constant datum. Neither carries a definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputDecl {
    pub name: String,
    pub kind: InputDeclKind,
    pub quantity_kind: Option<QuantityKindId>,
    pub quantity_kind_span: Option<SourceSpan>,
    pub unit: Option<UnitId>,
    pub unit_span: Option<SourceSpan>,
    /// Present exactly for `InputDeclKind::Field`.
    pub domain: Option<String>,
    pub domain_span: Option<SourceSpan>,
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputDeclKind {
    Field,
    Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyBinding {
    pub name: String,
    pub value: Expr,
    pub span: SourceSpan,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstitutiveBinding {
    pub name: String,
    pub law: Expr,
    pub span: SourceSpan,
}

/// `provider NAME(input: Kind, ...) -> Kind { ... }` (GX-A1, contract C1.1). Inputs use the
/// pseudo-kind `selector` (represented as `kind: None`) for a non-physical integer catalog
/// selector; every other input names a Quantitas quantity kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderDecl {
    pub name: String,
    pub inputs: Vec<ProviderInputDecl>,
    pub output_kind: String,
    pub output_kind_span: SourceSpan,
    pub unit: Option<UnitId>,
    pub unit_span: Option<SourceSpan>,
    pub shape: ValueShape,
    pub locality: PropertyLocality,
    pub differentiability: DerivativeContract,
    pub domain: Vec<ProviderDomainBound>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderInputDecl {
    pub name: String,
    /// `None` is the `selector` pseudo-kind; `Some(name)` is a Quantitas quantity kind name.
    pub kind: Option<String>,
    pub kind_span: SourceSpan,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderDomainBound {
    pub input: String,
    pub input_span: SourceSpan,
    pub min: QuantityLiteral,
    pub max: QuantityLiteral,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EquationDecl {
    pub name: String,
    pub domain: Option<String>,
    pub domain_span: Option<SourceSpan>,
    pub lhs: Expr,
    pub rhs: Expr,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FormDecl {
    pub name: String,
    pub integrals: Vec<IntegralDecl>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntegralDecl {
    pub measure: Measure,
    pub target_span: SourceSpan,
    pub integrand: Expr,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Measure {
    Cell(String),
    Boundary(String),
    InteriorFacet(String),
    Interface(String),
    Point(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditionDecl {
    pub target: String,
    pub value: Expr,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundaryConditionDecl {
    pub name: String,
    pub region: Expr,
    pub kind: BoundaryConditionKind,
    pub target: String,
    pub target_span: SourceSpan,
    pub value: Expr,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryConditionKind {
    Dirichlet,
    Neumann,
    Robin,
    Interface,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservableDecl {
    pub name: String,
    pub value: Expr,
    pub span: SourceSpan,
}

/// `objective NAME { minimize EXPR; }`: a scalar functional of the model with an optimization
/// sense (SV1-A, `scientia-derivative-request/2`). An objective is also an observable: it is
/// evaluated like one and appears as an `observable/<name>` slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveDecl {
    pub name: String,
    pub sense: crate::derivative::ObjectiveSense,
    pub value: Expr,
    pub span: SourceSpan,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerificationAnnotation {
    pub name: String,
    pub args: BTreeMap<String, Expr>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Number {
        value: f64,
        /// The literal's exact source spelling (never a `-` sign, which parses as `Unary::Neg`);
        /// this is what `semantic::ExactLiteral` is computed from, so `0.1` and `0.10` differ in
        /// digest identity only if this spelling differs (GX-A2, contract C8).
        lexeme: String,
        unit: Option<String>,
        span: SourceSpan,
    },
    String {
        value: String,
        span: SourceSpan,
    },
    Name {
        name: String,
        span: SourceSpan,
    },
    Unary {
        op: UnaryOp,
        arg: Box<Expr>,
        span: SourceSpan,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: SourceSpan,
    },
    Call {
        function: String,
        args: Vec<Expr>,
        span: SourceSpan,
    },
    Index {
        value: Box<Expr>,
        indices: Vec<Expr>,
        span: SourceSpan,
    },
    Vector {
        elements: Vec<Expr>,
        span: SourceSpan,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Expr {
    pub const fn span(&self) -> SourceSpan {
        match self {
            Self::Number { span, .. }
            | Self::String { span, .. }
            | Self::Name { span, .. }
            | Self::Unary { span, .. }
            | Self::Binary { span, .. }
            | Self::Call { span, .. }
            | Self::Index { span, .. }
            | Self::Vector { span, .. } => *span,
        }
    }

    pub fn synthetic_number(value: f64) -> Self {
        Self::Number {
            value,
            lexeme: format!("{value}"),
            unit: None,
            span: SourceSpan::default(),
        }
    }

    pub fn names(&self, out: &mut BTreeSet<String>) {
        match self {
            Expr::Name { name, .. } => {
                out.insert(name.clone());
            }
            Expr::Unary { arg, .. } => arg.names(out),
            Expr::Binary { lhs, rhs, .. } => {
                lhs.names(out);
                rhs.names(out);
            }
            Expr::Call { args, .. } | Expr::Vector { elements: args, .. } => {
                for arg in args {
                    arg.names(out);
                }
            }
            Expr::Index { value, indices, .. } => {
                value.names(out);
                for i in indices {
                    i.names(out);
                }
            }
            Expr::Number { .. } | Expr::String { .. } => {}
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScientificError {
    #[error("syntax error at {span:?}: {message}")]
    Syntax { message: String, span: SourceSpan },
    #[error("duplicate declaration `{0}`")]
    Duplicate(String),
    #[error("unknown name `{0}`")]
    UnknownName(String),
    #[error("import cycle through `{name}`")]
    ImportCycle { name: String, span: SourceSpan },
    #[error("missing imported module `{name}`")]
    MissingModule { name: String, span: SourceSpan },
    #[error("quantity error: {0}")]
    Quantity(String),
    #[error("property evaluation failed: {0}")]
    Property(String),
}

impl ScientificError {
    pub fn diagnostic(&self) -> SourceDiagnostic {
        match self {
            Self::Syntax { message, span } => {
                SourceDiagnostic::error("PARSE_SYNTAX", message, *span).phase("parsing")
            }
            Self::Duplicate(name) => SourceDiagnostic::error(
                "RESOLVE_DUPLICATE_NAME",
                format!("duplicate declaration `{name}`"),
                SourceSpan::default(),
            )
            .phase("resolution"),
            Self::UnknownName(name) => SourceDiagnostic::error(
                "RESOLVE_UNKNOWN_NAME",
                format!("unknown name `{name}`"),
                SourceSpan::default(),
            )
            .phase("resolution"),
            Self::ImportCycle { name, span } => SourceDiagnostic::error(
                "RESOLVE_IMPORT_CYCLE",
                format!("import cycle through `{name}`"),
                *span,
            )
            .phase("resolution"),
            Self::MissingModule { name, span } => SourceDiagnostic::error(
                "RESOLVE_MISSING_MODULE",
                format!("missing imported module `{name}`"),
                *span,
            )
            .phase("resolution"),
            Self::Quantity(message) => {
                SourceDiagnostic::error("TYPE_QUANTITY", message, SourceSpan::default())
                    .phase("elaboration")
            }
            Self::Property(message) => {
                SourceDiagnostic::error("PROPERTY_EVALUATION", message, SourceSpan::default())
                    .phase("evaluation")
            }
        }
    }
}

// ---------------- lexer/parser ----------------

#[derive(Clone, Debug, PartialEq)]
enum TokenKind {
    Ident(String),
    Number(f64, String),
    String(String),
    Punct(char),
    Op(String),
    Eof,
}
#[derive(Clone, Debug, PartialEq)]
struct Token {
    kind: TokenKind,
    span: SourceSpan,
}

fn lex(input: &str) -> Result<Vec<Token>, Vec<ScientificError>> {
    let bytes = input.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();
    let mut errors = Vec::new();
    while i < bytes.len() {
        let c = input[i..].chars().next().unwrap();
        if c.is_whitespace() {
            i += c.len_utf8();
            continue;
        }
        if c == '#' || input[i..].starts_with("//") {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        if c.is_ascii_alphabetic() || c == '_' || c == 'π' {
            i += c.len_utf8();
            while i < bytes.len() {
                let x = input[i..].chars().next().unwrap();
                if x.is_alphanumeric() || matches!(x, '_' | '.' | 'π') {
                    i += x.len_utf8();
                } else {
                    break;
                }
            }
            out.push(Token {
                kind: TokenKind::Ident(input[start..i].into()),
                span: SourceSpan::new(start, i),
            });
            continue;
        }
        if c.is_ascii_digit()
            || (c == '.'
                && input[i + 1..]
                    .chars()
                    .next()
                    .is_some_and(|x| x.is_ascii_digit()))
        {
            i += c.len_utf8();
            while i < bytes.len() {
                let x = input[i..].chars().next().unwrap();
                if x.is_ascii_digit() || matches!(x, '.' | 'e' | 'E' | '+' | '-') {
                    if (x == '+' || x == '-')
                        && !matches!(input[..i].chars().last(), Some('e' | 'E'))
                    {
                        break;
                    }
                    i += x.len_utf8();
                } else {
                    break;
                }
            }
            match input[start..i].parse::<f64>() {
                Ok(v) => out.push(Token {
                    kind: TokenKind::Number(v, input[start..i].to_string()),
                    span: SourceSpan::new(start, i),
                }),
                Err(_) => errors.push(ScientificError::Syntax {
                    message: "invalid number".into(),
                    span: SourceSpan::new(start, i),
                }),
            }
            continue;
        }
        if c == '"' {
            i += 1;
            let body = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i >= bytes.len() {
                errors.push(ScientificError::Syntax {
                    message: "unterminated string".into(),
                    span: SourceSpan::new(start, bytes.len()),
                });
                break;
            }
            out.push(Token {
                kind: TokenKind::String(input[body..i].into()),
                span: SourceSpan::new(start, i + 1),
            });
            i += 1;
            continue;
        }
        let two = if i + 1 < bytes.len() {
            &input[i..i + 2]
        } else {
            ""
        };
        if matches!(two, "==" | "<=" | ">=" | "->") {
            out.push(Token {
                kind: TokenKind::Op(two.into()),
                span: SourceSpan::new(i, i + 2),
            });
            i += 2;
            continue;
        }
        if "+-*/^=<>".contains(c) {
            out.push(Token {
                kind: TokenKind::Op(c.to_string()),
                span: SourceSpan::new(i, i + c.len_utf8()),
            });
            i += c.len_utf8();
            continue;
        }
        if "{}();:,[]@".contains(c) {
            out.push(Token {
                kind: TokenKind::Punct(c),
                span: SourceSpan::new(i, i + c.len_utf8()),
            });
            i += c.len_utf8();
            continue;
        }
        errors.push(ScientificError::Syntax {
            message: format!("unexpected character `{c}`"),
            span: SourceSpan::new(start, start + c.len_utf8()),
        });
        i += c.len_utf8();
    }
    out.push(Token {
        kind: TokenKind::Eof,
        span: SourceSpan::new(input.len(), input.len()),
    });
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

struct Parser {
    tokens: Vec<Token>,
    i: usize,
    errors: Vec<ScientificError>,
}
impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            i: 0,
            errors: vec![],
        }
    }
    fn token(&self) -> &Token {
        &self.tokens[self.i]
    }
    fn bump(&mut self) -> Token {
        let t = self.tokens[self.i].clone();
        if !matches!(t.kind, TokenKind::Eof) {
            self.i += 1;
        }
        t
    }
    fn ident_is(&self, s: &str) -> bool {
        matches!(&self.token().kind, TokenKind::Ident(x) if x == s)
    }
    /// The identifier after the current token, if any (one-token lookahead for soft keywords).
    fn peek_ident(&self) -> Option<&str> {
        match self.tokens.get(self.i + 1).map(|token| &token.kind) {
            Some(TokenKind::Ident(x)) => Some(x.as_str()),
            _ => None,
        }
    }
    fn expect_ident(&mut self, s: &str) {
        if !self.eat_ident(s) {
            let span = self.token().span;
            self.errors.push(ScientificError::Syntax {
                message: format!("expected `{s}`"),
                span,
            });
        }
    }
    fn eat_ident(&mut self, s: &str) -> bool {
        if self.ident_is(s) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn eat_punct(&mut self, c: char) -> bool {
        if matches!(self.token().kind, TokenKind::Punct(x) if x == c) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn eat_op(&mut self, op: &str) -> bool {
        if matches!(&self.token().kind, TokenKind::Op(x) if x == op) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn expect_punct(&mut self, c: char) -> bool {
        if self.eat_punct(c) {
            true
        } else {
            self.error(format!("expected `{c}`"));
            false
        }
    }
    fn expect_ident_value(&mut self) -> Option<(String, SourceSpan)> {
        let t = self.bump();
        match t.kind {
            TokenKind::Ident(x) => Some((x, t.span)),
            _ => {
                self.errors.push(ScientificError::Syntax {
                    message: "expected identifier".into(),
                    span: t.span,
                });
                None
            }
        }
    }
    fn error(&mut self, message: String) {
        self.errors.push(ScientificError::Syntax {
            message,
            span: self.token().span,
        });
    }
    fn sync(&mut self) {
        while !matches!(self.token().kind, TokenKind::Eof) {
            if self.eat_punct(';') {
                break;
            }
            if matches!(self.token().kind, TokenKind::Punct('}')) {
                break;
            }
            self.bump();
        }
    }

    fn module(&mut self) -> ScientificModule {
        let start = self.token().span.start;
        let name = if self.eat_ident("module") {
            let name = self
                .expect_ident_value()
                .map(|x| x.0)
                .unwrap_or_else(|| "invalid".into());
            self.expect_punct(';');
            name
        } else {
            "main".into()
        };
        let mut imports = vec![];
        while self.eat_ident("use") {
            if let Some((value, span)) = self.expect_ident_value() {
                imports.push(Spanned { value, span });
            }
            self.expect_punct(';');
        }
        let mut models = vec![];
        while !matches!(self.token().kind, TokenKind::Eof) {
            if self.eat_ident("model") {
                if let Some(model) = self.model() {
                    models.push(model);
                }
            } else {
                self.error("expected `model` declaration".into());
                self.sync();
                if matches!(self.token().kind, TokenKind::Punct('}')) {
                    self.bump();
                }
            }
        }
        ScientificModule {
            schema: SCIENTIFIC_SCHEMA.into(),
            name,
            imports,
            models,
            span: SourceSpan::new(start, self.token().span.end),
        }
    }

    fn model(&mut self) -> Option<ScientificModel> {
        let (name, span) = self.expect_ident_value()?;
        self.expect_punct('{');
        let mut model = ScientificModel {
            name,
            domains: vec![],
            fields: vec![],
            parameters: vec![],
            constants: vec![],
            sources: vec![],
            inputs: vec![],
            providers: vec![],
            properties: vec![],
            constitutive_laws: vec![],
            equations: vec![],
            forms: vec![],
            initial_conditions: vec![],
            boundary_conditions: vec![],
            interface_conditions: vec![],
            observables: vec![],
            invariants: vec![],
            objectives: vec![],
            verifications: vec![],
            span,
        };
        while !matches!(self.token().kind, TokenKind::Eof | TokenKind::Punct('}')) {
            if self.eat_ident("domain") {
                if let Some(x) = self.domain() {
                    model.domains.push(x);
                }
            } else if self.eat_ident("field") {
                if let Some(x) = self.field() {
                    model.fields.push(x);
                }
            } else if self.eat_ident("parameter") {
                if let Some(x) = self.value_decl() {
                    model.parameters.push(x);
                }
            } else if self.eat_ident("constant") {
                if let Some(x) = self.value_decl() {
                    model.constants.push(x);
                }
            } else if self.eat_ident("source") {
                if let Some(x) = self.value_decl() {
                    model.sources.push(x);
                }
            } else if self.ident_is("input") && matches!(self.peek_ident(), Some("field" | "value"))
            {
                // Soft keyword (§3.2): `input` opens a declaration only when followed by
                // `field` or `value`, so a provider parameter or symbol named `input` still
                // parses.
                self.bump();
                if let Some(x) = self.input_decl() {
                    model.inputs.push(x);
                }
            } else if self.eat_ident("provider") {
                if let Some(x) = self.provider() {
                    model.providers.push(x);
                }
            } else if self.eat_ident("property") {
                if let Some(x) = self.property() {
                    model.properties.push(x);
                }
            } else if self.eat_ident("constitutive") {
                if let Some(x) = self.constitutive() {
                    model.constitutive_laws.push(x);
                }
            } else if self.eat_ident("equation") {
                if let Some(x) = self.equation() {
                    model.equations.push(x);
                }
            } else if self.eat_ident("form") {
                if let Some(x) = self.form_decl() {
                    model.forms.push(x);
                }
            } else if self.eat_ident("initial") {
                model.initial_conditions.extend(self.assignment_block());
            } else if self.eat_ident("boundary") {
                if let Some(x) = self.boundary(false) {
                    model.boundary_conditions.push(x);
                }
            } else if self.eat_ident("interface") {
                if let Some(x) = self.boundary(true) {
                    model.interface_conditions.push(x);
                }
            } else if self.eat_ident("observable") {
                if let Some(x) = self.observable() {
                    model.observables.push(x);
                }
            } else if self.eat_ident("invariant") {
                if let Some(x) = self.observable() {
                    model.invariants.push(x);
                }
            } else if self.eat_ident("objective") {
                if let Some(x) = self.objective() {
                    model.objectives.push(x);
                }
            } else if self.eat_punct('@') {
                if let Some(x) = self.verification() {
                    model.verifications.push(x);
                }
            } else {
                self.error("unknown model declaration".into());
                self.sync();
            }
        }
        self.expect_punct('}');
        Some(model)
    }

    fn domain(&mut self) -> Option<DomainDecl> {
        let (name, span) = self.expect_ident_value()?;
        self.expect_punct('{');
        let mut dimension = 2;
        let mut coordinates = CoordinateSystem::Cartesian;
        let mut coordinates_span = span;
        while !matches!(self.token().kind, TokenKind::Eof | TokenKind::Punct('}')) {
            let key = self.expect_ident_value()?.0;
            self.eat_op("=");
            if key == "dimension" {
                dimension = self.number_u8(2);
            } else if key == "coordinates" {
                if let Some((v, value_span)) = self.expect_ident_value() {
                    coordinates_span = value_span;
                    coordinates = match v.as_str() {
                        "cartesian" => CoordinateSystem::Cartesian,
                        "cylindrical" => CoordinateSystem::Cylindrical,
                        "spherical" => CoordinateSystem::Spherical,
                        _ => CoordinateSystem::Custom(v),
                    };
                }
            } else {
                self.bump();
            }
            self.eat_punct(';');
        }
        self.expect_punct('}');
        self.eat_punct(';');
        Some(DomainDecl {
            name,
            dimension,
            coordinates,
            coordinates_span,
            span,
        })
    }

    fn field(&mut self) -> Option<FieldDecl> {
        let (name, span) = self.expect_ident_value()?;
        self.expect_punct(':');
        let (role_name, role_span) = self.expect_ident_value()?;
        let role = match role_name.as_str() {
            "state" => FieldRole::State,
            "unknown" => FieldRole::Unknown,
            "test" => FieldRole::Test,
            "trial" => FieldRole::Trial,
            "coefficient" => FieldRole::Coefficient,
            "parameter" => FieldRole::Parameter,
            "derived" => FieldRole::Derived,
            _ => {
                self.errors.push(ScientificError::Syntax {
                    message: format!("unknown field role `{role_name}`"),
                    span: role_span,
                });
                FieldRole::Unknown
            }
        };
        let mut shape = ValueShape::Scalar;
        if self.ident_is("scalar") {
            self.bump();
        } else if self.eat_ident("vector") {
            self.expect_punct('(');
            let n = self.number_u8(3);
            self.expect_punct(')');
            shape = ValueShape::Vector(n);
        } else if self.eat_ident("tensor") {
            self.expect_punct('(');
            let a = self.number_u8(3);
            self.eat_punct(',');
            let b = self.number_u8(a);
            self.expect_punct(')');
            shape = ValueShape::Tensor { rows: a, cols: b };
        }
        let (family, family_span) = self.expect_ident_value()?;
        self.expect_punct('(');
        let mut order = 1;
        if self.eat_ident("order") {
            self.eat_op("=");
            order = self.number_u8(1);
        } else if matches!(self.token().kind, TokenKind::Number(_, _)) {
            order = self.number_u8(1);
        }
        self.expect_punct(')');
        let space = match family.as_str() {
            "H1" => SpaceSpec {
                family: SpaceFamily::H1,
                order,
                continuity: Continuity::Continuous,
            },
            "L2" => SpaceSpec {
                family: SpaceFamily::L2,
                order,
                continuity: Continuity::Discontinuous,
            },
            "HCurl" | "Hcurl" => SpaceSpec {
                family: SpaceFamily::HCurl,
                order,
                continuity: Continuity::Tangential,
            },
            "HDiv" | "Hdiv" => SpaceSpec {
                family: SpaceFamily::HDiv,
                order,
                continuity: Continuity::Normal,
            },
            "DG" => SpaceSpec {
                family: SpaceFamily::Dg,
                order,
                continuity: Continuity::Discontinuous,
            },
            _ => {
                self.errors.push(ScientificError::Syntax {
                    message: format!("unsupported function space `{family}`"),
                    span: family_span,
                });
                SpaceSpec {
                    family: SpaceFamily::H1,
                    order,
                    continuity: Continuity::Continuous,
                }
            }
        };
        if !self.eat_ident("on") {
            self.error("field requires `on <domain>`".into());
        }
        let (domain, domain_span) = self
            .expect_ident_value()
            .unwrap_or_else(|| ("Omega".into(), self.token().span));
        let mut quantity_kind = None;
        let mut quantity_kind_span = None;
        let mut unit = None;
        let mut unit_span = None;
        let mut nominal = None;
        let mut physical_min = None;
        let mut physical_max = None;
        let mut time_role = None;
        if self.eat_punct('{') {
            while !matches!(self.token().kind, TokenKind::Eof | TokenKind::Punct('}')) {
                let key = self.expect_ident_value()?.0;
                self.eat_op("=");
                match key.as_str() {
                    "quantity" => {
                        if let Some((kind, span)) = self.expect_ident_value() {
                            quantity_kind = Some(QuantityKindId::new(kind));
                            quantity_kind_span = Some(span);
                        }
                    }
                    "unit" => {
                        if let Some((text, span)) = self.unit_expr() {
                            unit = Some(UnitId::new(text));
                            unit_span = Some(span);
                        }
                    }
                    "nominal" => nominal = self.quantity_literal(quantity_kind.clone()),
                    "min" => physical_min = self.quantity_literal(quantity_kind.clone()),
                    "max" => physical_max = self.quantity_literal(quantity_kind.clone()),
                    "time_role" => {
                        if let Some((name, span)) = self.expect_ident_value() {
                            time_role = match name.as_str() {
                                "algebraic" => Some(TimeRole::Algebraic),
                                "differential" => Some(TimeRole::Differential),
                                _ => {
                                    self.errors.push(ScientificError::Syntax {
                                        message: format!("unknown time role `{name}`"),
                                        span,
                                    });
                                    None
                                }
                            };
                        }
                    }
                    _ => {
                        self.error(format!("unknown field attribute `{key}`"));
                        self.sync();
                    }
                }
                self.eat_punct(';');
            }
            self.expect_punct('}');
        }
        self.eat_punct(';');
        Some(FieldDecl {
            name,
            role,
            shape,
            space,
            domain,
            domain_span,
            quantity_kind,
            quantity_kind_span,
            unit,
            unit_span,
            nominal,
            physical_min,
            physical_max,
            time_role,
            span,
        })
    }

    fn number_u8(&mut self, default: u8) -> u8 {
        let t = self.bump();
        if let TokenKind::Number(v, _) = t.kind
            && v.fract() == 0.0
            && (0.0..=f64::from(u8::MAX)).contains(&v)
        {
            v as u8
        } else {
            self.errors.push(ScientificError::Syntax {
                message: "expected an integer from 0 through 255".into(),
                span: t.span,
            });
            default
        }
    }
    fn quantity_literal(&mut self, kind: Option<QuantityKindId>) -> Option<QuantityLiteral> {
        let t = self.bump();
        let value = if let TokenKind::Number(v, _) = t.kind {
            v
        } else {
            self.error("expected quantity value".into());
            return None;
        };
        let unit = self.expect_ident_value()?.0;
        Some(QuantityLiteral {
            value,
            unit: UnitId::new(unit),
            kind: kind.unwrap_or_else(|| QuantityKindId::new("scientia:Unspecified")),
        })
    }

    fn input_decl(&mut self) -> Option<InputDecl> {
        let kind = if self.eat_ident("field") {
            InputDeclKind::Field
        } else {
            self.expect_ident("value");
            InputDeclKind::Value
        };
        let (name, span) = self.expect_ident_value()?;
        let mut quantity_kind = None;
        let mut quantity_kind_span = None;
        let mut unit = None;
        let mut unit_span = None;
        if self.eat_punct(':')
            && let Some((kind_name, kind_span)) = self.expect_ident_value()
        {
            quantity_kind = Some(QuantityKindId::new(kind_name));
            quantity_kind_span = Some(kind_span);
        }
        if self.eat_punct('[') {
            if let Some((unit_name, span)) = self.expect_ident_value() {
                unit = Some(UnitId::new(unit_name));
                unit_span = Some(span);
            }
            self.expect_punct(']');
        }
        let mut domain = None;
        let mut domain_span = None;
        match kind {
            InputDeclKind::Field => {
                if self.eat_ident("on") {
                    if let Some((domain_name, span)) = self.expect_ident_value() {
                        domain = Some(domain_name);
                        domain_span = Some(span);
                    }
                } else {
                    self.errors.push(ScientificError::Syntax {
                        message: format!("input field `{name}` must declare `on <domain>`"),
                        span,
                    });
                }
            }
            InputDeclKind::Value => {
                if self.ident_is("on") {
                    self.errors.push(ScientificError::Syntax {
                        message: format!("input value `{name}` has no domain; use `input field`"),
                        span,
                    });
                }
            }
        }
        if self.eat_op("=") {
            let value = self.expr(0)?;
            self.errors.push(ScientificError::Syntax {
                message: format!(
                    "input `{name}` cannot carry a definition; declare a `source`, `parameter`, \
                     or `property` instead"
                ),
                span: value.span(),
            });
        }
        self.expect_punct(';');
        Some(InputDecl {
            name,
            kind,
            quantity_kind,
            quantity_kind_span,
            unit,
            unit_span,
            domain,
            domain_span,
            span,
        })
    }

    fn value_decl(&mut self) -> Option<ValueDecl> {
        let (name, span) = self.expect_ident_value()?;
        let mut kind = None;
        let mut quantity_kind_span = None;
        let mut unit = None;
        let mut unit_span = None;
        if self.eat_punct(':')
            && let Some((name, span)) = self.expect_ident_value()
        {
            kind = Some(QuantityKindId::new(name));
            quantity_kind_span = Some(span);
        }
        if self.eat_punct('[') {
            if let Some((name, span)) = self.expect_ident_value() {
                unit = Some(UnitId::new(name));
                unit_span = Some(span);
            }
            self.expect_punct(']');
        }
        let value = if self.eat_op("=") {
            Some(self.expr(0)?)
        } else {
            None
        };
        self.expect_punct(';');
        Some(ValueDecl {
            name,
            quantity_kind: kind,
            quantity_kind_span,
            unit,
            unit_span,
            value,
            span,
        })
    }
    fn property(&mut self) -> Option<PropertyBinding> {
        let (name, span) = self.expect_ident_value()?;
        if !self.eat_op("=") {
            self.error("property requires `=`".into());
        }
        let value = self.expr(0)?;
        self.expect_punct(';');
        Some(PropertyBinding { name, value, span })
    }
    fn constitutive(&mut self) -> Option<ConstitutiveBinding> {
        let (name, span) = self.expect_ident_value()?;
        self.eat_op("=");
        let law = self.expr(0)?;
        self.expect_punct(';');
        Some(ConstitutiveBinding { name, law, span })
    }

    fn provider(&mut self) -> Option<ProviderDecl> {
        let (name, span) = self.expect_ident_value()?;
        self.expect_punct('(');
        let mut inputs = vec![];
        while !matches!(self.token().kind, TokenKind::Eof | TokenKind::Punct(')')) {
            let (input_name, input_span) = self.expect_ident_value()?;
            self.expect_punct(':');
            let (kind_name, kind_span) = self.expect_ident_value()?;
            let kind = if kind_name == "selector" {
                None
            } else {
                Some(kind_name)
            };
            inputs.push(ProviderInputDecl {
                name: input_name,
                kind,
                kind_span,
                span: input_span,
            });
            if !self.eat_punct(',') {
                break;
            }
        }
        self.expect_punct(')');
        if !self.eat_op("->") {
            self.error("provider requires `-> QuantityKind`".into());
        }
        let (output_kind, output_kind_span) = self
            .expect_ident_value()
            .unwrap_or_else(|| ("scientia:Unspecified".into(), self.token().span));
        let mut unit = None;
        let mut unit_span = None;
        let mut shape = ValueShape::Scalar;
        let mut locality = PropertyLocality::Pointwise;
        let mut differentiability = DerivativeContract::Symbolic;
        let mut domain = vec![];
        if self.eat_punct('{') {
            while !matches!(self.token().kind, TokenKind::Eof | TokenKind::Punct('}')) {
                if self.eat_ident("domain") {
                    self.expect_punct('{');
                    while !matches!(self.token().kind, TokenKind::Eof | TokenKind::Punct('}')) {
                        let Some((input_name, input_span)) = self.expect_ident_value() else {
                            self.sync();
                            continue;
                        };
                        if !self.eat_ident("in") {
                            self.error("provider domain bound requires `in`".into());
                        }
                        self.expect_punct('[');
                        let min = self.quantity_literal(None);
                        self.eat_punct(',');
                        let max = self.quantity_literal(None);
                        self.expect_punct(']');
                        self.eat_punct(';');
                        if let (Some(min), Some(max)) = (min, max) {
                            domain.push(ProviderDomainBound {
                                input: input_name,
                                input_span,
                                min,
                                max,
                                span: input_span,
                            });
                        }
                    }
                    self.expect_punct('}');
                    self.eat_punct(';');
                    continue;
                }
                let Some((key, _key_span)) = self.expect_ident_value() else {
                    self.sync();
                    continue;
                };
                self.eat_op("=");
                match key.as_str() {
                    "unit" => {
                        if let Some((text, span)) = self.unit_expr() {
                            unit = Some(UnitId::new(text));
                            unit_span = Some(span);
                        }
                    }
                    "shape" => shape = self.provider_shape(),
                    "locality" => {
                        if let Some((name, name_span)) = self.expect_ident_value() {
                            locality = match name.as_str() {
                                "pointwise" => PropertyLocality::Pointwise,
                                "element_constant" => PropertyLocality::ElementConstant,
                                "external" => PropertyLocality::ExternalProvider,
                                _ => {
                                    self.errors.push(ScientificError::Syntax {
                                        message: format!("unknown provider locality `{name}`"),
                                        span: name_span,
                                    });
                                    PropertyLocality::Pointwise
                                }
                            };
                        }
                    }
                    "differentiability" => {
                        if let Some((name, name_span)) = self.expect_ident_value() {
                            differentiability = match name.as_str() {
                                "symbolic" => DerivativeContract::Symbolic,
                                "analytic_provided" => DerivativeContract::AnalyticProvided,
                                "automatic" => DerivativeContract::Automatic,
                                "piecewise" => DerivativeContract::Piecewise,
                                "numerical_allowed" => DerivativeContract::NumericalAllowed,
                                "none" => DerivativeContract::None,
                                _ => {
                                    self.errors.push(ScientificError::Syntax {
                                        message: format!(
                                            "unknown provider differentiability `{name}`"
                                        ),
                                        span: name_span,
                                    });
                                    DerivativeContract::Symbolic
                                }
                            };
                        }
                    }
                    _ => {
                        self.error(format!("unknown provider attribute `{key}`"));
                        self.sync();
                    }
                }
                self.eat_punct(';');
            }
            self.expect_punct('}');
        }
        self.eat_punct(';');
        Some(ProviderDecl {
            name,
            inputs,
            output_kind,
            output_kind_span,
            unit,
            unit_span,
            shape,
            locality,
            differentiability,
            domain,
            span,
        })
    }

    /// A product/quotient/power unit-symbol grammar (`W/(m*K)`, `m^2/s`), mirroring
    /// `quantitas::UnitRegistry::parse_unit_expression`'s `expr := term (('*'|'/') term)*`,
    /// `term := primary ('^' '-'? DIGITS)?`, `primary := IDENT | '(' expr ')'`. The result is not
    /// dimensionally evaluated here; it becomes the literal `UnitId` symbol text resolved later
    /// (either directly or, for a compound expression, through `parse_unit_expression`) against
    /// the registry, exactly like a simple `unit = K;` symbol.
    fn unit_expr(&mut self) -> Option<(String, SourceSpan)> {
        let (mut text, mut span) = self.unit_term()?;
        loop {
            if self.eat_op("*") {
                let (rhs, rhs_span) = self.unit_term()?;
                text.push('*');
                text.push_str(&rhs);
                span = SourceSpan::new(span.start, rhs_span.end);
            } else if self.eat_op("/") {
                let (rhs, rhs_span) = self.unit_term()?;
                text.push('/');
                text.push_str(&rhs);
                span = SourceSpan::new(span.start, rhs_span.end);
            } else {
                break;
            }
        }
        Some((text, span))
    }

    /// A unit atom with an optional integer exponent (`m^2`, `s^-1`).
    fn unit_term(&mut self) -> Option<(String, SourceSpan)> {
        let (mut text, mut span) = self.unit_atom()?;
        if self.eat_op("^") {
            let negative = self.eat_op("-");
            let exponent_token = self.bump();
            match exponent_token.kind {
                TokenKind::Number(value, _) if value.fract() == 0.0 => {
                    text.push('^');
                    if negative {
                        text.push('-');
                    }
                    text.push_str(&(value as i64).to_string());
                    span = SourceSpan::new(span.start, exponent_token.span.end);
                }
                _ => {
                    self.errors.push(ScientificError::Syntax {
                        message: "expected an integer unit exponent".into(),
                        span: exponent_token.span,
                    });
                }
            }
        }
        Some((text, span))
    }

    fn unit_atom(&mut self) -> Option<(String, SourceSpan)> {
        if matches!(self.token().kind, TokenKind::Punct('(')) {
            let start = self.token().span.start;
            self.bump();
            let (inner, _) = self.unit_expr()?;
            let end = self.token().span.end;
            self.expect_punct(')');
            Some((format!("({inner})"), SourceSpan::new(start, end)))
        } else {
            self.expect_ident_value()
        }
    }

    /// Non-erroring continuation of a numeric literal's trailing-unit chain (contract GX-F3
    /// item 4): starting from an already-parsed first unit term (`text`, ending at `end`), keep
    /// extending through `*`/`/` **only** when the token immediately following the operator can
    /// itself start a unit atom (an identifier or `(`); anything else -- a number, a closing
    /// token, end of input -- leaves that operator for the surrounding expression grammar
    /// instead of erroring. This is what makes `2.5 W/(m*K)` and `1.0 m^2/s` single literals
    /// while `300 K / 2` and `2.5 W / rho` still divide by a following number or bare symbol.
    ///
    /// Known trade-off: because this is a syntactic, registry-free rule, `2.5 W / rho` (a plain
    /// identifier, not a number, immediately after an already unit-bearing literal) is
    /// indistinguishable at parse time from a genuine compound unit and is swallowed into the
    /// literal's unit text -- exactly as the pre-existing single-trailing-ident rule already
    /// swallowed a lone identifier directly after a number. A mis-swallowed unit surfaces at
    /// elaboration as `RESOLVE_UNKNOWN_UNIT` rather than silently succeeding, so this is a
    /// compile error rather than a silent semantic change for any model that hits it.
    fn trailing_unit_chain(&mut self, mut text: String, mut end: usize) -> (String, usize) {
        loop {
            let is_chain_op =
                matches!(&self.token().kind, TokenKind::Op(op) if op == "*" || op == "/");
            if !is_chain_op {
                break;
            }
            let continues = matches!(
                self.tokens.get(self.i + 1).map(|token| &token.kind),
                Some(TokenKind::Ident(_)) | Some(TokenKind::Punct('('))
            );
            if !continues {
                break;
            }
            let TokenKind::Op(op) = self.bump().kind else {
                unreachable!("matched TokenKind::Op above")
            };
            let Some((atom_text, atom_span)) = self.unit_term() else {
                break;
            };
            text.push_str(&op);
            text.push_str(&atom_text);
            end = atom_span.end;
        }
        (text, end)
    }

    fn provider_shape(&mut self) -> ValueShape {
        if self.eat_ident("scalar") {
            ValueShape::Scalar
        } else if self.eat_ident("vector") {
            self.expect_punct('(');
            let n = self.number_u8(3);
            self.expect_punct(')');
            ValueShape::Vector(n)
        } else if self.eat_ident("tensor") {
            self.expect_punct('(');
            let a = self.number_u8(3);
            self.eat_punct(',');
            let b = self.number_u8(a);
            self.expect_punct(')');
            ValueShape::Tensor { rows: a, cols: b }
        } else if self.eat_ident("symmetric_tensor") {
            self.expect_punct('(');
            let n = self.number_u8(3);
            self.expect_punct(')');
            ValueShape::SymmetricTensor(n)
        } else {
            self.error("unknown provider shape".into());
            ValueShape::Scalar
        }
    }

    fn equation(&mut self) -> Option<EquationDecl> {
        let (name, span) = self.expect_ident_value()?;
        let (domain, domain_span) = if self.eat_ident("on") {
            match self.expect_ident_value() {
                Some((name, span)) => (Some(name), Some(span)),
                None => (None, None),
            }
        } else {
            (None, None)
        };
        self.expect_punct('{');
        // Top-level equality belongs to the equation declaration, not the expression tree.
        // Parse above equality precedence so comparisons remain legal inside each side.
        let lhs = self.expr(2)?;
        if !self.eat_op("=") {
            self.error("equation requires `=`".into());
        }
        let rhs = self.expr(0)?;
        self.eat_punct(';');
        self.expect_punct('}');
        self.eat_punct(';');
        Some(EquationDecl {
            name,
            domain,
            domain_span,
            lhs,
            rhs,
            span,
        })
    }

    fn form_decl(&mut self) -> Option<FormDecl> {
        let (name, span) = self.expect_ident_value()?;
        self.expect_punct('{');
        let mut integrals = vec![];
        while !matches!(self.token().kind, TokenKind::Eof | TokenKind::Punct('}')) {
            let measure_span = self.token().span;
            let measure_name = self.expect_ident_value()?.0;
            self.expect_punct('(');
            let (target, target_span) = self.expect_ident_value()?;
            self.expect_punct(')');
            self.eat_punct(':');
            let integrand = self.expr(0)?;
            self.expect_punct(';');
            let measure = match measure_name.as_str() {
                "cell" => Measure::Cell(target),
                "boundary" => Measure::Boundary(target),
                "interior_facet" => Measure::InteriorFacet(target),
                "interface" => Measure::Interface(target),
                "point" => Measure::Point(target),
                _ => {
                    self.errors.push(ScientificError::Syntax {
                        message: format!("unknown measure `{measure_name}`"),
                        span: measure_span,
                    });
                    Measure::Cell(target)
                }
            };
            integrals.push(IntegralDecl {
                measure,
                target_span,
                integrand,
                span: measure_span,
            });
        }
        self.expect_punct('}');
        self.eat_punct(';');
        Some(FormDecl {
            name,
            integrals,
            span,
        })
    }

    fn assignment_block(&mut self) -> Vec<ConditionDecl> {
        let mut out = vec![];
        if !self.expect_punct('{') {
            return out;
        }
        while !matches!(self.token().kind, TokenKind::Eof | TokenKind::Punct('}')) {
            let Some((target, span)) = self.expect_ident_value() else {
                self.sync();
                continue;
            };
            self.eat_op("=");
            if let Some(value) = self.expr(0) {
                out.push(ConditionDecl {
                    target,
                    value,
                    span,
                });
            }
            self.eat_punct(';');
        }
        self.expect_punct('}');
        self.eat_punct(';');
        out
    }

    fn boundary(&mut self, interface: bool) -> Option<BoundaryConditionDecl> {
        let (name, span) = self.expect_ident_value()?;
        if !self.eat_ident("on") {
            self.error("boundary/interface requires `on`".into());
        }
        let region = self.expr(0)?;
        self.expect_punct('{');
        let kind_name = self.expect_ident_value()?.0;
        let kind = if interface {
            BoundaryConditionKind::Interface
        } else {
            match kind_name.as_str() {
                "dirichlet" => BoundaryConditionKind::Dirichlet,
                "neumann" => BoundaryConditionKind::Neumann,
                "robin" => BoundaryConditionKind::Robin,
                _ => {
                    self.error(format!("unknown boundary condition `{kind_name}`"));
                    BoundaryConditionKind::Dirichlet
                }
            }
        };
        let (target, target_span) = self.expect_ident_value()?;
        self.eat_op("=");
        let value = self.expr(0)?;
        self.eat_punct(';');
        self.expect_punct('}');
        self.eat_punct(';');
        Some(BoundaryConditionDecl {
            name,
            region,
            kind,
            target,
            target_span,
            value,
            span,
        })
    }

    fn observable(&mut self) -> Option<ObservableDecl> {
        let (name, span) = self.expect_ident_value()?;
        self.expect_punct('{');
        let value = self.expr(0)?;
        self.eat_punct(';');
        self.expect_punct('}');
        self.eat_punct(';');
        Some(ObservableDecl { name, value, span })
    }
    fn objective(&mut self) -> Option<ObjectiveDecl> {
        use crate::derivative::ObjectiveSense;
        let (name, span) = self.expect_ident_value()?;
        self.expect_punct('{');
        let sense = if self.eat_ident("minimize") {
            ObjectiveSense::Minimize
        } else if self.eat_ident("maximize") {
            ObjectiveSense::Maximize
        } else if self.eat_ident("measure") {
            ObjectiveSense::Measure
        } else {
            let span = self.token().span;
            self.errors.push(ScientificError::Syntax {
                message: format!(
                    "objective `{name}` must open with `minimize`, `maximize`, or `measure`"
                ),
                span,
            });
            return None;
        };
        let value = self.expr(0)?;
        self.eat_punct(';');
        self.expect_punct('}');
        self.eat_punct(';');
        Some(ObjectiveDecl {
            name,
            sense,
            value,
            span,
        })
    }
    fn verification(&mut self) -> Option<VerificationAnnotation> {
        let (name, span) = self.expect_ident_value()?;
        let mut args = BTreeMap::new();
        if self.eat_punct('(') {
            while !self.eat_punct(')') && !matches!(self.token().kind, TokenKind::Eof) {
                let key = self.expect_ident_value()?.0;
                self.eat_op("=");
                let value = self.expr(0)?;
                args.insert(key, value);
                if !self.eat_punct(',') {
                    self.expect_punct(')');
                    break;
                }
            }
        }
        self.eat_punct(';');
        Some(VerificationAnnotation { name, args, span })
    }

    fn expr(&mut self, min_bp: u8) -> Option<Expr> {
        let mut lhs = match self.bump() {
            Token {
                kind: TokenKind::Number(value, lexeme),
                span: number_span,
            } => {
                let (unit, end) = if matches!(self.token().kind, TokenKind::Ident(_)) {
                    match self.unit_term() {
                        Some((first_text, first_span)) => {
                            let (text, end) = self.trailing_unit_chain(first_text, first_span.end);
                            (Some(text), end)
                        }
                        None => (None, number_span.end),
                    }
                } else {
                    (None, number_span.end)
                };
                Expr::Number {
                    value,
                    lexeme,
                    unit,
                    span: SourceSpan::new(number_span.start, end),
                }
            }
            Token {
                kind: TokenKind::String(s),
                span,
            } => Expr::String { value: s, span },
            Token {
                kind: TokenKind::Ident(name),
                span: name_span,
            } => {
                if self.eat_punct('(') {
                    let mut args = vec![];
                    if !self.eat_punct(')') {
                        loop {
                            args.push(self.expr(0)?);
                            if self.eat_punct(')') {
                                break;
                            }
                            self.expect_punct(',');
                        }
                    }
                    let end = self.tokens[self.i.saturating_sub(1)].span.end;
                    Expr::Call {
                        function: name,
                        args,
                        span: SourceSpan::new(name_span.start, end),
                    }
                } else {
                    Expr::Name {
                        name,
                        span: name_span,
                    }
                }
            }
            Token {
                kind: TokenKind::Op(op),
                span: op_span,
            } if op == "-" => {
                let arg = Box::new(self.expr(11)?);
                let span = SourceSpan::new(op_span.start, arg.span().end);
                Expr::Unary {
                    op: UnaryOp::Neg,
                    arg,
                    span,
                }
            }
            Token {
                kind: TokenKind::Punct('('),
                ..
            } => {
                let x = self.expr(0)?;
                self.expect_punct(')');
                x
            }
            Token {
                kind: TokenKind::Punct('['),
                span: open_span,
            } => {
                let mut xs = vec![];
                if !self.eat_punct(']') {
                    loop {
                        xs.push(self.expr(0)?);
                        if self.eat_punct(']') {
                            break;
                        }
                        self.expect_punct(',');
                    }
                }
                let end = self.tokens[self.i.saturating_sub(1)].span.end;
                Expr::Vector {
                    elements: xs,
                    span: SourceSpan::new(open_span.start, end),
                }
            }
            t => {
                self.errors.push(ScientificError::Syntax {
                    message: "expected expression".into(),
                    span: t.span,
                });
                return None;
            }
        };
        loop {
            if self.eat_punct('[') {
                let start = lhs.span().start;
                let mut indices = vec![];
                if !self.eat_punct(']') {
                    loop {
                        indices.push(self.expr(0)?);
                        if self.eat_punct(']') {
                            break;
                        }
                        self.expect_punct(',');
                    }
                }
                lhs = Expr::Index {
                    value: Box::new(lhs),
                    indices,
                    span: SourceSpan::new(start, self.tokens[self.i.saturating_sub(1)].span.end),
                };
                continue;
            }
            let op_text = match &self.token().kind {
                TokenKind::Op(x) => x.clone(),
                _ => break,
            };
            let Some((lbp, rbp, op)) = binary_binding(&op_text) else {
                break;
            };
            if lbp < min_bp {
                break;
            }
            self.bump();
            let rhs = self.expr(rbp)?;
            let span = SourceSpan::new(lhs.span().start, rhs.span().end);
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Some(lhs)
    }
}

fn binary_binding(op: &str) -> Option<(u8, u8, BinaryOp)> {
    Some(match op {
        "=" | "==" => (1, 2, BinaryOp::Eq),
        "<" => (3, 4, BinaryOp::Lt),
        "<=" => (3, 4, BinaryOp::Le),
        ">" => (3, 4, BinaryOp::Gt),
        ">=" => (3, 4, BinaryOp::Ge),
        "+" => (5, 6, BinaryOp::Add),
        "-" => (5, 6, BinaryOp::Sub),
        "*" => (7, 8, BinaryOp::Mul),
        "/" => (7, 8, BinaryOp::Div),
        "^" => (10, 9, BinaryOp::Pow),
        _ => return None,
    })
}

pub fn parse_scientific_module(input: &str) -> Result<ScientificModule, Vec<ScientificError>> {
    let tokens = lex(input)?;
    let mut parser = Parser::new(tokens);
    let module = parser.module();
    if parser.errors.is_empty() {
        Ok(module)
    } else {
        Err(parser.errors)
    }
}

/// Parse one standalone expression using the same grammar as any expression position in a
/// model (GX-A2, contracts C7/C3.2), with no surrounding module, domains, or declarations. A
/// case-file `expression`/`piecewise` binding's `expr` text is parsed with this so it can be
/// turned into a [`PropertyModel`] and then, via [`crate::projection::lift_standalone_expr`] and
/// [`crate::property_kernel::lower_property_kernel`], into a Malleus kernel with symbolic
/// tangents.
pub fn parse_expression(text: &str) -> Result<Expr, ScientificError> {
    let tokens = lex(text).map_err(|mut errors| errors.remove(0))?;
    let mut parser = Parser::new(tokens);
    let expr = parser.expr(0).ok_or_else(|| {
        parser
            .errors
            .first()
            .cloned()
            .unwrap_or(ScientificError::Syntax {
                message: "expected an expression".into(),
                span: SourceSpan::default(),
            })
    })?;
    if !matches!(parser.token().kind, TokenKind::Eof) {
        parser.error("expected end of expression".into());
    }
    if let Some(error) = parser.errors.into_iter().next() {
        return Err(error);
    }
    Ok(expr)
}

/// Parse `.res` source with stable structured diagnostics for CI, editors, and agents.
pub fn parse_scientific_module_diagnostics(
    input: &str,
) -> Result<ScientificModule, Vec<SourceDiagnostic>> {
    parse_scientific_module(input)
        .map_err(|errors| errors.iter().map(ScientificError::diagnostic).collect())
}

pub fn semantic_digest(module: &ScientificModule) -> String {
    // Spans are provenance, not scientific meaning. Strip them before hashing so
    // whitespace/comments/formatting do not perturb the physics identity.
    let mut value =
        serde_json::to_value(module).expect("scientific module serialization is infallible");
    fn normalize_digest_projection(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                map.retain(|key, _| key != "span" && !key.ends_with("_span"));
                for child in map.values_mut() {
                    normalize_digest_projection(child);
                }
            }
            serde_json::Value::Array(items) => {
                for child in items.iter_mut() {
                    normalize_digest_projection(child);
                }
                // Only declaration-like collections and explicit string/value sets are
                // order-independent. Expression argument and integral order is preserved.
                if items
                    .iter()
                    .all(|item| item.get("name").and_then(|x| x.as_str()).is_some())
                {
                    items.sort_by(|a, b| {
                        a.get("name")
                            .and_then(|x| x.as_str())
                            .cmp(&b.get("name").and_then(|x| x.as_str()))
                    });
                } else if items.iter().all(|item| item.as_str().is_some()) {
                    items.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
                } else if items
                    .iter()
                    .all(|item| item.get("value").and_then(|x| x.as_str()).is_some())
                {
                    items.sort_by(|a, b| {
                        a.get("value")
                            .and_then(|x| x.as_str())
                            .cmp(&b.get("value").and_then(|x| x.as_str()))
                    });
                }
            }
            _ => {}
        }
    }
    normalize_digest_projection(&mut value);
    let bytes =
        serde_json::to_vec(&value).expect("semantic projection serialization is infallible");
    blake3::hash(&bytes).to_hex().to_string()
}

pub fn format_scientific_module(module: &ScientificModule) -> String {
    let mut out = String::new();
    out.push_str(&format!("module {};\n\n", module.name));
    for import in &module.imports {
        out.push_str(&format!("use {};\n", import.value));
    }
    if !module.imports.is_empty() {
        out.push('\n');
    }
    for model in &module.models {
        out.push_str(&format!("model {} {{\n", model.name));
        for d in &model.domains {
            out.push_str(&format!(
                "    domain {} {{ dimension = {}; coordinates = {}; }}\n",
                d.name,
                d.dimension,
                coordinate_name(&d.coordinates)
            ));
        }
        for f in &model.fields {
            out.push_str(&format!(
                "    field {}: {} {} {}(order={}) on {}",
                f.name,
                field_role_name(&f.role),
                shape_name(&f.shape),
                space_name(&f.space.family),
                f.space.order,
                f.domain
            ));
            if f.quantity_kind.is_some()
                || f.unit.is_some()
                || f.nominal.is_some()
                || f.physical_min.is_some()
                || f.physical_max.is_some()
                || f.time_role.is_some()
            {
                out.push_str(" {\n");
                if let Some(k) = &f.quantity_kind {
                    out.push_str(&format!("        quantity = {};\n", k.as_str()));
                }
                if let Some(u) = &f.unit {
                    out.push_str(&format!("        unit = {};\n", u.as_str()));
                }
                if let Some(n) = &f.nominal {
                    out.push_str(&format!(
                        "        nominal = {} {};\n",
                        n.value,
                        n.unit.as_str()
                    ));
                }
                if let Some(n) = &f.physical_min {
                    out.push_str(&format!("        min = {} {};\n", n.value, n.unit.as_str()));
                }
                if let Some(n) = &f.physical_max {
                    out.push_str(&format!("        max = {} {};\n", n.value, n.unit.as_str()));
                }
                if let Some(role) = f.time_role {
                    let role = match role {
                        TimeRole::Differential => "differential",
                        TimeRole::Algebraic => "algebraic",
                    };
                    out.push_str(&format!("        time_role = {role};\n"));
                }
                out.push_str("    }");
            }
            out.push_str(";\n");
        }
        for p in &model.parameters {
            out.push_str(&format_value_decl("parameter", p));
        }
        for c in &model.constants {
            out.push_str(&format_value_decl("constant", c));
        }
        for s in &model.sources {
            out.push_str(&format_value_decl("source", s));
        }
        for input in &model.inputs {
            let ty = input
                .quantity_kind
                .as_ref()
                .map(|x| format!(": {}", x.as_str()))
                .unwrap_or_default();
            let unit = input
                .unit
                .as_ref()
                .map(|x| format!(" [{}]", x.as_str()))
                .unwrap_or_default();
            let domain = input
                .domain
                .as_ref()
                .map(|x| format!(" on {x}"))
                .unwrap_or_default();
            let kind = match input.kind {
                InputDeclKind::Field => "field",
                InputDeclKind::Value => "value",
            };
            out.push_str(&format!(
                "    input {kind} {}{ty}{unit}{domain};\n",
                input.name
            ));
        }
        for p in &model.providers {
            out.push_str(&format!(
                "    provider {}({}) -> {}",
                p.name,
                p.inputs
                    .iter()
                    .map(|input| format!(
                        "{}: {}",
                        input.name,
                        input.kind.as_deref().unwrap_or("selector")
                    ))
                    .collect::<Vec<_>>()
                    .join(", "),
                p.output_kind
            ));
            let has_body = p.unit.is_some()
                || p.shape != ValueShape::Scalar
                || p.locality != PropertyLocality::Pointwise
                || p.differentiability != DerivativeContract::Symbolic
                || !p.domain.is_empty();
            if has_body {
                out.push_str(" {\n");
                if let Some(unit) = &p.unit {
                    out.push_str(&format!("        unit = {};\n", unit.as_str()));
                }
                if p.shape != ValueShape::Scalar {
                    out.push_str(&format!(
                        "        shape = {};\n",
                        provider_shape_name(&p.shape)
                    ));
                }
                if p.locality != PropertyLocality::Pointwise {
                    out.push_str(&format!(
                        "        locality = {};\n",
                        provider_locality_name(&p.locality)
                    ));
                }
                if p.differentiability != DerivativeContract::Symbolic {
                    out.push_str(&format!(
                        "        differentiability = {};\n",
                        provider_differentiability_name(&p.differentiability)
                    ));
                }
                if !p.domain.is_empty() {
                    out.push_str("        domain {\n");
                    for bound in &p.domain {
                        out.push_str(&format!(
                            "            {} in [{} {}, {} {}];\n",
                            bound.input,
                            bound.min.value,
                            bound.min.unit.as_str(),
                            bound.max.value,
                            bound.max.unit.as_str()
                        ));
                    }
                    out.push_str("        }\n");
                }
                out.push_str("    }");
            }
            out.push_str(";\n");
        }
        for p in &model.properties {
            out.push_str(&format!(
                "    property {} = {};\n",
                p.name,
                format_expr(&p.value)
            ));
        }
        for c in &model.constitutive_laws {
            out.push_str(&format!(
                "    constitutive {} = {};\n",
                c.name,
                format_expr(&c.law)
            ));
        }
        for e in &model.equations {
            out.push_str(&format!(
                "    equation {}{} {{ {} = {}; }}\n",
                e.name,
                e.domain
                    .as_ref()
                    .map(|d| format!(" on {d}"))
                    .unwrap_or_default(),
                format_expr(&e.lhs),
                format_expr(&e.rhs)
            ));
        }
        for form in &model.forms {
            out.push_str(&format!("    form {} {{\n", form.name));
            for integral in &form.integrals {
                let (measure, target) = match &integral.measure {
                    Measure::Cell(target) => ("cell", target),
                    Measure::Boundary(target) => ("boundary", target),
                    Measure::InteriorFacet(target) => ("interior_facet", target),
                    Measure::Interface(target) => ("interface", target),
                    Measure::Point(target) => ("point", target),
                };
                out.push_str(&format!(
                    "        {measure}({target}): {};\n",
                    format_expr(&integral.integrand)
                ));
            }
            out.push_str("    }\n");
        }
        if !model.initial_conditions.is_empty() {
            out.push_str("    initial {\n");
            for c in &model.initial_conditions {
                out.push_str(&format!(
                    "        {} = {};\n",
                    c.target,
                    format_expr(&c.value)
                ));
            }
            out.push_str("    }\n");
        }
        for bc in &model.boundary_conditions {
            let kind = match bc.kind {
                BoundaryConditionKind::Dirichlet => "dirichlet",
                BoundaryConditionKind::Neumann => "neumann",
                BoundaryConditionKind::Robin => "robin",
                BoundaryConditionKind::Interface => "interface",
            };
            out.push_str(&format!(
                "    boundary {} on {} {{\n        {kind} {} = {};\n    }}\n",
                bc.name,
                format_expr(&bc.region),
                bc.target,
                format_expr(&bc.value)
            ));
        }
        for bc in &model.interface_conditions {
            out.push_str(&format!(
                "    interface {} on {} {{\n        interface {} = {};\n    }}\n",
                bc.name,
                format_expr(&bc.region),
                bc.target,
                format_expr(&bc.value)
            ));
        }
        for o in &model.observables {
            out.push_str(&format!(
                "    observable {} {{ {}; }}\n",
                o.name,
                format_expr(&o.value)
            ));
        }
        for i in &model.invariants {
            out.push_str(&format!(
                "    invariant {} {{ {}; }}\n",
                i.name,
                format_expr(&i.value)
            ));
        }
        for objective in &model.objectives {
            let sense = match objective.sense {
                crate::derivative::ObjectiveSense::Minimize => "minimize",
                crate::derivative::ObjectiveSense::Maximize => "maximize",
                crate::derivative::ObjectiveSense::Measure => "measure",
            };
            out.push_str(&format!(
                "    objective {} {{ {sense} {}; }}\n",
                objective.name,
                format_expr(&objective.value)
            ));
        }
        for v in &model.verifications {
            out.push_str(&format!("    @{}", v.name));
            if !v.args.is_empty() {
                out.push('(');
                out.push_str(
                    &v.args
                        .iter()
                        .map(|(k, x)| format!("{k} = {}", format_expr(x)))
                        .collect::<Vec<_>>()
                        .join(", "),
                );
                out.push(')');
            }
            out.push_str(";\n");
        }
        out.push_str("}\n\n");
    }
    out
}
fn coordinate_name(c: &CoordinateSystem) -> &str {
    match c {
        CoordinateSystem::Cartesian => "cartesian",
        CoordinateSystem::Cylindrical => "cylindrical",
        CoordinateSystem::Spherical => "spherical",
        CoordinateSystem::Custom(x) => x,
    }
}
fn field_role_name(r: &FieldRole) -> &str {
    match r {
        FieldRole::State => "state",
        FieldRole::Unknown => "unknown",
        FieldRole::Test => "test",
        FieldRole::Trial => "trial",
        FieldRole::Coefficient => "coefficient",
        FieldRole::Parameter => "parameter",
        FieldRole::Derived => "derived",
    }
}
fn shape_name(s: &ValueShape) -> String {
    match s {
        ValueShape::Scalar => "scalar".into(),
        ValueShape::Vector(n) => format!("vector({n})"),
        ValueShape::Tensor { rows, cols } => format!("tensor({rows},{cols})"),
        ValueShape::SymmetricTensor(n) => format!("tensor({n},{n})"),
    }
}
/// Unlike [`shape_name`] (used for `FieldDecl`, whose parser never produces
/// `SymmetricTensor`), provider shapes can parse `symmetric_tensor(n)` directly, so this
/// renders it distinctly to keep provider formatting idempotent.
fn provider_shape_name(s: &ValueShape) -> String {
    match s {
        ValueShape::SymmetricTensor(n) => format!("symmetric_tensor({n})"),
        other => shape_name(other),
    }
}
fn provider_locality_name(l: &PropertyLocality) -> &'static str {
    match l {
        PropertyLocality::Pointwise => "pointwise",
        PropertyLocality::ElementConstant => "element_constant",
        PropertyLocality::ExternalProvider => "external",
    }
}
fn provider_differentiability_name(d: &DerivativeContract) -> &'static str {
    match d {
        DerivativeContract::Symbolic => "symbolic",
        DerivativeContract::AnalyticProvided => "analytic_provided",
        DerivativeContract::Automatic => "automatic",
        DerivativeContract::Piecewise => "piecewise",
        DerivativeContract::NumericalAllowed => "numerical_allowed",
        DerivativeContract::None => "none",
    }
}
fn space_name(s: &SpaceFamily) -> &str {
    match s {
        SpaceFamily::H1 => "H1",
        SpaceFamily::L2 => "L2",
        SpaceFamily::HCurl => "HCurl",
        SpaceFamily::HDiv => "HDiv",
        SpaceFamily::Dg => "DG",
    }
}
fn format_value_decl(kind: &str, d: &ValueDecl) -> String {
    let ty = d
        .quantity_kind
        .as_ref()
        .map(|x| format!(": {}", x.as_str()))
        .unwrap_or_default();
    let unit = d
        .unit
        .as_ref()
        .map(|x| format!(" [{}]", x.as_str()))
        .unwrap_or_default();
    let value = d
        .value
        .as_ref()
        .map(|x| format!(" = {}", format_expr(x)))
        .unwrap_or_default();
    format!("    {kind} {}{ty}{unit}{value};\n", d.name)
}
/// Canonical source rendering of one authored expression (the formatter's spelling).
pub fn format_expression(expression: &Expr) -> String {
    format_expr(expression)
}

fn format_expr(e: &Expr) -> String {
    match e {
        Expr::Number { value, unit, .. } => format!(
            "{value}{}",
            unit.as_ref().map(|u| format!(" {u}")).unwrap_or_default()
        ),
        Expr::String { value, .. } => format!("\"{value}\""),
        Expr::Name { name, .. } => name.clone(),
        Expr::Unary { arg, .. } => format!("-{}", format_expr(arg)),
        Expr::Binary { op, lhs, rhs, .. } => format!(
            "({} {} {})",
            format_expr(lhs),
            binary_name(*op),
            format_expr(rhs)
        ),
        Expr::Call { function, args, .. } => format!(
            "{}({})",
            function,
            args.iter().map(format_expr).collect::<Vec<_>>().join(", ")
        ),
        Expr::Index { value, indices, .. } => format!(
            "{}[{}]",
            format_expr(value),
            indices
                .iter()
                .map(format_expr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Vector { elements: xs, .. } => format!(
            "[{}]",
            xs.iter().map(format_expr).collect::<Vec<_>>().join(", ")
        ),
    }
}
fn binary_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Pow => "^",
        BinaryOp::Eq => "=",
        BinaryOp::Lt => "<",
        BinaryOp::Le => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Ge => ">=",
    }
}

pub trait ModuleSource {
    fn load(&self, name: &str) -> Option<String>;
}
impl ModuleSource for BTreeMap<String, String> {
    fn load(&self, name: &str) -> Option<String> {
        self.get(name).cloned()
    }
}

/// A [`ModuleSource`] that refuses every `use` import (contract GX-F4): `load` always returns
/// `None`, so `resolve_modules` reports `RESOLVE_MISSING_MODULE` for any model that declares an
/// import. Hermetic callers (this crate's own tests, and any caller with no module root) pass
/// this so a model with no `use` statements still elaborates -- `resolve_modules` only calls
/// `load` for a declared import, never unconditionally.
pub struct NoImports;
impl ModuleSource for NoImports {
    fn load(&self, _name: &str) -> Option<String> {
        None
    }
}

/// A [`ModuleSource`] that maps a dotted module name to a `.res` file under `root` (contract
/// GX-F4): `use physics.providers.thermal;` loads `<root>/physics/providers/thermal.res`. A
/// missing or unreadable file is treated as a missing module (`None`), which `resolve_modules`
/// reports as `RESOLVE_MISSING_MODULE`.
pub struct FilesystemModuleSource {
    pub root: std::path::PathBuf,
}
impl ModuleSource for FilesystemModuleSource {
    fn load(&self, name: &str) -> Option<String> {
        let relative = format!("{}.res", name.replace('.', "/"));
        std::fs::read_to_string(self.root.join(relative)).ok()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolvedModules {
    pub modules: BTreeMap<String, ScientificModule>,
    pub semantic_digest: String,
    /// Per-module [`ModuleDigest`]s (SC §2.1), keyed like `modules`; the identity half of every
    /// [`crate::SourceLocator`] and `GlobalDeclId` that points into this closure.
    #[serde(default)]
    pub module_digests: BTreeMap<String, ModuleDigest>,
}

/// `blake3` of a module's span-stripped parse (`sinbad/ARCHITECTURE.md` §2.1): the same value
/// as [`semantic_digest`], carried as a type so declaration identities and source locators
/// cannot be confused with artifact [`crate::Digest`]s.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModuleDigest(pub String);

impl ModuleDigest {
    pub fn of(module: &ScientificModule) -> Self {
        Self(semantic_digest(module))
    }
}

impl std::fmt::Display for ModuleDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub fn resolve_modules(
    root: ScientificModule,
    source: &(impl ModuleSource + ?Sized),
) -> Result<ResolvedModules, ScientificError> {
    fn visit(
        name: &str,
        module: ScientificModule,
        source: &(impl ModuleSource + ?Sized),
        state: &mut BTreeMap<String, u8>,
        out: &mut BTreeMap<String, ScientificModule>,
    ) -> Result<(), ScientificError> {
        match state.get(name).copied() {
            Some(1) => {
                return Err(ScientificError::ImportCycle {
                    name: name.into(),
                    span: module.span,
                });
            }
            Some(2) => return Ok(()),
            _ => {}
        }
        state.insert(name.into(), 1);
        for import in &module.imports {
            let text =
                source
                    .load(&import.value)
                    .ok_or_else(|| ScientificError::MissingModule {
                        name: import.value.clone(),
                        span: import.span,
                    })?;
            let parsed =
                parse_scientific_module(&text).map_err(|e| e.into_iter().next().unwrap())?;
            if matches!(state.get(&import.value), Some(1)) {
                return Err(ScientificError::ImportCycle {
                    name: import.value.clone(),
                    span: import.span,
                });
            }
            visit(&import.value, parsed, source, state, out)?;
        }
        state.insert(name.into(), 2);
        out.insert(name.into(), module);
        Ok(())
    }
    let root_name = root.name.clone();
    let mut state = BTreeMap::new();
    let mut modules = BTreeMap::new();
    visit(&root_name, root, source, &mut state, &mut modules)?;
    let semantic_projection = modules
        .iter()
        .map(|(name, module)| (name, semantic_digest(module)))
        .collect::<BTreeMap<_, _>>();
    let bytes = serde_json::to_vec(&semantic_projection).unwrap();
    let digest = blake3::hash(&bytes).to_hex().to_string();
    let module_digests = semantic_projection
        .into_iter()
        .map(|(name, digest)| (name.clone(), ModuleDigest(digest)))
        .collect();
    Ok(ResolvedModules {
        modules,
        semantic_digest: digest,
        module_digests,
    })
}

// ---------------- properties/material semantics ----------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorSymmetry {
    None,
    Symmetric,
    MajorMinor,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameSemantics {
    Scalar,
    Material,
    Reference,
    Spatial,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyLocality {
    Pointwise,
    ElementConstant,
    ExternalProvider,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DerivativeContract {
    Symbolic,
    AnalyticProvided,
    Automatic,
    Piecewise,
    NumericalAllowed,
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyInput {
    pub name: String,
    pub quantity_kind: QuantityKindId,
    pub dimension: Dimension,
    pub shape: ValueShape,
    pub physical_min: Option<f64>,
    pub physical_max: Option<f64>,
    pub nominal: Option<Quantity>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyOutput {
    pub quantity_kind: QuantityKindId,
    pub dimension: Dimension,
    pub shape: ValueShape,
    pub symmetry: TensorSymmetry,
    pub frame: FrameSemantics,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertySignature {
    pub id: String,
    pub inputs: Vec<PropertyInput>,
    pub output: PropertyOutput,
    pub locality: PropertyLocality,
    pub differentiability: DerivativeContract,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyBranch {
    pub when: Option<Predicate>,
    pub value: Expr,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PropertyModel {
    Constant(Expr),
    Expression(Expr),
    Piecewise(Vec<PropertyBranch>),
    Table(PropertyTable),
    External(PropertyProviderRef),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyProviderRef {
    pub provider: String,
    pub property: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Predicate {
    pub variable: String,
    pub op: CompareOp,
    pub value: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyTable {
    pub axes: Vec<TableAxis>,
    pub values: Vec<f64>,
    pub interpolation: Interpolation,
    pub derivative_policy: TableDerivativePolicy,
    pub out_of_range: OutOfValidityPolicy,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableAxis {
    pub name: String,
    pub points: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interpolation {
    Linear,
    Multilinear,
    MonotoneCubic,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TableDerivativePolicy {
    PiecewiseConstantSlope,
    Numerical,
    Unavailable,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutOfValidityPolicy {
    Error,
    Warn,
    ExplicitExtrapolation(String),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputBounds {
    pub input: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyDomain {
    pub physical_bounds: Vec<InputBounds>,
    pub validity_bounds: Vec<InputBounds>,
    pub phase_constraints: Vec<String>,
    pub composition_constraints: Vec<String>,
    pub assumptions: Vec<String>,
    pub out_of_validity: OutOfValidityPolicy,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum UncertaintyModel {
    StandardAbsolute(f64),
    StandardRelative(f64),
    Expanded {
        value: f64,
        confidence: f64,
        coverage_factor: Option<f64>,
    },
    TablePerPoint(Vec<f64>),
    CovarianceRef(String),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyEvidence {
    pub sources: Vec<String>,
    pub dataset_digest: Option<String>,
    pub fit_digest: Option<String>,
    pub uncertainty: Option<UncertaintyModel>,
    pub notes: BTreeMap<String, String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PropertyDefinition {
    pub signature: PropertySignature,
    pub model: PropertyModel,
    pub domain: PropertyDomain,
    pub evidence: PropertyEvidence,
}

impl PropertyDefinition {
    pub fn evaluate(&self, inputs: &BTreeMap<String, f64>) -> Result<f64, ScientificError> {
        for bound in &self.domain.physical_bounds {
            check_bound(bound, inputs).map_err(ScientificError::Property)?;
        }
        for bound in &self.domain.validity_bounds {
            if let Err(message) = check_bound(bound, inputs)
                && matches!(self.domain.out_of_validity, OutOfValidityPolicy::Error)
            {
                return Err(ScientificError::Property(message));
            }
        }
        evaluate_property_model(&self.model, inputs)
    }
}
fn check_bound(b: &InputBounds, inputs: &BTreeMap<String, f64>) -> Result<(), String> {
    let v = *inputs
        .get(&b.input)
        .ok_or_else(|| format!("missing property input `{}`", b.input))?;
    if b.min.is_some_and(|x| v < x) || b.max.is_some_and(|x| v > x) {
        Err(format!(
            "input `{}`={v} outside [{:?},{:?}]",
            b.input, b.min, b.max
        ))
    } else {
        Ok(())
    }
}
fn predicate(p: &Predicate, inputs: &BTreeMap<String, f64>) -> bool {
    let Some(v) = inputs.get(&p.variable) else {
        return false;
    };
    match p.op {
        CompareOp::Lt => *v < p.value,
        CompareOp::Le => *v <= p.value,
        CompareOp::Gt => *v > p.value,
        CompareOp::Ge => *v >= p.value,
    }
}
fn evaluate_property_model(
    model: &PropertyModel,
    inputs: &BTreeMap<String, f64>,
) -> Result<f64, ScientificError> {
    match model {
        PropertyModel::Constant(e) | PropertyModel::Expression(e) => eval_expr(e, inputs),
        PropertyModel::Piecewise(bs) => bs
            .iter()
            .find(|b| b.when.as_ref().is_none_or(|p| predicate(p, inputs)))
            .map(|b| eval_expr(&b.value, inputs))
            .unwrap_or_else(|| {
                Err(ScientificError::Property(
                    "no piecewise branch matched".into(),
                ))
            }),
        PropertyModel::Table(t) => table_evaluate(t, inputs),
        PropertyModel::External(p) => Err(ScientificError::Property(format!(
            "external provider {}:{} requires runtime ABI",
            p.provider, p.property
        ))),
    }
}

pub fn eval_expr(expr: &Expr, env: &BTreeMap<String, f64>) -> Result<f64, ScientificError> {
    let e = |x: &Expr| eval_expr(x, env);
    Ok(match expr {
        Expr::Number { value, .. } => *value,
        Expr::Name { name, .. } => *env
            .get(name)
            .ok_or_else(|| ScientificError::UnknownName(name.clone()))?,
        Expr::Unary { arg, .. } => -e(arg)?,
        Expr::Binary { op, lhs, rhs, .. } => {
            let a = e(lhs)?;
            let b = e(rhs)?;
            match op {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
                BinaryOp::Div => a / b,
                BinaryOp::Pow => a.powf(b),
                BinaryOp::Eq => (a == b) as u8 as f64,
                BinaryOp::Lt => (a < b) as u8 as f64,
                BinaryOp::Le => (a <= b) as u8 as f64,
                BinaryOp::Gt => (a > b) as u8 as f64,
                BinaryOp::Ge => (a >= b) as u8 as f64,
            }
        }
        Expr::Call { function, args, .. } => {
            let xs = args.iter().map(e).collect::<Result<Vec<_>, _>>()?;
            match (function.as_str(), xs.as_slice()) {
                ("sin", [x]) => x.sin(),
                ("cos", [x]) => x.cos(),
                ("exp", [x]) => x.exp(),
                ("log" | "ln", [x]) => x.ln(),
                ("sqrt", [x]) => x.sqrt(),
                ("abs", [x]) => x.abs(),
                ("min", [a, b]) => a.min(*b),
                ("max", [a, b]) => a.max(*b),
                _ => {
                    return Err(ScientificError::Property(format!(
                        "cannot numerically evaluate `{function}`"
                    )));
                }
            }
        }
        Expr::String { .. } | Expr::Index { .. } | Expr::Vector { .. } => {
            return Err(ScientificError::Property("non-scalar expression".into()));
        }
    })
}

fn table_evaluate(
    t: &PropertyTable,
    inputs: &BTreeMap<String, f64>,
) -> Result<f64, ScientificError> {
    match t.axes.as_slice() {
        [axis] => {
            if t.values.len() != axis.points.len() {
                return Err(ScientificError::Property("1-D table shape mismatch".into()));
            }
            let x = *inputs
                .get(&axis.name)
                .ok_or_else(|| ScientificError::UnknownName(axis.name.clone()))?;
            linear_axis(axis, &t.values, x, &t.out_of_range).map(|x| x.0)
        }
        [a, b] => bilinear(t, a, b, inputs),
        _ => Err(ScientificError::Property(
            "tables currently support one or two axes".into(),
        )),
    }
}
fn linear_axis(
    axis: &TableAxis,
    values: &[f64],
    x: f64,
    policy: &OutOfValidityPolicy,
) -> Result<(f64, f64), ScientificError> {
    if axis.points.len() < 2 {
        return Err(ScientificError::Property(
            "table axis needs >=2 points".into(),
        ));
    }
    let outside = x < axis.points[0] || x > *axis.points.last().unwrap();
    if outside && matches!(policy, OutOfValidityPolicy::Error) {
        return Err(ScientificError::Property(format!(
            "{x} outside table axis `{}`",
            axis.name
        )));
    }
    let mut i = 0;
    while i + 1 < axis.points.len() - 1 && x > axis.points[i + 1] {
        i += 1;
    }
    let x0 = axis.points[i];
    let x1 = axis.points[i + 1];
    let slope = (values[i + 1] - values[i]) / (x1 - x0);
    Ok((values[i] + slope * (x - x0), slope))
}
fn bilinear(
    t: &PropertyTable,
    a: &TableAxis,
    b: &TableAxis,
    inputs: &BTreeMap<String, f64>,
) -> Result<f64, ScientificError> {
    if t.values.len() != a.points.len() * b.points.len() {
        return Err(ScientificError::Property("2-D table shape mismatch".into()));
    }
    let x = *inputs
        .get(&a.name)
        .ok_or_else(|| ScientificError::UnknownName(a.name.clone()))?;
    let y = *inputs
        .get(&b.name)
        .ok_or_else(|| ScientificError::UnknownName(b.name.clone()))?;
    let bracket = |points: &[f64], v: f64| {
        let mut i = 0;
        while i + 1 < points.len() - 1 && v > points[i + 1] {
            i += 1;
        }
        i
    };
    if (x < a.points[0]
        || x > *a.points.last().unwrap()
        || y < b.points[0]
        || y > *b.points.last().unwrap())
        && matches!(t.out_of_range, OutOfValidityPolicy::Error)
    {
        return Err(ScientificError::Property(
            "point outside bilinear table".into(),
        ));
    }
    let i = bracket(&a.points, x);
    let j = bracket(&b.points, y);
    let idx = |ii: usize, jj: usize| ii * b.points.len() + jj;
    let tx = (x - a.points[i]) / (a.points[i + 1] - a.points[i]);
    let ty = (y - b.points[j]) / (b.points[j + 1] - b.points[j]);
    let q00 = t.values[idx(i, j)];
    let q10 = t.values[idx(i + 1, j)];
    let q01 = t.values[idx(i, j + 1)];
    let q11 = t.values[idx(i + 1, j + 1)];
    Ok((1.0 - tx) * (1.0 - ty) * q00
        + tx * (1.0 - ty) * q10
        + (1.0 - tx) * ty * q01
        + tx * ty * q11)
}

// ---------------- constitutive semantics ----------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstitutiveLaw {
    pub id: String,
    pub driving: Vec<LawVariable>,
    pub forces: Vec<LawVariable>,
    pub internal_state: Vec<StateVariable>,
    pub potential: Option<Expr>,
    pub direct_relations: BTreeMap<String, Expr>,
    pub dissipation: Option<Expr>,
    pub tangent: TangentContract,
    pub update: UpdateContract,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LawVariable {
    pub name: String,
    pub quantity_kind: QuantityKindId,
    pub shape: ValueShape,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateVariable {
    pub name: String,
    pub shape: ValueShape,
    pub initial: Expr,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TangentContract {
    Symbolic,
    Automatic,
    AnalyticProvided,
    NumericalAllowed,
    None,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateContract {
    Stateless,
    TransactionalLocal { may_request_step_reduction: bool },
}

pub fn standard_constitutive_laws() -> Vec<&'static str> {
    vec![
        "thermal.fourier",
        "diffusion.fick",
        "electrical.ohm",
        "mechanics.hooke_isotropic",
        "fluids.newtonian",
    ]
}

// ---------------- coupling semantics ----------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnknownBlock {
    pub name: String,
    pub field: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingReason {
    DirectFieldUse,
    DefinedValueDependency(String),
    PropertyDependency(String),
    ConstitutiveDependency(String),
    InterfaceTerm,
    HistoryState,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouplingEdge {
    pub from: String,
    pub to: String,
    pub reason: CouplingReason,
    pub path: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockDerivative {
    pub residual: String,
    pub unknown: String,
    pub structurally_nonzero: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouplingGraph {
    pub unknowns: Vec<UnknownBlock>,
    pub residual_blocks: Vec<String>,
    pub edges: Vec<CouplingEdge>,
    pub derivatives: Vec<BlockDerivative>,
}

pub(crate) fn transitive_field_dependencies<'a>(
    model: &'a ScientificModel,
    expressions: impl IntoIterator<Item = &'a Expr>,
) -> BTreeSet<String> {
    let fields = model
        .fields
        .iter()
        .filter(|field| matches!(field.role, FieldRole::State | FieldRole::Unknown))
        .map(|field| field.name.as_str())
        .collect::<BTreeSet<_>>();
    let definitions = model
        .parameters
        .iter()
        .chain(model.constants.iter())
        .chain(model.sources.iter())
        .filter_map(|value| {
            value
                .value
                .as_ref()
                .map(|expression| (value.name.as_str(), expression))
        })
        .chain(
            model
                .properties
                .iter()
                .map(|property| (property.name.as_str(), &property.value)),
        )
        .chain(
            model
                .constitutive_laws
                .iter()
                .map(|law| (law.name.as_str(), &law.law)),
        )
        .collect::<BTreeMap<_, _>>();

    fn trace(
        symbol: &str,
        fields: &BTreeSet<&str>,
        definitions: &BTreeMap<&str, &Expr>,
        expanding: &mut BTreeSet<String>,
        dependencies: &mut BTreeSet<String>,
    ) {
        if fields.contains(symbol) {
            dependencies.insert(symbol.to_owned());
            return;
        }
        if !expanding.insert(symbol.to_owned()) {
            return;
        }
        if let Some(expression) = definitions.get(symbol) {
            let mut names = BTreeSet::new();
            expression.names(&mut names);
            for name in names {
                trace(&name, fields, definitions, expanding, dependencies);
            }
        }
        expanding.remove(symbol);
    }

    let mut dependencies = BTreeSet::new();
    for expression in expressions {
        let mut names = BTreeSet::new();
        expression.names(&mut names);
        for name in names {
            trace(
                &name,
                &fields,
                &definitions,
                &mut BTreeSet::new(),
                &mut dependencies,
            );
        }
    }
    dependencies
}

pub fn derive_coupling_graph(model: &ScientificModel) -> CouplingGraph {
    let field_names: BTreeSet<_> = model
        .fields
        .iter()
        .filter(|f| matches!(f.role, FieldRole::State | FieldRole::Unknown))
        .map(|f| f.name.clone())
        .collect();
    let property_map: BTreeMap<_, _> = model
        .properties
        .iter()
        .map(|p| (p.name.clone(), p.value.clone()))
        .collect();
    let value_map: BTreeMap<_, _> = model
        .parameters
        .iter()
        .chain(model.constants.iter())
        .chain(model.sources.iter())
        .filter_map(|value| {
            value
                .value
                .as_ref()
                .map(|expression| (value.name.clone(), expression.clone()))
        })
        .collect();
    let constitutive_map: BTreeMap<_, _> = model
        .constitutive_laws
        .iter()
        .map(|law| (law.name.clone(), law.law.clone()))
        .collect();

    struct TraceContext<'a> {
        field_names: &'a BTreeSet<String>,
        value_map: &'a BTreeMap<String, Expr>,
        property_map: &'a BTreeMap<String, Expr>,
        constitutive_map: &'a BTreeMap<String, Expr>,
    }
    fn trace(
        symbol: &str,
        residual: &str,
        context: &TraceContext<'_>,
        path: &mut Vec<String>,
        seen: &mut BTreeSet<String>,
        reason: Option<CouplingReason>,
        out: &mut Vec<CouplingEdge>,
    ) {
        if context.field_names.contains(symbol) {
            let mut full = vec![symbol.to_string()];
            full.extend(path.iter().cloned());
            full.push(residual.to_string());
            out.push(CouplingEdge {
                from: symbol.to_string(),
                to: residual.to_string(),
                reason: reason.unwrap_or(CouplingReason::DirectFieldUse),
                path: full,
            });
            return;
        }
        if !seen.insert(symbol.to_string()) {
            return;
        }
        if let Some(expr) = context.value_map.get(symbol) {
            path.insert(0, symbol.to_string());
            let mut names = BTreeSet::new();
            expr.names(&mut names);
            for name in names {
                trace(
                    &name,
                    residual,
                    context,
                    path,
                    seen,
                    Some(CouplingReason::DefinedValueDependency(symbol.to_string())),
                    out,
                );
            }
            path.remove(0);
        } else if let Some(expr) = context.property_map.get(symbol) {
            path.insert(0, symbol.to_string());
            let mut names = BTreeSet::new();
            expr.names(&mut names);
            for name in names {
                trace(
                    &name,
                    residual,
                    context,
                    path,
                    seen,
                    Some(CouplingReason::PropertyDependency(symbol.to_string())),
                    out,
                );
            }
            path.remove(0);
        } else if let Some(expr) = context.constitutive_map.get(symbol) {
            path.insert(0, symbol.to_string());
            let mut names = BTreeSet::new();
            expr.names(&mut names);
            for name in names {
                trace(
                    &name,
                    residual,
                    context,
                    path,
                    seen,
                    Some(CouplingReason::ConstitutiveDependency(symbol.to_string())),
                    out,
                );
            }
            path.remove(0);
        }
        seen.remove(symbol);
    }

    let unknowns = field_names
        .iter()
        .map(|f| UnknownBlock {
            name: f.clone(),
            field: f.clone(),
        })
        .collect::<Vec<_>>();
    let mut residual_blocks = model
        .equations
        .iter()
        .map(|e| e.name.clone())
        .collect::<Vec<_>>();
    residual_blocks.extend(model.forms.iter().map(|f| f.name.clone()));
    residual_blocks.sort();
    residual_blocks.dedup();

    let mut edges = vec![];
    let context = TraceContext {
        field_names: &field_names,
        value_map: &value_map,
        property_map: &property_map,
        constitutive_map: &constitutive_map,
    };
    for equation in &model.equations {
        let mut names = BTreeSet::new();
        equation.lhs.names(&mut names);
        equation.rhs.names(&mut names);
        for name in names {
            trace(
                &name,
                &equation.name,
                &context,
                &mut Vec::new(),
                &mut BTreeSet::new(),
                None,
                &mut edges,
            );
        }
    }
    for form in &model.forms {
        for integral in &form.integrals {
            let mut names = BTreeSet::new();
            integral.integrand.names(&mut names);
            for name in names {
                trace(
                    &name,
                    &form.name,
                    &context,
                    &mut Vec::new(),
                    &mut BTreeSet::new(),
                    None,
                    &mut edges,
                );
            }
        }
    }
    // Conditions contribute to the residual block for their target field. Their region/value
    // dependencies are explicit interface/boundary coupling rather than invisible runtime state.
    for condition in model
        .boundary_conditions
        .iter()
        .chain(model.interface_conditions.iter())
    {
        let mut names = BTreeSet::new();
        condition.region.names(&mut names);
        condition.value.names(&mut names);
        for name in names {
            let before = edges.len();
            trace(
                &name,
                &condition.target,
                &context,
                &mut Vec::new(),
                &mut BTreeSet::new(),
                Some(CouplingReason::InterfaceTerm),
                &mut edges,
            );
            for edge in &mut edges[before..] {
                edge.reason = CouplingReason::InterfaceTerm;
            }
        }
    }

    edges.sort_by(|a, b| (&a.to, &a.from, &a.path).cmp(&(&b.to, &b.from, &b.path)));
    edges.dedup_by(|a, b| a.from == b.from && a.to == b.to && a.path == b.path);
    let mut derivatives = Vec::new();
    for residual in &residual_blocks {
        for unknown in &field_names {
            derivatives.push(BlockDerivative {
                residual: residual.clone(),
                unknown: unknown.clone(),
                structurally_nonzero: edges
                    .iter()
                    .any(|edge| &edge.to == residual && &edge.from == unknown),
            });
        }
    }
    CouplingGraph {
        unknowns,
        residual_blocks,
        edges,
        derivatives,
    }
}

// ---------------- time/state semantics ----------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeRole {
    Differential,
    Algebraic,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeField {
    pub field: String,
    pub role: TimeRole,
    pub initial: Option<Expr>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventSurface {
    pub name: String,
    pub expression: Expr,
    pub direction: EventDirection,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventDirection {
    Any,
    Rising,
    Falling,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryStateSchema {
    pub law: String,
    pub variables: Vec<StateVariable>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeStateSemantics {
    pub fields: Vec<TimeField>,
    pub events: Vec<EventSurface>,
    pub history: Vec<HistoryStateSchema>,
    pub dae_form: String,
}
impl TimeStateSemantics {
    pub fn from_model(model: &ScientificModel) -> Self {
        let initial: BTreeMap<_, _> = model
            .initial_conditions
            .iter()
            .map(|c| (c.target.clone(), c.value.clone()))
            .collect();
        let fields = model
            .fields
            .iter()
            .filter(|f| matches!(f.role, FieldRole::State | FieldRole::Unknown))
            .map(|f| TimeField {
                field: f.name.clone(),
                role: f.time_role.unwrap_or(TimeRole::Differential),
                initial: initial.get(&f.name).cloned(),
            })
            .collect();
        Self {
            fields,
            events: vec![],
            history: vec![],
            dae_form: "F(t, y, ydot, p) = 0".into(),
        }
    }
}

pub fn validate_quantities(
    model: &ScientificModel,
    registry: &UnitRegistry,
) -> Result<(), ScientificError> {
    for field in &model.fields {
        if let Some(nominal) = &field.nominal {
            canonicalize_authored_quantity(registry, nominal)?;
        }
        if let (Some(min), Some(max)) = (&field.physical_min, &field.physical_max) {
            let a = canonicalize_authored_quantity(registry, min)?;
            let b = canonicalize_authored_quantity(registry, max)?;
            if a.value_si() > b.value_si() {
                return Err(ScientificError::Quantity(format!(
                    "field `{}` has min > max",
                    field.name
                )));
            }
        }
    }
    Ok(())
}

/// Resolve authored unit symbols and unqualified kind names against a Quantitas registry, then
/// return the canonical quantity. The returned value is a Quantitas type, not a Scientia wrapper.
pub fn canonicalize_authored_quantity(
    registry: &UnitRegistry,
    literal: &QuantityLiteral,
) -> Result<Quantity, ScientificError> {
    let Some(unit) = registry
        .get(&literal.unit)
        .or_else(|| registry.by_symbol(literal.unit.as_str()))
    else {
        return registry
            .canonicalize(literal)
            .map_err(|error| ScientificError::Quantity(error.to_string()));
    };
    let kind = if unit.admitted_kinds.is_empty() || unit.admitted_kinds.contains(&literal.kind) {
        literal.kind.clone()
    } else {
        let suffix = literal.kind.as_str().rsplit(':').next().unwrap_or_default();
        let mut candidates = unit.admitted_kinds.iter().filter(|candidate| {
            candidate
                .as_str()
                .rsplit(':')
                .next()
                .is_some_and(|candidate_suffix| candidate_suffix == suffix)
        });
        match (candidates.next().cloned(), candidates.next()) {
            (Some(candidate), None) => candidate,
            _ => literal.kind.clone(),
        }
    };
    registry
        .canonicalize(&QuantityLiteral {
            value: literal.value,
            unit: unit.id.clone(),
            kind,
        })
        .map_err(|error| ScientificError::Quantity(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAT: &str = r#"
module examples.nonlinear_heat;
use physics.thermal.fourier;
model NonlinearHeat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { quantity = ThermodynamicTemperature; unit = K; nominal = 300 K; time_role = differential; };
  property rho = density(T);
  property cp = specific_heat(T);
  property k = thermal_conductivity(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { rho * cp * dt(T) - div(k * grad(T)) = Q; }
  initial { T = exact_T(0); }
  observable total_energy { integrate(rho * cp * T); }
}
"#;

    #[test]
    fn parses_structured_heat_source() {
        // The import is intentionally unresolved here: parsing and module resolution are separate phases.
        let m = parse_scientific_module(HEAT).unwrap();
        assert_eq!(m.name, "examples.nonlinear_heat");
        assert_eq!(m.models[0].properties.len(), 3);
        assert_eq!(m.models[0].equations.len(), 1);
        let graph = derive_coupling_graph(&m.models[0]);
        assert!(
            graph
                .edges
                .iter()
                .any(|e| e.from == "T" && e.to == "energy")
        );
    }

    // Symbolic differentiation of a property expression, and refusing (not zeroing) a
    // comparison operator, now project through `crate::projection` (contract C8) instead of
    // the old `algebra.rs` bridge on the parser-owned `Expr`; see
    // `projection::tests::differentiate_matches_finite_difference` and
    // `projection::tests::comparisons_are_refused_not_zeroed`.

    #[test]
    fn authored_symbols_resolve_through_quantitas() {
        let module = parse_scientific_module(HEAT).unwrap();
        let registry = UnitRegistry::si_bootstrap();
        validate_quantities(&module.models[0], &registry).unwrap();
        let quantity = canonicalize_authored_quantity(
            &registry,
            module.models[0].fields[0].nominal.as_ref().unwrap(),
        )
        .unwrap();
        assert_eq!(
            quantity.kind(),
            &QuantityKindId::thermodynamic_temperature()
        );
        assert_eq!(quantity.value_si(), 300.0);
    }
}
