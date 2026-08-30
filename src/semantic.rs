//! Typed semantic elaboration for `.res` source models.
//!
//! The parser-owned [`ScientificModule`] retains authored syntax. This module resolves every
//! scientific reference into stable arena identities and assigns shape, axes, Quantitas
//! dimensions/kinds, frame, and role metadata to every expression node.

use crate::id::span_independent_digest;
use crate::scientific::{
    BinaryOp, BoundaryConditionKind, CoordinateSystem, DerivativeContract, Expr, FieldDecl,
    FieldRole, InputBounds, Measure, OutOfValidityPolicy, PropertyDomain, PropertyLocality,
    ProviderDecl, ScientificModel as SourceModel, ScientificModule, SpaceSpec, UnaryOp, ValueDecl,
    ValueShape, canonicalize_authored_quantity,
};
use crate::source::{RelatedSpan, SourceDiagnostic, SourceSeverity, SourceSpan};
use quantitas::{Dimension, QuantityKindId, QuantityLiteral, UnitId, UnitRegistry};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

macro_rules! arena_id {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u32);

        impl $name {
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

arena_id!(DomainId);
arena_id!(RegionId);
arena_id!(SymbolId);
arena_id!(ExprId);
arena_id!(DeclarationId);
arena_id!(ProviderId);

impl RegionId {
    /// Reserved bit that marks a compiler-synthesized region id -- today, only the GX-A6
    /// implicit whole-boundary region `derive_variational_form_for` synthesizes when
    /// integration by parts needs an exterior boundary and the model declares none for that
    /// domain. As with [`SymbolId::GENERATED_BASE`], no real model reaches this magnitude.
    pub const GENERATED_BASE: u32 = 0x8000_0000;

    pub const fn is_generated(self) -> bool {
        self.0 >= Self::GENERATED_BASE
    }

    /// Build the deterministic implicit-natural-boundary region id for `domain`: the same
    /// domain always synthesizes the same region id, so repeated derivations agree.
    pub const fn generated_for(domain: DomainId) -> Self {
        Self(Self::GENERATED_BASE + domain.0)
    }

    /// Decode the domain a generated region id was built for, or `None` for a real
    /// (arena-indexed) region id.
    pub const fn generated_domain(self) -> Option<DomainId> {
        if self.is_generated() {
            Some(DomainId(self.0 - Self::GENERATED_BASE))
        } else {
            None
        }
    }
}

impl SymbolId {
    /// Reserved bit that marks a compiler-synthesized symbol id (see
    /// [`SymbolId::is_generated`]). Every id produced by [`SemanticModel`] elaboration is a
    /// dense arena index starting at zero, so no real model reaches this magnitude; a
    /// generated id is therefore never confusable with `SemanticModel::symbols[id]`.
    pub const GENERATED_BASE: u32 = 0x8000_0000;

    /// True when this id was synthesized by the compiler (for example, a derived form's
    /// generated test argument) rather than allocated for an authored field, property,
    /// source, or parameter declaration. `SemanticModel::symbols[id]` is invalid for a
    /// generated id; use the id only through the artifact (such as a derived form's
    /// generated test argument) that documents its meaning.
    pub const fn is_generated(self) -> bool {
        self.0 >= Self::GENERATED_BASE
    }

    /// Build the generated symbol id for a compiler-synthesized argument keyed on the
    /// declaration (for example an `equation`) it was derived from. Distinct declarations
    /// always receive distinct generated ids, even within the same model, so generated
    /// arguments from different derived forms never collide.
    pub const fn generated_for(declaration: DeclarationId) -> Self {
        Self(Self::GENERATED_BASE + declaration.0)
    }
}

// GX-A1 (C1.2) landed provider signatures/calls as "/4". GX-A2's exact-literal change
// (contract C8) adds `ExactLiteral` to `SemanticExprKind::Number` and bumps to "/5".
pub const SEMANTIC_SCHEMA: &str = "scientia-semantic/5";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticModule {
    pub schema: String,
    pub name: String,
    pub models: Vec<SemanticModel>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticCompilation {
    pub source: ScientificModule,
    pub semantic: SemanticModule,
    /// Non-fatal diagnostics from elaboration (today: only `RESOLVE_UNDECLARED_PROVIDER`).
    /// Excluded from `semantic`'s arena so it never perturbs `semantic_arena_digest`.
    #[serde(default)]
    pub advisories: Vec<SourceDiagnostic>,
}

/// Parse and elaborate source through the complete FC1 boundary.
pub fn compile_semantics(
    source: &str,
    registry: &UnitRegistry,
) -> Result<SemanticCompilation, Vec<SourceDiagnostic>> {
    let parsed = crate::scientific::parse_scientific_module_diagnostics(source)?;
    let (semantic, advisories) = elaborate_module(&parsed, registry)?;
    Ok(SemanticCompilation {
        source: parsed,
        semantic,
        advisories,
    })
}

/// One canonical, typed semantic arena for a source model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticModel {
    pub name: String,
    pub domains: Vec<SemanticDomain>,
    pub regions: Vec<SemanticRegion>,
    pub providers: Vec<SemanticProvider>,
    pub symbols: Vec<SemanticSymbol>,
    pub expressions: Arc<[SemanticExpr]>,
    pub declarations: Vec<SemanticDeclaration>,
    pub span: SourceSpan,
}

/// A `provider` signature (GX-A1, contract C1.2): `SemanticExprKind::ProviderCall` nodes
/// reference one of these by [`ProviderId`] rather than carrying an opaque function name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticProvider {
    pub id: ProviderId,
    pub name: String,
    pub inputs: Vec<SemanticProviderInput>,
    pub output: SemanticProviderOutput,
    pub locality: PropertyLocality,
    pub differentiability: DerivativeContract,
    pub domain: PropertyDomain,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticProviderInput {
    pub name: String,
    /// `None` is the `selector` pseudo-kind (a non-physical integer catalog selector); it
    /// never enters dimension analysis.
    pub quantity_kind: Option<QuantityKindId>,
    pub dimension: Option<Dimension>,
    pub shape: ValueShape,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticProviderOutput {
    pub quantity_kind: QuantityKindId,
    /// `None` when the declared quantity kind has no known dimension in this compiler's
    /// minimal kind table (see `known_kind_dimension`); arity/selector typing still applies.
    pub dimension: Option<Dimension>,
    pub shape: ValueShape,
    pub unit: Option<UnitId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticDomain {
    pub id: DomainId,
    pub name: String,
    pub spatial_dimension: u8,
    pub coordinates: CoordinateSystem,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    ExteriorFacet,
    InteriorFacet,
    Interface,
    Point,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticRegion {
    pub id: RegionId,
    pub name: String,
    pub kind: RegionKind,
    pub domain: Option<DomainId>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticRole {
    PhysicalField(FieldRole),
    Parameter,
    Constant,
    Source,
    Property,
    ConstitutiveLaw,
    Equation,
    Form,
    InitialCondition,
    BoundaryCondition,
    InterfaceCondition,
    Observable,
    Invariant,
    Verification,
    Literal,
    Intrinsic,
    /// The type role of a `SemanticExprKind::ProviderCall` expression, distinct from
    /// `Intrinsic` because its type comes from a declared `provider` signature.
    Provider,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticShape {
    Numeric(ValueShape),
    Boolean,
    String,
    Region,
    /// A declared scientific function whose output contract has not been supplied yet.
    Deferred,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Axis {
    pub position: u8,
    pub extent: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Frame {
    Neutral,
    Domain(DomainId),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticType {
    pub shape: SemanticShape,
    pub axes: Vec<Axis>,
    /// `None` is a deferred dimension constraint, not a second dimension representation.
    pub dimension: Option<Dimension>,
    pub quantity_kind: Option<QuantityKindId>,
    pub frame: Frame,
    pub role: SemanticRole,
}

impl SemanticType {
    pub(crate) fn numeric(
        shape: ValueShape,
        dimension: Option<Dimension>,
        frame: Frame,
        role: SemanticRole,
    ) -> Self {
        let axes = axes(&shape);
        Self {
            shape: SemanticShape::Numeric(shape),
            axes,
            dimension,
            quantity_kind: None,
            frame,
            role,
        }
    }

    fn deferred(role: SemanticRole) -> Self {
        Self {
            shape: SemanticShape::Deferred,
            axes: vec![],
            dimension: None,
            quantity_kind: None,
            frame: Frame::Neutral,
            role,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticSymbol {
    pub id: SymbolId,
    pub name: String,
    pub ty: SemanticType,
    pub domain: Option<DomainId>,
    pub space: Option<SpaceSpec>,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticExpr {
    pub id: ExprId,
    pub kind: SemanticExprKind,
    pub ty: SemanticType,
    pub span: SourceSpan,
}

/// The exact rational identity of a numeric literal (GX-A2, contract C8), computed from its
/// source lexeme rather than from the rounded `f64` carried alongside it. This is Scientia's
/// own type -- consumed directly by the Resolvent projection (`TermStore::exact_integer` /
/// `exact_decimal`) but never itself a Resolvent type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactLiteral {
    Integer(i128),
    /// `mantissa * 10^exponent`.
    Decimal {
        mantissa: i128,
        exponent: i16,
    },
}

/// `i128` has no native JSON representation (`serde_json::Value` supports only `i64`/`u64`/
/// `f64`), and `span_independent_digest` goes through `serde_json::to_value`, so `ExactLiteral`
/// serializes its `i128` fields as decimal strings rather than deriving `Serialize`/
/// `Deserialize` directly.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ExactLiteralWire {
    Integer { value: String },
    Decimal { mantissa: String, exponent: i16 },
}

impl Serialize for ExactLiteral {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            ExactLiteral::Integer(value) => ExactLiteralWire::Integer {
                value: value.to_string(),
            },
            ExactLiteral::Decimal { mantissa, exponent } => ExactLiteralWire::Decimal {
                mantissa: mantissa.to_string(),
                exponent: *exponent,
            },
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ExactLiteral {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match ExactLiteralWire::deserialize(deserializer)? {
            ExactLiteralWire::Integer { value } => {
                ExactLiteral::Integer(value.parse().map_err(serde::de::Error::custom)?)
            }
            ExactLiteralWire::Decimal { mantissa, exponent } => ExactLiteral::Decimal {
                mantissa: mantissa.parse().map_err(serde::de::Error::custom)?,
                exponent,
            },
        })
    }
}

impl ExactLiteral {
    /// Parse a lexer-produced numeric lexeme (ASCII digits, an optional `.`, and an optional
    /// `e`/`E` exponent; the lexer never includes a leading sign in a number token) into its
    /// exact decimal identity. Trailing fractional zeros are normalized away, so `0.10` and
    /// `0.1` parse to the same [`ExactLiteral`] while a spelling that changes the significant
    /// digits (or their count) does not.
    pub fn from_lexeme(lexeme: &str) -> Self {
        let (mantissa_text, exponent_text) = match lexeme.find(['e', 'E']) {
            Some(index) => (&lexeme[..index], &lexeme[index + 1..]),
            None => (lexeme, ""),
        };
        let exponent_bias: i64 = if exponent_text.is_empty() {
            0
        } else {
            exponent_text.parse().unwrap_or(0)
        };
        let (integer_part, fraction_part) = match mantissa_text.find('.') {
            Some(index) => (&mantissa_text[..index], &mantissa_text[index + 1..]),
            None => (mantissa_text, ""),
        };
        let mut digits = format!("{integer_part}{fraction_part}");
        if digits.is_empty() {
            digits = "0".into();
        }
        let mut exponent = exponent_bias - fraction_part.len() as i64;
        while digits.len() > 1 && digits.ends_with('0') {
            digits.pop();
            exponent += 1;
        }
        let mantissa: i128 = digits.parse().unwrap_or(0);
        if let Ok(shift) = u32::try_from(exponent)
            && shift <= 38
            && let Some(scale) = 10i128.checked_pow(shift)
            && let Some(value) = mantissa.checked_mul(scale)
        {
            return ExactLiteral::Integer(value);
        }
        ExactLiteral::Decimal {
            mantissa,
            exponent: exponent.clamp(i16::MIN as i64, i16::MAX as i64) as i16,
        }
    }

    /// Best-effort exact literal for a value with no authored source lexeme (a synthesized
    /// literal such as the `pi`/`π` intrinsic constant). Unlike [`Self::from_lexeme`] this is
    /// not a spelling identity -- only a real source lexeme carries one.
    pub fn from_value(value: f64) -> Self {
        Self::from_lexeme(&format!("{value:.17e}"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SemanticExprKind {
    Number {
        value: f64,
        unit: Option<UnitId>,
        /// Exact rational identity computed from the literal's source lexeme (GX-A2, contract
        /// C8), independent of `value`'s `f64` rounding. Included in every arena digest, so a
        /// model whose literals differ only in lexeme spelling (not exact value, per
        /// [`ExactLiteral::from_lexeme`]'s trailing-zero normalization) keeps identical digest
        /// identity.
        exact: ExactLiteral,
    },
    String {
        value: String,
    },
    Symbol {
        symbol: SymbolId,
    },
    Unary {
        op: UnaryOp,
        arg: ExprId,
    },
    Binary {
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
    },
    Call {
        function: String,
        args: Vec<ExprId>,
    },
    Differential {
        operator: DifferentialOperator,
        arg: ExprId,
    },
    Contraction {
        lhs: ExprId,
        rhs: ExprId,
        axes: Vec<AxisContraction>,
        conjugate_lhs: bool,
    },
    TensorTrace {
        value: ExprId,
        axes: AxisContraction,
    },
    FacetTrace {
        value: ExprId,
        side: TraceSide,
    },
    Jump {
        value: ExprId,
    },
    Average {
        value: ExprId,
    },
    Conjugate {
        value: ExprId,
    },
    NormalComponent {
        value: ExprId,
        side: TraceSide,
    },
    Index {
        value: ExprId,
        indices: Vec<ExprId>,
    },
    Vector {
        elements: Vec<ExprId>,
    },
    /// A call whose function name matches a declared `provider` signature (contract C1.2).
    /// Calls to undeclared, non-intrinsic functions keep `SemanticExprKind::Call` instead.
    ProviderCall {
        provider: ProviderId,
        args: Vec<ExprId>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentialOperator {
    Gradient,
    Divergence,
    Curl,
    RotatedGradient,
    SymmetricGradient,
    TimeDerivative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisContraction {
    pub lhs: u8,
    pub rhs: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceSide {
    Exterior,
    Minus,
    Plus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticDeclaration {
    pub id: DeclarationId,
    pub name: String,
    pub role: SemanticRole,
    pub symbol: Option<SymbolId>,
    pub domain: Option<DomainId>,
    pub kind: SemanticDeclarationKind,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SemanticDeclarationKind {
    Value {
        value: Option<ExprId>,
    },
    Property {
        value: ExprId,
    },
    ConstitutiveLaw {
        value: ExprId,
    },
    Equation {
        lhs: ExprId,
        rhs: ExprId,
    },
    Form {
        integrals: Vec<SemanticIntegral>,
    },
    InitialCondition {
        target: Option<SymbolId>,
        value: ExprId,
    },
    BoundaryCondition {
        region: RegionId,
        selector: ExprId,
        target: Option<SymbolId>,
        condition: BoundaryConditionKind,
        value: ExprId,
    },
    Observable {
        value: ExprId,
    },
    Invariant {
        value: ExprId,
    },
    Verification {
        arguments: BTreeMap<String, ExprId>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SemanticMeasure {
    Cell { domain: DomainId },
    ExteriorFacet { region: RegionId },
    InteriorFacet { region: RegionId },
    Interface { region: RegionId },
    Point { region: RegionId },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticIntegral {
    pub measure: SemanticMeasure,
    pub integrand: ExprId,
    pub span: SourceSpan,
}

/// Elaborate every model in `module`. The `Ok` side carries advisory (non-`Error`-severity)
/// diagnostics alongside the model: today the only advisory is `RESOLVE_UNDECLARED_PROVIDER`,
/// which lets every corpus model keep elaborating even with unregistered provider calls
/// (contract C1.2). `Err` is returned only when at least one `Error`-severity diagnostic
/// remains, and (as before) carries the complete diagnostic list, advisories included.
pub fn elaborate_module(
    module: &ScientificModule,
    registry: &UnitRegistry,
) -> Result<(SemanticModule, Vec<SourceDiagnostic>), Vec<SourceDiagnostic>> {
    let mut diagnostics = vec![];
    let models = module
        .models
        .iter()
        .map(|model| Elaborator::new(model, registry).run(&mut diagnostics))
        .collect();
    diagnostics.sort_by(|left, right| {
        (left.span.start, left.span.end, &left.code, &left.message).cmp(&(
            right.span.start,
            right.span.end,
            &right.code,
            &right.message,
        ))
    });
    diagnostics.dedup();
    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == SourceSeverity::Error)
    {
        Err(diagnostics)
    } else {
        Ok((
            SemanticModule {
                schema: SEMANTIC_SCHEMA.into(),
                name: module.name.clone(),
                models,
                span: module.span,
            },
            diagnostics,
        ))
    }
}

pub fn semantic_arena_digest(module: &SemanticModule) -> String {
    span_independent_digest(module).hex
}

struct Elaborator<'a> {
    source: &'a SourceModel,
    registry: &'a UnitRegistry,
    domains: Vec<SemanticDomain>,
    domain_names: BTreeMap<String, DomainId>,
    regions: Vec<SemanticRegion>,
    region_names: BTreeMap<(RegionKind, String), RegionId>,
    providers: Vec<SemanticProvider>,
    provider_names: BTreeMap<String, ProviderId>,
    symbols: Vec<SemanticSymbol>,
    symbol_names: BTreeMap<String, SymbolId>,
    expressions: Vec<SemanticExpr>,
    declarations: Vec<SemanticDeclaration>,
}

impl<'a> Elaborator<'a> {
    fn new(source: &'a SourceModel, registry: &'a UnitRegistry) -> Self {
        Self {
            source,
            registry,
            domains: vec![],
            domain_names: BTreeMap::new(),
            regions: vec![],
            region_names: BTreeMap::new(),
            providers: vec![],
            provider_names: BTreeMap::new(),
            symbols: vec![],
            symbol_names: BTreeMap::new(),
            expressions: vec![],
            declarations: vec![],
        }
    }

    fn run(mut self, diagnostics: &mut Vec<SourceDiagnostic>) -> SemanticModel {
        self.declare_domains(diagnostics);
        self.declare_providers(diagnostics);
        self.declare_value_symbols(diagnostics);
        self.declare_regions();
        self.elaborate_definitions(diagnostics);
        SemanticModel {
            name: self.source.name.clone(),
            domains: self.domains,
            regions: self.regions,
            providers: self.providers,
            symbols: self.symbols,
            expressions: self.expressions.into(),
            declarations: self.declarations,
            span: self.source.span,
        }
    }

    fn declare_domains(&mut self, diagnostics: &mut Vec<SourceDiagnostic>) {
        for domain in &self.source.domains {
            if let Some(previous) = self.domain_names.get(&domain.name).copied() {
                diagnostics.push(duplicate(
                    "domain",
                    &domain.name,
                    domain.span,
                    self.domains[previous.index()].span,
                ));
                continue;
            }
            if domain.dimension > 3 {
                diagnostics.push(error(
                    "RESOLVE_INVALID_DOMAIN_DIMENSION",
                    format!(
                        "domain `{}` has unsupported spatial dimension {}",
                        domain.name, domain.dimension
                    ),
                    domain.span,
                ));
            }
            if matches!(
                domain.coordinates,
                CoordinateSystem::Cylindrical | CoordinateSystem::Spherical
            ) && domain.dimension != 3
            {
                diagnostics.push(error(
                    "RESOLVE_FRAME_DIMENSION_MISMATCH",
                    format!(
                        "{:?} coordinates require a three-dimensional domain",
                        domain.coordinates
                    ),
                    domain.coordinates_span,
                ));
            }
            let id = DomainId(self.domains.len() as u32);
            self.domain_names.insert(domain.name.clone(), id);
            self.domains.push(SemanticDomain {
                id,
                name: domain.name.clone(),
                spatial_dimension: domain.dimension,
                coordinates: domain.coordinates.clone(),
                span: domain.span,
            });
        }
    }

    fn declare_providers(&mut self, diagnostics: &mut Vec<SourceDiagnostic>) {
        for provider in &self.source.providers {
            if let Some(previous) = self.provider_names.get(&provider.name).copied() {
                diagnostics.push(duplicate(
                    "provider",
                    &provider.name,
                    provider.span,
                    self.providers[previous.index()].span,
                ));
                continue;
            }
            if is_intrinsic_call(&provider.name) {
                diagnostics.push(error(
                    "RESOLVE_DUPLICATE_NAME",
                    format!(
                        "provider `{}` cannot reuse the reserved intrinsic function name",
                        provider.name
                    ),
                    provider.span,
                ));
                continue;
            }
            let inputs = provider
                .inputs
                .iter()
                .map(|input| match &input.kind {
                    None => SemanticProviderInput {
                        name: input.name.clone(),
                        quantity_kind: None,
                        dimension: None,
                        shape: ValueShape::Scalar,
                        span: input.span,
                    },
                    Some(kind_name) => {
                        let kind = QuantityKindId::new(kind_name.clone());
                        let dimension = known_kind_dimension(&kind);
                        SemanticProviderInput {
                            name: input.name.clone(),
                            quantity_kind: Some(kind),
                            dimension,
                            shape: ValueShape::Scalar,
                            span: input.span,
                        }
                    }
                })
                .collect::<Vec<_>>();
            let declared_output_kind = QuantityKindId::new(provider.output_kind.clone());
            let (output_dimension, canonical_output_kind) = self.declared_quantity_type(
                Some(&declared_output_kind),
                Some(provider.output_kind_span),
                provider.unit.as_ref(),
                provider.unit_span,
                diagnostics,
            );
            let output = SemanticProviderOutput {
                quantity_kind: canonical_output_kind.unwrap_or(declared_output_kind),
                dimension: output_dimension,
                shape: provider.shape.clone(),
                unit: provider.unit.clone(),
            };
            let domain = self.provider_domain(provider, &inputs, diagnostics);
            let id = ProviderId(self.providers.len() as u32);
            self.provider_names.insert(provider.name.clone(), id);
            self.providers.push(SemanticProvider {
                id,
                name: provider.name.clone(),
                inputs,
                output,
                locality: provider.locality.clone(),
                differentiability: provider.differentiability.clone(),
                domain,
                span: provider.span,
            });
        }
    }

    /// Canonicalize each `domain { input in [lo, hi]; }` bound into an SI-valued `InputBounds`,
    /// reusing the property vocabulary's `PropertyDomain`/`InputBounds` (contract C1.1/C1.2)
    /// rather than a parallel type. An unresolvable unit is `RESOLVE_UNKNOWN_UNIT`, matching
    /// how a field's own `unit`/`nominal` literals are already diagnosed.
    fn provider_domain(
        &self,
        provider: &ProviderDecl,
        inputs: &[SemanticProviderInput],
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> PropertyDomain {
        let mut validity_bounds = vec![];
        for bound in &provider.domain {
            let kind = inputs
                .iter()
                .find(|input| input.name == bound.input)
                .and_then(|input| input.quantity_kind.clone());
            let mut min = bound.min.clone();
            let mut max = bound.max.clone();
            if let Some(kind) = kind {
                min.kind = kind.clone();
                max.kind = kind;
            }
            match (
                canonicalize_authored_quantity(self.registry, &min),
                canonicalize_authored_quantity(self.registry, &max),
            ) {
                (Ok(min), Ok(max)) => validity_bounds.push(InputBounds {
                    input: bound.input.clone(),
                    min: Some(min.value_si()),
                    max: Some(max.value_si()),
                }),
                (min_result, max_result) => {
                    for result in [min_result, max_result] {
                        if let Err(error_value) = result {
                            diagnostics.push(error(
                                "RESOLVE_UNKNOWN_UNIT",
                                error_value.to_string(),
                                bound.span,
                            ));
                        }
                    }
                }
            }
        }
        PropertyDomain {
            physical_bounds: vec![],
            validity_bounds,
            phase_constraints: vec![],
            composition_constraints: vec![],
            assumptions: vec![],
            out_of_validity: OutOfValidityPolicy::Error,
        }
    }

    fn declare_value_symbols(&mut self, diagnostics: &mut Vec<SourceDiagnostic>) {
        for field in &self.source.fields {
            let domain = self.resolve_domain(&field.domain, field.domain_span, diagnostics);
            let mut ty = self.field_type(field, domain, diagnostics);
            ty.role = SemanticRole::PhysicalField(field.role.clone());
            self.insert_symbol(
                &field.name,
                ty,
                domain,
                Some(field.space.clone()),
                field.span,
                diagnostics,
            );
        }
        for value in &self.source.parameters {
            self.declare_value(value, SemanticRole::Parameter, diagnostics);
        }
        for value in &self.source.constants {
            self.declare_value(value, SemanticRole::Constant, diagnostics);
        }
        for value in &self.source.sources {
            self.declare_value(value, SemanticRole::Source, diagnostics);
        }
        let properties = self
            .source
            .properties
            .iter()
            .map(|item| (&item.name, item.span));
        for (name, span) in properties {
            self.insert_symbol(
                name,
                SemanticType::deferred(SemanticRole::Property),
                None,
                None,
                span,
                diagnostics,
            );
        }
        let laws = self
            .source
            .constitutive_laws
            .iter()
            .map(|item| (&item.name, item.span));
        for (name, span) in laws {
            self.insert_symbol(
                name,
                SemanticType::deferred(SemanticRole::ConstitutiveLaw),
                None,
                None,
                span,
                diagnostics,
            );
        }
        let equations = self
            .source
            .equations
            .iter()
            .map(|item| (&item.name, item.span));
        for (name, span) in equations {
            self.insert_symbol(
                name,
                SemanticType::deferred(SemanticRole::Equation),
                None,
                None,
                span,
                diagnostics,
            );
        }
    }

    fn declare_regions(&mut self) {
        for condition in &self.source.boundary_conditions {
            let name = region_name(&condition.region).unwrap_or(&condition.name);
            let domain = self
                .symbol_names
                .get(&condition.target)
                .and_then(|symbol| self.symbols[symbol.index()].domain);
            self.intern_region(
                RegionKind::ExteriorFacet,
                name,
                domain,
                condition.region.span(),
            );
        }
        for condition in &self.source.interface_conditions {
            let name = region_name(&condition.region).unwrap_or(&condition.name);
            self.intern_region(RegionKind::Interface, name, None, condition.region.span());
        }
        for form in &self.source.forms {
            // GX-A6: a region referenced only through a form's own facet/interface/point
            // measures has no boundary-condition target to infer a domain from, but it still
            // has a natural one -- the same form's `cell(<Domain>)` measure, when the form
            // declares one.
            let form_domain = form
                .integrals
                .iter()
                .find_map(|integral| match &integral.measure {
                    Measure::Cell(name) => self.domain_names.get(name).copied(),
                    _ => None,
                });
            for integral in &form.integrals {
                match &integral.measure {
                    Measure::Cell(_) => {}
                    Measure::Boundary(name) => {
                        self.intern_region(
                            RegionKind::ExteriorFacet,
                            name,
                            form_domain,
                            integral.target_span,
                        );
                    }
                    Measure::InteriorFacet(name) => {
                        self.intern_region(
                            RegionKind::InteriorFacet,
                            name,
                            form_domain,
                            integral.target_span,
                        );
                    }
                    Measure::Interface(name) => {
                        self.intern_region(
                            RegionKind::Interface,
                            name,
                            form_domain,
                            integral.target_span,
                        );
                    }
                    Measure::Point(name) => {
                        self.intern_region(
                            RegionKind::Point,
                            name,
                            form_domain,
                            integral.target_span,
                        );
                    }
                }
            }
        }
    }

    fn intern_region(
        &mut self,
        kind: RegionKind,
        name: &str,
        domain: Option<DomainId>,
        span: SourceSpan,
    ) -> RegionId {
        let key = (kind.clone(), name.to_owned());
        if let Some(id) = self.region_names.get(&key).copied() {
            if self.regions[id.index()].domain.is_none() {
                self.regions[id.index()].domain = domain;
            }
            return id;
        }
        let id = RegionId(self.regions.len() as u32);
        self.region_names.insert(key, id);
        self.regions.push(SemanticRegion {
            id,
            name: name.to_owned(),
            kind,
            domain,
            span,
        });
        id
    }

    fn region_id(&self, kind: RegionKind, name: &str) -> RegionId {
        self.region_names[&(kind, name.to_owned())]
    }

    fn declare_value(
        &mut self,
        value: &ValueDecl,
        role: SemanticRole,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) {
        let (dimension, kind) = self.declared_quantity_type(
            value.quantity_kind.as_ref(),
            value.quantity_kind_span,
            value.unit.as_ref(),
            value.unit_span,
            diagnostics,
        );
        let mut ty = if matches!(role, SemanticRole::Source) && dimension.is_none() {
            SemanticType::deferred(role)
        } else {
            SemanticType::numeric(ValueShape::Scalar, dimension, Frame::Neutral, role)
        };
        ty.quantity_kind = kind;
        self.insert_symbol(&value.name, ty, None, None, value.span, diagnostics);
    }

    fn field_type(
        &self,
        field: &FieldDecl,
        domain: Option<DomainId>,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> SemanticType {
        validate_shape(&field.shape, field.span, diagnostics);
        let (dimension, kind) = self.declared_quantity_type(
            field.quantity_kind.as_ref(),
            field.quantity_kind_span,
            field.unit.as_ref(),
            field.unit_span,
            diagnostics,
        );
        for literal in [&field.nominal, &field.physical_min, &field.physical_max]
            .into_iter()
            .flatten()
        {
            self.validate_literal(
                literal,
                field.span,
                field.quantity_kind.is_some(),
                diagnostics,
            );
            if let Some(expected) = dimension
                && self
                    .literal_dimension(literal)
                    .is_some_and(|actual| actual != expected)
            {
                diagnostics.push(error(
                    "TYPE_DIMENSION_MISMATCH",
                    format!(
                        "quantity literal for field `{}` has an incompatible dimension",
                        field.name
                    ),
                    field.span,
                ));
            }
        }
        if let (Some(min), Some(max)) = (&field.physical_min, &field.physical_max)
            && let (Ok(min), Ok(max)) = (
                canonicalize_authored_quantity(self.registry, min),
                canonicalize_authored_quantity(self.registry, max),
            )
            && min.value_si() > max.value_si()
        {
            diagnostics.push(error(
                "TYPE_INVALID_RANGE",
                format!("field `{}` has min greater than max", field.name),
                field.span,
            ));
        }
        let mut ty = SemanticType::numeric(
            field.shape.clone(),
            dimension,
            domain.map_or(Frame::Neutral, Frame::Domain),
            SemanticRole::PhysicalField(field.role.clone()),
        );
        ty.quantity_kind = kind;
        ty
    }

    fn declared_quantity_type(
        &self,
        kind: Option<&QuantityKindId>,
        kind_span: Option<SourceSpan>,
        unit: Option<&UnitId>,
        unit_span: Option<SourceSpan>,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> (Option<Dimension>, Option<QuantityKindId>) {
        let definition = unit.and_then(|unit| {
            self.registry
                .get(unit)
                .or_else(|| self.registry.by_symbol(unit.as_str()))
                .or_else(|| {
                    diagnostics.push(error(
                        "RESOLVE_UNKNOWN_UNIT",
                        format!("unknown unit `{unit}`"),
                        unit_span.unwrap_or_default(),
                    ));
                    None
                })
        });
        let mut canonical_kind = kind.cloned();
        if let (Some(kind), Some(definition)) = (kind, definition) {
            if known_kind_dimension(kind).is_some_and(|expected| expected != definition.dimension) {
                diagnostics.push(error(
                    "RESOLVE_UNIT_KIND_MISMATCH",
                    format!(
                        "unit `{}` has dimension `{}`, which is incompatible with quantity kind `{kind}`",
                        definition.symbol, definition.dimension
                    ),
                    kind_span.or(unit_span).unwrap_or_default(),
                ));
            }
            let literal = QuantityLiteral {
                value: 0.0,
                unit: definition.id.clone(),
                kind: kind.clone(),
            };
            match canonicalize_authored_quantity(self.registry, &literal) {
                Ok(quantity) => canonical_kind = Some(quantity.kind().clone()),
                Err(error_value) => diagnostics.push(error(
                    "RESOLVE_UNIT_KIND_MISMATCH",
                    error_value.to_string(),
                    kind_span.or(unit_span).unwrap_or_default(),
                )),
            }
        }
        let dimension = definition
            .map(|definition| definition.dimension)
            .or_else(|| kind.and_then(known_kind_dimension));
        (dimension, canonical_kind)
    }

    fn validate_literal(
        &self,
        literal: &QuantityLiteral,
        span: SourceSpan,
        check_kind: bool,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) {
        let definition = self
            .registry
            .get(&literal.unit)
            .or_else(|| self.registry.by_symbol(literal.unit.as_str()));
        let Some(definition) = definition else {
            diagnostics.push(error(
                "RESOLVE_UNKNOWN_UNIT",
                format!("unknown unit `{}`", literal.unit),
                span,
            ));
            return;
        };
        if check_kind {
            let mut literal = literal.clone();
            literal.unit = definition.id.clone();
            if let Err(error_value) = canonicalize_authored_quantity(self.registry, &literal) {
                diagnostics.push(error(
                    "RESOLVE_UNIT_KIND_MISMATCH",
                    error_value.to_string(),
                    span,
                ));
            }
        }
    }

    fn literal_dimension(&self, literal: &QuantityLiteral) -> Option<Dimension> {
        self.registry
            .get(&literal.unit)
            .or_else(|| self.registry.by_symbol(literal.unit.as_str()))
            .map(|definition| definition.dimension)
    }

    fn insert_symbol(
        &mut self,
        name: &str,
        ty: SemanticType,
        domain: Option<DomainId>,
        space: Option<SpaceSpec>,
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> Option<SymbolId> {
        if let Some(previous) = self.symbol_names.get(name).copied() {
            diagnostics.push(duplicate(
                "symbol",
                name,
                span,
                self.symbols[previous.index()].span,
            ));
            return None;
        }
        let id = SymbolId(self.symbols.len() as u32);
        self.symbol_names.insert(name.into(), id);
        self.symbols.push(SemanticSymbol {
            id,
            name: name.into(),
            ty,
            domain,
            space,
            span,
        });
        Some(id)
    }

    fn resolve_domain(
        &self,
        name: &str,
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> Option<DomainId> {
        self.domain_names.get(name).copied().or_else(|| {
            diagnostics.push(error(
                "RESOLVE_UNKNOWN_DOMAIN",
                format!("unknown domain `{name}`"),
                span,
            ));
            None
        })
    }

    fn elaborate_definitions(&mut self, diagnostics: &mut Vec<SourceDiagnostic>) {
        for value in &self.source.parameters {
            self.elaborate_value(value, SemanticRole::Parameter, diagnostics);
        }
        for value in &self.source.constants {
            self.elaborate_value(value, SemanticRole::Constant, diagnostics);
        }
        for value in &self.source.sources {
            self.elaborate_value(value, SemanticRole::Source, diagnostics);
        }
        for property in &self.source.properties {
            let expr = self.elaborate_expr(&property.value, diagnostics);
            self.update_deferred_symbol(&property.name, expr);
            self.push_declaration(
                &property.name,
                SemanticRole::Property,
                None,
                SemanticDeclarationKind::Property { value: expr },
                property.span,
            );
        }
        for law in &self.source.constitutive_laws {
            let expr = self.elaborate_expr(&law.law, diagnostics);
            self.update_deferred_symbol(&law.name, expr);
            self.push_declaration(
                &law.name,
                SemanticRole::ConstitutiveLaw,
                None,
                SemanticDeclarationKind::ConstitutiveLaw { value: expr },
                law.span,
            );
        }
        for equation in &self.source.equations {
            let domain = equation.domain.as_ref().and_then(|name| {
                self.resolve_domain(
                    name,
                    equation.domain_span.unwrap_or(equation.span),
                    diagnostics,
                )
            });
            let lhs = self.elaborate_expr(&equation.lhs, diagnostics);
            let rhs = self.elaborate_expr(&equation.rhs, diagnostics);
            self.require_compatible(lhs, rhs, equation.span, diagnostics);
            self.push_declaration(
                &equation.name,
                SemanticRole::Equation,
                domain,
                SemanticDeclarationKind::Equation { lhs, rhs },
                equation.span,
            );
        }
        for form in &self.source.forms {
            let mut integrals = vec![];
            for integral in &form.integrals {
                let measure = match &integral.measure {
                    Measure::Cell(name) => self
                        .resolve_domain(name, integral.target_span, diagnostics)
                        .map(|domain| (SemanticMeasure::Cell { domain }, Some(domain))),
                    Measure::Boundary(name) => Some((
                        SemanticMeasure::ExteriorFacet {
                            region: self.region_id(RegionKind::ExteriorFacet, name),
                        },
                        None,
                    )),
                    Measure::InteriorFacet(name) => Some((
                        SemanticMeasure::InteriorFacet {
                            region: self.region_id(RegionKind::InteriorFacet, name),
                        },
                        None,
                    )),
                    Measure::Interface(name) => Some((
                        SemanticMeasure::Interface {
                            region: self.region_id(RegionKind::Interface, name),
                        },
                        None,
                    )),
                    Measure::Point(name) => Some((
                        SemanticMeasure::Point {
                            region: self.region_id(RegionKind::Point, name),
                        },
                        None,
                    )),
                };
                let expression = self.elaborate_expr(&integral.integrand, diagnostics);
                if let Some((_, Some(domain))) = &measure {
                    self.require_frame(expression, *domain, integral.integrand.span(), diagnostics);
                }
                if let Some((measure, _)) = measure {
                    integrals.push(SemanticIntegral {
                        measure,
                        integrand: expression,
                        span: integral.span,
                    });
                }
            }
            self.push_declaration(
                &form.name,
                SemanticRole::Form,
                None,
                SemanticDeclarationKind::Form { integrals },
                form.span,
            );
        }
        for condition in &self.source.initial_conditions {
            let target = self.resolve_target(&condition.target, condition.span, true, diagnostics);
            let value = self.elaborate_expr(&condition.value, diagnostics);
            if let Some(target) = target {
                self.require_compatible_symbol(target, value, condition.span, diagnostics);
            }
            self.push_declaration(
                &condition.target,
                SemanticRole::InitialCondition,
                None,
                SemanticDeclarationKind::InitialCondition { target, value },
                condition.span,
            );
        }
        for condition in &self.source.boundary_conditions {
            self.elaborate_boundary(condition, SemanticRole::BoundaryCondition, diagnostics);
        }
        for condition in &self.source.interface_conditions {
            self.elaborate_boundary(condition, SemanticRole::InterfaceCondition, diagnostics);
        }
        for observable in &self.source.observables {
            let expression = self.elaborate_expr(&observable.value, diagnostics);
            self.push_declaration(
                &observable.name,
                SemanticRole::Observable,
                None,
                SemanticDeclarationKind::Observable { value: expression },
                observable.span,
            );
        }
        for invariant in &self.source.invariants {
            let expression = self.elaborate_expr(&invariant.value, diagnostics);
            self.push_declaration(
                &invariant.name,
                SemanticRole::Invariant,
                None,
                SemanticDeclarationKind::Invariant { value: expression },
                invariant.span,
            );
        }
        for verification in &self.source.verifications {
            let arguments = verification
                .args
                .iter()
                .map(|(name, expression)| {
                    (name.clone(), self.elaborate_expr(expression, diagnostics))
                })
                .collect();
            self.push_declaration(
                &verification.name,
                SemanticRole::Verification,
                None,
                SemanticDeclarationKind::Verification { arguments },
                verification.span,
            );
        }
    }

    fn elaborate_value(
        &mut self,
        value: &ValueDecl,
        role: SemanticRole,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) {
        let expression = value.value.as_ref().map(|expression| {
            let id = self.elaborate_expr(expression, diagnostics);
            if let Some(symbol) = self.symbol_names.get(&value.name).copied() {
                self.require_compatible_symbol(symbol, id, expression.span(), diagnostics);
            }
            id
        });
        self.push_declaration(
            &value.name,
            role,
            None,
            SemanticDeclarationKind::Value { value: expression },
            value.span,
        );
    }

    fn elaborate_boundary(
        &mut self,
        condition: &crate::scientific::BoundaryConditionDecl,
        role: SemanticRole,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) {
        let region = self.elaborate_expr(&condition.region, diagnostics);
        if !matches!(
            self.expressions[region.index()].ty.shape,
            SemanticShape::Region | SemanticShape::Deferred
        ) {
            diagnostics.push(error(
                "TYPE_REGION_REQUIRED",
                "boundary/interface selector must produce a region",
                condition.region.span(),
            ));
        }
        let target =
            self.resolve_target(&condition.target, condition.target_span, false, diagnostics);
        let value = self.elaborate_expr(&condition.value, diagnostics);
        if let Some(target) = target {
            self.require_compatible_symbol(target, value, condition.value.span(), diagnostics);
        }
        if matches!(role, SemanticRole::InterfaceCondition)
            && condition.kind != BoundaryConditionKind::Interface
        {
            diagnostics.push(error(
                "TYPE_ROLE_MISMATCH",
                "interface declaration must use an interface condition",
                condition.span,
            ));
        }
        let region_name = region_name(&condition.region).unwrap_or(&condition.name);
        let region_kind = if matches!(role, SemanticRole::InterfaceCondition) {
            RegionKind::Interface
        } else {
            RegionKind::ExteriorFacet
        };
        let region_id = self.region_id(region_kind, region_name);
        self.push_declaration(
            &condition.name,
            role,
            None,
            SemanticDeclarationKind::BoundaryCondition {
                region: region_id,
                selector: region,
                target,
                condition: condition.kind.clone(),
                value,
            },
            condition.span,
        );
    }

    fn resolve_target(
        &self,
        name: &str,
        span: SourceSpan,
        initial: bool,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> Option<SymbolId> {
        let Some(id) = self.symbol_names.get(name).copied() else {
            diagnostics.push(error(
                "RESOLVE_UNKNOWN_NAME",
                format!("unknown target `{name}`"),
                span,
            ));
            return None;
        };
        let role = &self.symbols[id.index()].ty.role;
        let allowed = if initial {
            matches!(role, SemanticRole::PhysicalField(FieldRole::State))
        } else {
            matches!(
                role,
                SemanticRole::PhysicalField(_) | SemanticRole::ConstitutiveLaw
            )
        };
        if !allowed {
            diagnostics.push(error(
                "TYPE_ROLE_MISMATCH",
                if initial {
                    format!("initial condition target `{name}` is not a state field")
                } else {
                    format!("boundary condition target `{name}` is not a field")
                },
                span,
            ));
        }
        Some(id)
    }

    fn elaborate_expr(
        &mut self,
        expression: &Expr,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> ExprId {
        let (kind, ty) = match expression {
            Expr::Number {
                value,
                lexeme,
                unit,
                span,
            } => {
                let (unit_id, dimension) = if let Some(authored) = unit {
                    match self
                        .registry
                        .by_symbol(authored)
                        .or_else(|| self.registry.get(&UnitId::new(authored)))
                    {
                        Some(definition) => {
                            (Some(definition.id.clone()), Some(definition.dimension))
                        }
                        None => {
                            diagnostics.push(error(
                                "RESOLVE_UNKNOWN_UNIT",
                                format!("unknown unit `{authored}`"),
                                *span,
                            ));
                            (Some(UnitId::new(authored)), None)
                        }
                    }
                } else {
                    (None, Some(Dimension::DIMENSIONLESS))
                };
                (
                    SemanticExprKind::Number {
                        value: *value,
                        unit: unit_id,
                        exact: ExactLiteral::from_lexeme(lexeme),
                    },
                    SemanticType::numeric(
                        ValueShape::Scalar,
                        dimension,
                        Frame::Neutral,
                        SemanticRole::Literal,
                    ),
                )
            }
            Expr::String { value, .. } => (
                SemanticExprKind::String {
                    value: value.clone(),
                },
                SemanticType {
                    shape: SemanticShape::String,
                    axes: vec![],
                    dimension: None,
                    quantity_kind: None,
                    frame: Frame::Neutral,
                    role: SemanticRole::Literal,
                },
            ),
            Expr::Name { name, span } => {
                if name == "t" {
                    (
                        SemanticExprKind::Call {
                            function: "time".into(),
                            args: vec![],
                        },
                        SemanticType::numeric(
                            ValueShape::Scalar,
                            Some(Dimension::TIME),
                            Frame::Neutral,
                            SemanticRole::Intrinsic,
                        ),
                    )
                } else if name == "pi" || name == "π" {
                    (
                        SemanticExprKind::Number {
                            value: std::f64::consts::PI,
                            unit: None,
                            exact: ExactLiteral::from_value(std::f64::consts::PI),
                        },
                        SemanticType::numeric(
                            ValueShape::Scalar,
                            Some(Dimension::DIMENSIONLESS),
                            Frame::Neutral,
                            SemanticRole::Literal,
                        ),
                    )
                } else if let Some(symbol) = self.symbol_names.get(name).copied() {
                    (
                        SemanticExprKind::Symbol { symbol },
                        self.symbols[symbol.index()].ty.clone(),
                    )
                } else {
                    diagnostics.push(error(
                        "RESOLVE_UNKNOWN_NAME",
                        format!("unknown name `{name}`"),
                        *span,
                    ));
                    (
                        SemanticExprKind::Call {
                            function: "unresolved".into(),
                            args: vec![],
                        },
                        SemanticType::deferred(SemanticRole::Intrinsic),
                    )
                }
            }
            Expr::Unary { op, arg, .. } => {
                let arg = self.elaborate_expr(arg, diagnostics);
                let ty = self.expressions[arg.index()].ty.clone();
                if !is_numeric_or_deferred(&ty.shape) {
                    diagnostics.push(error(
                        "TYPE_NUMERIC_REQUIRED",
                        "unary negation requires a numeric operand",
                        expression.span(),
                    ));
                }
                (SemanticExprKind::Unary { op: *op, arg }, ty)
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let lhs = self.elaborate_expr(lhs, diagnostics);
                let rhs = self.elaborate_expr(rhs, diagnostics);
                let ty = self.binary_type(*op, lhs, rhs, expression.span(), diagnostics);
                (SemanticExprKind::Binary { op: *op, lhs, rhs }, ty)
            }
            Expr::Call { function, args, .. } => {
                let args: Vec<_> = args
                    .iter()
                    .map(|arg| self.elaborate_expr(arg, diagnostics))
                    .collect();
                if is_intrinsic_call(function) {
                    let ty = self.call_type(function, &args, expression.span(), diagnostics);
                    (self.call_kind(function, args), ty)
                } else if let Some(&provider) = self.provider_names.get(function) {
                    self.provider_call(provider, args, expression.span(), diagnostics)
                } else {
                    diagnostics.push(advisory(
                        "RESOLVE_UNDECLARED_PROVIDER",
                        format!(
                            "call to undeclared provider `{function}`; declare \
                             `provider {function}(...) -> Kind;` to type it"
                        ),
                        expression.span(),
                    ));
                    let ty = self.call_type(function, &args, expression.span(), diagnostics);
                    (self.call_kind(function, args), ty)
                }
            }
            Expr::Index { value, indices, .. } => {
                let value = self.elaborate_expr(value, diagnostics);
                let indices: Vec<_> = indices
                    .iter()
                    .map(|index| self.elaborate_expr(index, diagnostics))
                    .collect();
                let ty = self.index_type(value, &indices, expression.span(), diagnostics);
                (SemanticExprKind::Index { value, indices }, ty)
            }
            Expr::Vector { elements, .. } => {
                let elements: Vec<_> = elements
                    .iter()
                    .map(|item| self.elaborate_expr(item, diagnostics))
                    .collect();
                let ty = self.vector_type(&elements, expression.span(), diagnostics);
                (SemanticExprKind::Vector { elements }, ty)
            }
        };
        let id = ExprId(self.expressions.len() as u32);
        self.expressions.push(SemanticExpr {
            id,
            kind,
            ty,
            span: expression.span(),
        });
        id
    }

    fn binary_type(
        &self,
        op: BinaryOp,
        lhs: ExprId,
        rhs: ExprId,
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> SemanticType {
        let left = &self.expressions[lhs.index()].ty;
        let right = &self.expressions[rhs.index()].ty;
        match op {
            BinaryOp::Add | BinaryOp::Sub => {
                if self.is_zero(lhs) {
                    let mut ty = right.clone();
                    ty.role = SemanticRole::Intrinsic;
                    return ty;
                }
                if self.is_zero(rhs) {
                    let mut ty = left.clone();
                    ty.role = SemanticRole::Intrinsic;
                    return ty;
                }
                check_shape_compatibility(left, right, span, diagnostics);
                check_dimension_compatibility(left, right, span, diagnostics);
                check_frame_compatibility(left, right, span, diagnostics);
                merge_types(left, right, SemanticRole::Intrinsic)
            }
            BinaryOp::Mul | BinaryOp::Div => {
                let shape = multiply_shape(&left.shape, &right.shape, span, diagnostics);
                let dimension = match (left.dimension, right.dimension) {
                    (Some(left), Some(right)) if op == BinaryOp::Mul => {
                        left.checked_product(right).ok()
                    }
                    (Some(left), Some(right)) => left.checked_quotient(right).ok(),
                    _ => None,
                };
                SemanticType {
                    axes: semantic_axes(&shape),
                    shape,
                    dimension,
                    quantity_kind: None,
                    frame: merge_frame(&left.frame, &right.frame, span, diagnostics),
                    role: SemanticRole::Intrinsic,
                }
            }
            BinaryOp::Pow => {
                require_scalar_dimensionless(right, span, diagnostics);
                let dimension = self
                    .integer_literal(rhs)
                    .and_then(|power| left.dimension?.checked_powi(power).ok());
                let mut ty = left.clone();
                ty.dimension = dimension;
                ty.quantity_kind = None;
                ty.role = SemanticRole::Intrinsic;
                ty
            }
            BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                check_shape_compatibility(left, right, span, diagnostics);
                check_dimension_compatibility(left, right, span, diagnostics);
                check_frame_compatibility(left, right, span, diagnostics);
                SemanticType {
                    shape: SemanticShape::Boolean,
                    axes: vec![],
                    dimension: None,
                    quantity_kind: None,
                    frame: Frame::Neutral,
                    role: SemanticRole::Intrinsic,
                }
            }
        }
    }

    fn call_type(
        &self,
        function: &str,
        args: &[ExprId],
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> SemanticType {
        let arg = |index: usize| args.get(index).map(|id| &self.expressions[id.index()].ty);
        match function {
            "boundary" | "interface" | "interface_region" => {
                require_arity(function, args, 1, span, diagnostics);
                if let Some(arg) = arg(0)
                    && !matches!(arg.shape, SemanticShape::String | SemanticShape::Deferred)
                {
                    diagnostics.push(error(
                        "TYPE_STRING_REQUIRED",
                        format!("`{function}` expects a string region name"),
                        span,
                    ));
                }
                SemanticType {
                    shape: SemanticShape::Region,
                    axes: vec![],
                    dimension: None,
                    quantity_kind: None,
                    frame: Frame::Neutral,
                    role: SemanticRole::Intrinsic,
                }
            }
            "dt" => {
                require_arity(function, args, 1, span, diagnostics);
                let mut ty = arg(0)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::deferred(SemanticRole::Intrinsic));
                ty.dimension = ty
                    .dimension
                    .and_then(|dimension| dimension.checked_quotient(Dimension::TIME).ok());
                ty.quantity_kind = None;
                ty.role = SemanticRole::Intrinsic;
                ty
            }
            "grad" | "rotated_grad" | "sym_grad" => {
                require_arity(function, args, 1, span, diagnostics);
                self.derivative_type(arg(0), true, diagnostics, span)
            }
            "div" | "curl" => {
                require_arity(function, args, 1, span, diagnostics);
                if function == "curl" {
                    self.curl_type(arg(0), diagnostics, span)
                } else {
                    self.derivative_type(arg(0), false, diagnostics, span)
                }
            }
            "dot" | "inner" => {
                require_arity(function, args, 2, span, diagnostics);
                if let (Some(left), Some(right)) = (arg(0), arg(1)) {
                    contraction_type(left, right, function == "inner", span, diagnostics)
                } else {
                    SemanticType::deferred(SemanticRole::Intrinsic)
                }
            }
            "sin" | "cos" | "exp" | "log" | "ln" => {
                require_arity(function, args, 1, span, diagnostics);
                if let Some(arg) = arg(0) {
                    require_scalar_dimensionless(arg, span, diagnostics);
                }
                SemanticType::numeric(
                    ValueShape::Scalar,
                    Some(Dimension::DIMENSIONLESS),
                    Frame::Neutral,
                    SemanticRole::Intrinsic,
                )
            }
            "sqrt" => {
                require_arity(function, args, 1, span, diagnostics);
                let mut ty = arg(0)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::deferred(SemanticRole::Intrinsic));
                ty.dimension = ty
                    .dimension
                    .and_then(|dimension| dimension.checked_root(2).ok());
                ty.quantity_kind = None;
                ty.role = SemanticRole::Intrinsic;
                ty
            }
            "abs" | "min" | "max" | "integrate" => {
                if args.is_empty() {
                    diagnostics.push(error(
                        "TYPE_ARITY",
                        format!("`{function}` requires at least one argument"),
                        span,
                    ));
                }
                arg(0)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::deferred(SemanticRole::Intrinsic))
            }
            "trace" | "trace_minus" | "trace_plus" => {
                require_arity(function, args, 1, span, diagnostics);
                let mut ty = arg(0)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::deferred(SemanticRole::Intrinsic));
                if matches!(
                    ty.shape,
                    SemanticShape::Numeric(ValueShape::Tensor { .. })
                        | SemanticShape::Numeric(ValueShape::SymmetricTensor(_))
                ) && function == "trace"
                {
                    ty.shape = SemanticShape::Numeric(ValueShape::Scalar);
                    ty.axes.clear();
                } else {
                    ty.frame = Frame::Neutral;
                }
                ty.role = SemanticRole::Intrinsic;
                ty
            }
            "jump" | "average" | "conj" => {
                require_arity(function, args, 1, span, diagnostics);
                let mut ty = arg(0)
                    .cloned()
                    .unwrap_or_else(|| SemanticType::deferred(SemanticRole::Intrinsic));
                if matches!(function, "jump" | "average") {
                    ty.frame = Frame::Neutral;
                }
                ty.role = SemanticRole::Intrinsic;
                ty
            }
            "normal_component" | "normal_component_minus" | "normal_component_plus" => {
                require_arity(function, args, 1, span, diagnostics);
                normal_component_type(arg(0), span, diagnostics)
            }
            "zero_vector" => {
                require_arity(function, args, 1, span, diagnostics);
                let extent = args
                    .first()
                    .and_then(|id| self.integer_literal(*id))
                    .and_then(|value| u8::try_from(value).ok())
                    .unwrap_or(0);
                if extent == 0 {
                    diagnostics.push(error(
                        "TYPE_INVALID_AXIS",
                        "zero_vector extent must be a positive integer",
                        span,
                    ));
                }
                SemanticType::numeric(
                    ValueShape::Vector(extent),
                    None,
                    Frame::Neutral,
                    SemanticRole::Intrinsic,
                )
            }
            "zero_tensor" => {
                require_arity(function, args, 1, span, diagnostics);
                let extent = args
                    .first()
                    .and_then(|id| self.integer_literal(*id))
                    .and_then(|value| u8::try_from(value).ok())
                    .unwrap_or(0);
                if extent == 0 {
                    diagnostics.push(error(
                        "TYPE_INVALID_AXIS",
                        "zero_tensor extent must be a positive integer",
                        span,
                    ));
                }
                SemanticType::numeric(
                    ValueShape::Tensor {
                        rows: extent,
                        cols: extent,
                    },
                    None,
                    Frame::Neutral,
                    SemanticRole::Intrinsic,
                )
            }
            _ => SemanticType::deferred(SemanticRole::Intrinsic),
        }
    }

    fn call_kind(&self, function: &str, args: Vec<ExprId>) -> SemanticExprKind {
        if args.is_empty() {
            return SemanticExprKind::Call {
                function: function.to_owned(),
                args,
            };
        }
        let unary_arg = || {
            args.first()
                .copied()
                .expect("call arity diagnosed before kind")
        };
        match function {
            "dt" => SemanticExprKind::Differential {
                operator: DifferentialOperator::TimeDerivative,
                arg: unary_arg(),
            },
            "grad" => SemanticExprKind::Differential {
                operator: DifferentialOperator::Gradient,
                arg: unary_arg(),
            },
            "div" => SemanticExprKind::Differential {
                operator: DifferentialOperator::Divergence,
                arg: unary_arg(),
            },
            "curl" => SemanticExprKind::Differential {
                operator: DifferentialOperator::Curl,
                arg: unary_arg(),
            },
            "rotated_grad" => SemanticExprKind::Differential {
                operator: DifferentialOperator::RotatedGradient,
                arg: unary_arg(),
            },
            "sym_grad" => SemanticExprKind::Differential {
                operator: DifferentialOperator::SymmetricGradient,
                arg: unary_arg(),
            },
            "dot" | "inner" if args.len() >= 2 => {
                let lhs = args[0];
                let rhs = args[1];
                let count = if function == "inner" {
                    self.expressions[lhs.index()]
                        .ty
                        .axes
                        .len()
                        .max(self.expressions[rhs.index()].ty.axes.len())
                } else {
                    1
                };
                SemanticExprKind::Contraction {
                    lhs,
                    rhs,
                    axes: (0..count)
                        .map(|axis| AxisContraction {
                            lhs: axis as u8,
                            rhs: axis as u8,
                        })
                        .collect(),
                    conjugate_lhs: function == "inner",
                }
            }
            "trace"
                if args.first().is_some_and(|value| {
                    matches!(
                        self.expressions[value.index()].ty.shape,
                        SemanticShape::Numeric(ValueShape::Tensor { .. })
                            | SemanticShape::Numeric(ValueShape::SymmetricTensor(_))
                    )
                }) =>
            {
                SemanticExprKind::TensorTrace {
                    value: unary_arg(),
                    axes: AxisContraction { lhs: 0, rhs: 1 },
                }
            }
            "trace" | "trace_minus" | "trace_plus" => SemanticExprKind::FacetTrace {
                value: unary_arg(),
                side: match function {
                    "trace_minus" => TraceSide::Minus,
                    "trace_plus" => TraceSide::Plus,
                    _ => TraceSide::Exterior,
                },
            },
            "jump" => SemanticExprKind::Jump { value: unary_arg() },
            "average" => SemanticExprKind::Average { value: unary_arg() },
            "conj" => SemanticExprKind::Conjugate { value: unary_arg() },
            "normal_component" | "normal_component_minus" | "normal_component_plus" => {
                SemanticExprKind::NormalComponent {
                    value: unary_arg(),
                    side: match function {
                        "normal_component_minus" => TraceSide::Minus,
                        "normal_component_plus" => TraceSide::Plus,
                        _ => TraceSide::Exterior,
                    },
                }
            }
            _ => SemanticExprKind::Call {
                function: function.to_owned(),
                args,
            },
        }
    }

    /// Type and build a `ProviderCall` node from a declared provider signature (contract
    /// C1.2): arity (`TYPE_PROVIDER_ARITY`), per-input selector literals
    /// (`TYPE_PROVIDER_SELECTOR`), and per-input dimension (`TYPE_PROVIDER_INPUT_KIND`, only
    /// when both the declared and argument dimensions are known).
    fn provider_call(
        &mut self,
        provider: ProviderId,
        args: Vec<ExprId>,
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> (SemanticExprKind, SemanticType) {
        let signature = &self.providers[provider.index()];
        if args.len() != signature.inputs.len() {
            diagnostics.push(error(
                "TYPE_PROVIDER_ARITY",
                format!(
                    "provider `{}` expects {} argument(s), got {}",
                    signature.name,
                    signature.inputs.len(),
                    args.len()
                ),
                span,
            ));
        }
        for (index, input) in signature.inputs.iter().enumerate() {
            let Some(&arg) = args.get(index) else {
                continue;
            };
            let arg_expr = &self.expressions[arg.index()];
            let arg_span = arg_expr.span;
            let arg_ty = arg_expr.ty.clone();
            if input.quantity_kind.is_none() {
                if self.integer_literal(arg).is_none() {
                    diagnostics.push(error(
                        "TYPE_PROVIDER_SELECTOR",
                        format!(
                            "provider `{}` input `{}` requires an integer selector literal",
                            signature.name, input.name
                        ),
                        arg_span,
                    ));
                }
            } else if let (Some(expected), Some(actual)) = (input.dimension, arg_ty.dimension)
                && expected != actual
                && !matches!(arg_ty.shape, SemanticShape::Deferred)
            {
                diagnostics.push(error(
                    "TYPE_PROVIDER_INPUT_KIND",
                    format!(
                        "provider `{}` input `{}` expects dimension `{expected}`, argument has dimension `{actual}`",
                        signature.name, input.name
                    ),
                    arg_span,
                ));
            }
        }
        let output = &signature.output;
        let ty = SemanticType {
            shape: SemanticShape::Numeric(output.shape.clone()),
            axes: axes(&output.shape),
            dimension: output.dimension,
            quantity_kind: Some(output.quantity_kind.clone()),
            frame: Frame::Neutral,
            role: SemanticRole::Provider,
        };
        (SemanticExprKind::ProviderCall { provider, args }, ty)
    }

    fn curl_type(
        &self,
        arg: Option<&SemanticType>,
        diagnostics: &mut Vec<SourceDiagnostic>,
        span: SourceSpan,
    ) -> SemanticType {
        let Some(arg) = arg else {
            return SemanticType::deferred(SemanticRole::Intrinsic);
        };
        let Frame::Domain(domain) = arg.frame else {
            return SemanticType::deferred(SemanticRole::Intrinsic);
        };
        let spatial = self.domains[domain.index()].spatial_dimension;
        let shape = match (&arg.shape, spatial) {
            (SemanticShape::Numeric(ValueShape::Vector(2)), 2) => {
                SemanticShape::Numeric(ValueShape::Scalar)
            }
            (SemanticShape::Numeric(ValueShape::Vector(3)), 3) => {
                SemanticShape::Numeric(ValueShape::Vector(3))
            }
            (SemanticShape::Numeric(ValueShape::Scalar), 2) => {
                SemanticShape::Numeric(ValueShape::Vector(2))
            }
            (SemanticShape::Deferred, _) => SemanticShape::Deferred,
            _ => {
                diagnostics.push(error(
                    "TYPE_SHAPE_MISMATCH",
                    "curl requires a scalar in 2D or a vector in 2D/3D",
                    span,
                ));
                SemanticShape::Deferred
            }
        };
        SemanticType {
            axes: semantic_axes(&shape),
            shape,
            dimension: arg
                .dimension
                .and_then(|dimension| dimension.checked_quotient(Dimension::LENGTH).ok()),
            quantity_kind: None,
            frame: Frame::Domain(domain),
            role: SemanticRole::Intrinsic,
        }
    }

    fn derivative_type(
        &self,
        arg: Option<&SemanticType>,
        gradient: bool,
        diagnostics: &mut Vec<SourceDiagnostic>,
        span: SourceSpan,
    ) -> SemanticType {
        let Some(arg) = arg else {
            return SemanticType::deferred(SemanticRole::Intrinsic);
        };
        let Frame::Domain(domain) = arg.frame else {
            let mut ty = arg.clone();
            ty.shape = SemanticShape::Deferred;
            ty.axes.clear();
            ty.dimension = None;
            ty.role = SemanticRole::Intrinsic;
            return ty;
        };
        let spatial = self.domains[domain.index()].spatial_dimension;
        let shape = match (&arg.shape, gradient, spatial) {
            (SemanticShape::Numeric(ValueShape::Scalar), true, 1) => {
                SemanticShape::Numeric(ValueShape::Scalar)
            }
            (SemanticShape::Numeric(ValueShape::Scalar), true, extent) => {
                SemanticShape::Numeric(ValueShape::Vector(extent))
            }
            (SemanticShape::Numeric(ValueShape::Vector(extent)), true, 1) => {
                SemanticShape::Numeric(ValueShape::Vector(*extent))
            }
            (SemanticShape::Numeric(ValueShape::Vector(extent)), true, columns) => {
                SemanticShape::Numeric(ValueShape::Tensor {
                    rows: *extent,
                    cols: columns,
                })
            }
            (SemanticShape::Numeric(ValueShape::Vector(extent)), false, 1) => {
                SemanticShape::Numeric(ValueShape::Vector(*extent))
            }
            (SemanticShape::Numeric(ValueShape::Vector(_)), false, _) => {
                SemanticShape::Numeric(ValueShape::Scalar)
            }
            (SemanticShape::Numeric(ValueShape::Tensor { rows, .. }), false, _) => {
                SemanticShape::Numeric(ValueShape::Vector(*rows))
            }
            (SemanticShape::Deferred, _, _) => SemanticShape::Deferred,
            _ => {
                diagnostics.push(error(
                    "TYPE_SHAPE_MISMATCH",
                    "differential operator received an incompatible shape",
                    span,
                ));
                SemanticShape::Deferred
            }
        };
        SemanticType {
            axes: semantic_axes(&shape),
            shape,
            dimension: arg
                .dimension
                .and_then(|dimension| dimension.checked_quotient(Dimension::LENGTH).ok()),
            quantity_kind: None,
            frame: Frame::Domain(domain),
            role: SemanticRole::Intrinsic,
        }
    }

    fn index_type(
        &self,
        value: ExprId,
        indices: &[ExprId],
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> SemanticType {
        let source = &self.expressions[value.index()].ty;
        if !matches!(source.shape, SemanticShape::Deferred) && indices.len() > source.axes.len() {
            diagnostics.push(error(
                "TYPE_AXIS_COUNT",
                format!(
                    "expression has {} axes but {} indices were supplied",
                    source.axes.len(),
                    indices.len()
                ),
                span,
            ));
        }
        for (position, index) in indices.iter().enumerate() {
            let index_ty = &self.expressions[index.index()].ty;
            require_scalar_dimensionless(
                index_ty,
                self.expressions[index.index()].span,
                diagnostics,
            );
            if let (Some(axis), Some(literal_index)) =
                (source.axes.get(position), self.integer_literal(*index))
                && (literal_index < 0 || literal_index >= i32::from(axis.extent))
            {
                diagnostics.push(error(
                    "TYPE_AXIS_BOUNDS",
                    format!(
                        "index {literal_index} is outside axis extent {}",
                        axis.extent
                    ),
                    self.expressions[index.index()].span,
                ));
            }
        }
        let remaining = source.axes.get(indices.len()..).unwrap_or_default();
        let shape = match remaining {
            [] => SemanticShape::Numeric(ValueShape::Scalar),
            [axis] => SemanticShape::Numeric(ValueShape::Vector(axis.extent)),
            [rows, cols] => SemanticShape::Numeric(ValueShape::Tensor {
                rows: rows.extent,
                cols: cols.extent,
            }),
            _ => SemanticShape::Deferred,
        };
        let mut ty = source.clone();
        ty.shape = shape;
        ty.axes = remaining.to_vec();
        ty.role = SemanticRole::Intrinsic;
        ty
    }

    fn vector_type(
        &self,
        elements: &[ExprId],
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) -> SemanticType {
        if elements.is_empty() {
            diagnostics.push(error(
                "TYPE_INVALID_AXIS",
                "vector literal cannot be empty",
                span,
            ));
        }
        if let Some(first) = elements.first() {
            for element in &elements[1..] {
                check_shape_compatibility(
                    &self.expressions[first.index()].ty,
                    &self.expressions[element.index()].ty,
                    span,
                    diagnostics,
                );
                check_dimension_compatibility(
                    &self.expressions[first.index()].ty,
                    &self.expressions[element.index()].ty,
                    span,
                    diagnostics,
                );
            }
            let first = &self.expressions[first.index()].ty;
            let mut ty = SemanticType::numeric(
                ValueShape::Vector(elements.len() as u8),
                first.dimension,
                first.frame.clone(),
                SemanticRole::Literal,
            );
            ty.quantity_kind = first.quantity_kind.clone();
            ty
        } else {
            SemanticType::numeric(
                ValueShape::Vector(0),
                None,
                Frame::Neutral,
                SemanticRole::Literal,
            )
        }
    }

    fn integer_literal(&self, id: ExprId) -> Option<i32> {
        match self.expressions[id.index()].kind {
            SemanticExprKind::Number {
                value, unit: None, ..
            } if value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64 => {
                Some(value as i32)
            }
            _ => None,
        }
    }

    fn update_deferred_symbol(&mut self, name: &str, expression: ExprId) {
        if let Some(symbol) = self.symbol_names.get(name).copied() {
            let role = self.symbols[symbol.index()].ty.role.clone();
            let mut ty = self.expressions[expression.index()].ty.clone();
            ty.role = role;
            self.symbols[symbol.index()].ty = ty;
            self.symbols[symbol.index()].domain =
                match self.expressions[expression.index()].ty.frame {
                    Frame::Domain(domain) => Some(domain),
                    Frame::Neutral => None,
                };
        }
    }

    fn require_compatible_symbol(
        &self,
        symbol: SymbolId,
        value: ExprId,
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) {
        let expected = &self.symbols[symbol.index()].ty;
        let actual = &self.expressions[value.index()].ty;
        if !self.is_zero(value) {
            check_shape_compatibility(expected, actual, span, diagnostics);
            check_dimension_compatibility(expected, actual, span, diagnostics);
            check_frame_compatibility(expected, actual, span, diagnostics);
        }
    }

    fn require_compatible(
        &self,
        left: ExprId,
        right: ExprId,
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) {
        if self.is_zero(left) || self.is_zero(right) {
            return;
        }
        let left = &self.expressions[left.index()].ty;
        let right = &self.expressions[right.index()].ty;
        check_shape_compatibility(left, right, span, diagnostics);
        check_dimension_compatibility(left, right, span, diagnostics);
        check_frame_compatibility(left, right, span, diagnostics);
    }

    fn require_frame(
        &self,
        expression: ExprId,
        domain: DomainId,
        span: SourceSpan,
        diagnostics: &mut Vec<SourceDiagnostic>,
    ) {
        if let Frame::Domain(actual) = self.expressions[expression.index()].ty.frame
            && actual != domain
        {
            diagnostics.push(error(
                "TYPE_FRAME_MISMATCH",
                "integrand frame does not match its integration domain",
                span,
            ));
        }
    }

    fn push_declaration(
        &mut self,
        name: &str,
        role: SemanticRole,
        domain: Option<DomainId>,
        kind: SemanticDeclarationKind,
        span: SourceSpan,
    ) {
        let id = DeclarationId(self.declarations.len() as u32);
        self.declarations.push(SemanticDeclaration {
            id,
            name: name.into(),
            symbol: self.symbol_names.get(name).copied(),
            role,
            domain,
            kind,
            span,
        });
    }

    fn is_zero(&self, expression: ExprId) -> bool {
        matches!(
            self.expressions[expression.index()].kind,
            SemanticExprKind::Number {
                value: 0.0,
                unit: None,
                ..
            }
        )
    }
}

fn axes(shape: &ValueShape) -> Vec<Axis> {
    match shape {
        ValueShape::Scalar => vec![],
        ValueShape::Vector(extent) => vec![Axis {
            position: 0,
            extent: *extent,
        }],
        ValueShape::Tensor { rows, cols } => vec![
            Axis {
                position: 0,
                extent: *rows,
            },
            Axis {
                position: 1,
                extent: *cols,
            },
        ],
        ValueShape::SymmetricTensor(extent) => vec![
            Axis {
                position: 0,
                extent: *extent,
            },
            Axis {
                position: 1,
                extent: *extent,
            },
        ],
    }
}

fn region_name(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::Call { function, args, .. }
            if matches!(
                function.as_str(),
                "boundary" | "interface" | "interface_region"
            ) && args.len() == 1 =>
        {
            match &args[0] {
                Expr::String { value, .. } => Some(value),
                _ => None,
            }
        }
        _ => None,
    }
}

fn semantic_axes(shape: &SemanticShape) -> Vec<Axis> {
    match shape {
        SemanticShape::Numeric(shape) => axes(shape),
        _ => vec![],
    }
}

fn validate_shape(shape: &ValueShape, span: SourceSpan, diagnostics: &mut Vec<SourceDiagnostic>) {
    let invalid = match shape {
        ValueShape::Scalar => false,
        ValueShape::Vector(extent) | ValueShape::SymmetricTensor(extent) => *extent == 0,
        ValueShape::Tensor { rows, cols } => *rows == 0 || *cols == 0,
    };
    if invalid {
        diagnostics.push(error(
            "TYPE_INVALID_AXIS",
            "shape axes must have positive extents",
            span,
        ));
    }
}

fn known_kind_dimension(kind: &QuantityKindId) -> Option<Dimension> {
    match kind.as_str().rsplit(':').next().unwrap_or_default() {
        "ThermodynamicTemperature" | "TemperatureDifference" => Some(Dimension::TEMPERATURE),
        "Dimensionless" => Some(Dimension::DIMENSIONLESS),
        _ => None,
    }
}

fn contraction_type(
    left: &SemanticType,
    right: &SemanticType,
    all_axes: bool,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) -> SemanticType {
    if !matches!(left.shape, SemanticShape::Deferred)
        && !matches!(right.shape, SemanticShape::Deferred)
    {
        let valid_shape = if all_axes {
            left.shape == right.shape
                && matches!(
                    left.shape,
                    SemanticShape::Numeric(ValueShape::Vector(_))
                        | SemanticShape::Numeric(ValueShape::Tensor { .. })
                        | SemanticShape::Numeric(ValueShape::SymmetricTensor(_))
                )
        } else {
            matches!(
                (&left.shape, &right.shape),
                (
                    SemanticShape::Numeric(ValueShape::Vector(left)),
                    SemanticShape::Numeric(ValueShape::Vector(right))
                ) if left == right
            )
        };
        if !valid_shape {
            diagnostics.push(error(
                if left.axes.len() == right.axes.len() {
                    "TYPE_SHAPE_MISMATCH"
                } else {
                    "TYPE_AXIS_MISMATCH"
                },
                if all_axes {
                    "inner product requires equal vector or tensor operands"
                } else {
                    "dot product requires equal vector operands"
                },
                span,
            ));
        }
    }
    check_frame_compatibility(left, right, span, diagnostics);
    SemanticType::numeric(
        ValueShape::Scalar,
        match (left.dimension, right.dimension) {
            (Some(left), Some(right)) => left.checked_product(right).ok(),
            _ => None,
        },
        merge_frame(&left.frame, &right.frame, span, diagnostics),
        SemanticRole::Intrinsic,
    )
}

fn normal_component_type(
    arg: Option<&SemanticType>,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) -> SemanticType {
    let Some(arg) = arg else {
        return SemanticType::deferred(SemanticRole::Intrinsic);
    };
    let shape = match arg.shape {
        SemanticShape::Numeric(ValueShape::Vector(_)) => SemanticShape::Numeric(ValueShape::Scalar),
        SemanticShape::Numeric(ValueShape::Tensor { rows, .. })
        | SemanticShape::Numeric(ValueShape::SymmetricTensor(rows)) => {
            SemanticShape::Numeric(ValueShape::Vector(rows))
        }
        SemanticShape::Deferred => SemanticShape::Deferred,
        _ => {
            diagnostics.push(error(
                "TYPE_SHAPE_MISMATCH",
                "normal component requires a vector or tensor operand",
                span,
            ));
            SemanticShape::Deferred
        }
    };
    SemanticType {
        axes: semantic_axes(&shape),
        shape,
        dimension: arg.dimension,
        quantity_kind: None,
        frame: Frame::Neutral,
        role: SemanticRole::Intrinsic,
    }
}

fn require_arity(
    function: &str,
    args: &[ExprId],
    expected: usize,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) {
    if args.len() != expected {
        diagnostics.push(error(
            "TYPE_ARITY",
            format!(
                "`{function}` expects {expected} argument(s), found {}",
                args.len()
            ),
            span,
        ));
    }
}

fn require_scalar_dimensionless(
    ty: &SemanticType,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) {
    if !matches!(
        ty.shape,
        SemanticShape::Numeric(ValueShape::Scalar) | SemanticShape::Deferred
    ) {
        diagnostics.push(error(
            "TYPE_SHAPE_MISMATCH",
            "expected a scalar value",
            span,
        ));
    }
    if ty
        .dimension
        .is_some_and(|dimension| !dimension.is_dimensionless())
    {
        diagnostics.push(error(
            "TYPE_DIMENSION_MISMATCH",
            "expected a dimensionless value",
            span,
        ));
    }
}

fn is_numeric_or_deferred(shape: &SemanticShape) -> bool {
    matches!(shape, SemanticShape::Numeric(_) | SemanticShape::Deferred)
}

fn multiply_shape(
    left: &SemanticShape,
    right: &SemanticShape,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) -> SemanticShape {
    match (left, right) {
        (SemanticShape::Deferred, _) | (_, SemanticShape::Deferred) => SemanticShape::Deferred,
        (SemanticShape::Numeric(ValueShape::Scalar), other)
        | (other, SemanticShape::Numeric(ValueShape::Scalar))
            if is_numeric_or_deferred(other) =>
        {
            other.clone()
        }
        (SemanticShape::Numeric(left), SemanticShape::Numeric(right)) if left == right => {
            SemanticShape::Numeric(left.clone())
        }
        _ => {
            diagnostics.push(error(
                "TYPE_SHAPE_MISMATCH",
                "multiplication has incompatible operand shapes",
                span,
            ));
            SemanticShape::Deferred
        }
    }
}

fn merge_types(left: &SemanticType, right: &SemanticType, role: SemanticRole) -> SemanticType {
    let mut ty = if matches!(left.shape, SemanticShape::Deferred) {
        right.clone()
    } else {
        left.clone()
    };
    ty.dimension = left.dimension.or(right.dimension);
    ty.quantity_kind = if left.quantity_kind == right.quantity_kind {
        left.quantity_kind.clone()
    } else {
        None
    };
    ty.frame = match (&left.frame, &right.frame) {
        (Frame::Neutral, frame) | (frame, Frame::Neutral) => frame.clone(),
        (frame, _) => frame.clone(),
    };
    ty.role = role;
    ty
}

fn merge_frame(
    left: &Frame,
    right: &Frame,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) -> Frame {
    match (left, right) {
        (Frame::Neutral, frame) | (frame, Frame::Neutral) => frame.clone(),
        (Frame::Domain(left), Frame::Domain(right)) if left == right => Frame::Domain(*left),
        _ => {
            diagnostics.push(error(
                "TYPE_FRAME_MISMATCH",
                "operands belong to different domain frames",
                span,
            ));
            left.clone()
        }
    }
}

fn check_shape_compatibility(
    left: &SemanticType,
    right: &SemanticType,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) {
    if !matches!(left.shape, SemanticShape::Deferred)
        && !matches!(right.shape, SemanticShape::Deferred)
        && left.shape != right.shape
    {
        diagnostics.push(error(
            "TYPE_SHAPE_MISMATCH",
            format!("incompatible shapes {:?} and {:?}", left.shape, right.shape),
            span,
        ));
    }
}

fn check_dimension_compatibility(
    left: &SemanticType,
    right: &SemanticType,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) {
    if let (Some(left), Some(right)) = (left.dimension, right.dimension)
        && left != right
    {
        diagnostics.push(error(
            "TYPE_DIMENSION_MISMATCH",
            format!("incompatible dimensions `{left}` and `{right}`"),
            span,
        ));
    }
}

fn check_frame_compatibility(
    left: &SemanticType,
    right: &SemanticType,
    span: SourceSpan,
    diagnostics: &mut Vec<SourceDiagnostic>,
) {
    if let (Frame::Domain(left), Frame::Domain(right)) = (&left.frame, &right.frame)
        && left != right
    {
        diagnostics.push(error(
            "TYPE_FRAME_MISMATCH",
            "operands belong to different domain frames",
            span,
        ));
    }
}

fn duplicate(kind: &str, name: &str, span: SourceSpan, previous: SourceSpan) -> SourceDiagnostic {
    let mut diagnostic = error(
        "RESOLVE_DUPLICATE_NAME",
        format!("duplicate {kind} `{name}`"),
        span,
    );
    diagnostic.related.push(RelatedSpan {
        span: previous,
        message: "first declared here".into(),
    });
    diagnostic
}

fn error(code: &'static str, message: impl Into<String>, span: SourceSpan) -> SourceDiagnostic {
    SourceDiagnostic::error(code, message, span).phase("elaboration")
}

/// A non-fatal diagnostic (contract C1.2: `RESOLVE_UNDECLARED_PROVIDER`). Advisories never
/// fail `elaborate_module`/`compile_semantics`; see `SemanticCompilation::advisories`.
fn advisory(code: &'static str, message: impl Into<String>, span: SourceSpan) -> SourceDiagnostic {
    SourceDiagnostic::warning(code, message, span).phase("elaboration")
}

/// Function names dispatched to a fixed intrinsic meaning by `call_type`/`call_kind`. A
/// declared `provider` may not reuse one of these names (see `declare_providers`), and a call
/// to one of these names is never looked up as a provider even if a same-named provider
/// exists, so intrinsic semantics can never be shadowed.
pub(crate) const INTRINSIC_CALL_NAMES: &[&str] = &[
    "boundary",
    "interface",
    "interface_region",
    // Synthesized directly by `elaborate_expr`'s `Expr::Name` arm for the bare identifier `t`;
    // never reached through `Expr::Call`'s provider/intrinsic dispatch, but still not a real
    // function call a `provider` declaration or an unbound-provider slot should ever name.
    "time",
    "dt",
    "grad",
    "rotated_grad",
    "sym_grad",
    "div",
    "curl",
    "dot",
    "inner",
    "sin",
    "cos",
    "exp",
    "log",
    "ln",
    "sqrt",
    "abs",
    "min",
    "max",
    "integrate",
    "trace",
    "trace_minus",
    "trace_plus",
    "jump",
    "average",
    "conj",
    "normal_component",
    "normal_component_minus",
    "normal_component_plus",
    "zero_vector",
    "zero_tensor",
];

pub(crate) fn is_intrinsic_call(name: &str) -> bool {
    INTRINSIC_CALL_NAMES.contains(&name)
}
