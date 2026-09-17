//! SC-W2 compiler-owned connection relations. No mesh, DOF or numerical solver lives here.
use crate::composition::{
    InstanceId, SysDomainId, SysRegionId, SysResId, SysVarId, SystemCompilation, SystemError,
};
use crate::id::span_independent_digest;
use crate::scientific::{DeclKind, GlobalDeclId, ModuleClosure, SystemDecl};
use crate::semantic::{Registries, SemanticPort};
use crate::{Digest, SourceLocator};
use quantitas::QuantityKindId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionPort {
    pub instance: InstanceId,
    pub name: String,
    pub region: SysRegionId,
    pub domain: SysDomainId,
    pub variable: SysVarId,
    pub residual: SysResId,
    /// Multiply the authored residual by this sign to make its flux outward.
    pub orientation: i8,
    pub origin: SourceLocator,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionSet {
    pub schema: String,
    pub interface: String,
    pub connector: GlobalDeclId,
    pub equal_kind: QuantityKindId,
    pub balance_kind: QuantityKindId,
    pub conserves: QuantityKindId,
    /// Bounded matching elimination admits exactly two exterior traces.
    pub ports: [ConnectionPort; 2],
    pub equal_coefficients: [i8; 2],
    pub balance_coefficients: [i8; 2],
    pub origin: SourceLocator,
    pub identity: Digest,
}
impl ConnectionSet {
    pub fn expected_identity(&self) -> Digest {
        let mut copy = self.clone();
        copy.identity = Digest::blake3(b"unset");
        span_independent_digest(&copy)
    }
    pub fn validate(&self) -> Result<(), SystemError> {
        if self.schema != "scientia-connection-set/1"
            || self.identity != self.expected_identity()
            || self.equal_coefficients != [1, -1]
            || self.balance_coefficients != [1, 1]
            || self.ports[0].domain == self.ports[1].domain
            || self.ports.iter().any(|p| ![-1, 1].contains(&p.orientation))
        {
            return Err(fail(
                "SYSTEM_CONNECTION_INVALID",
                "connection identity, domains or relation coefficients disagree",
            ));
        }
        Ok(())
    }
}
fn fail(code: &str, detail: impl Into<String>) -> SystemError {
    SystemError::Port {
        code: code.into(),
        detail: detail.into(),
    }
}

pub(crate) fn compile_connections(
    closure: &ModuleClosure,
    registries: Registries<'_>,
    declaration: &SystemDecl,
    compiled: &SystemCompilation,
) -> Result<Vec<ConnectionSet>, SystemError> {
    let system = &compiled.system;
    for entry in &closure.modules {
        let mut names = BTreeSet::new();
        for connector in &entry.module.connectors {
            if !names.insert(&connector.name) {
                return Err(fail("PORT_CONNECTOR_DUPLICATE", &connector.name));
            }
            let mut members = BTreeSet::new();
            if connector.members.iter().any(|m| !members.insert(&m.name)) {
                return Err(fail("PORT_MEMBER_DUPLICATE", &connector.name));
            }
        }
    }
    let mut ports = BTreeMap::new();
    for instance in &system.instances {
        let model = compiled.model(instance.instance);
        for port in &model.ports {
            ports.insert((instance.name.clone(), port.name.clone()), (instance, port));
        }
    }
    let mut seen = BTreeSet::new();
    let mut interfaces = BTreeSet::new();
    for interface in &declaration.interfaces {
        if !interfaces.insert(&interface.name)
            || interface.domains[0] == interface.domains[1]
            || interface
                .domains
                .iter()
                .any(|d| !system.domains.iter().any(|x| &x.name == d))
        {
            return Err(fail("SYSTEM_INTERFACE_INVALID", &interface.name));
        }
    }
    let connector_id = |instance: &crate::composition::InstanceRecord,
                        port: &SemanticPort|
     -> Result<GlobalDeclId, SystemError> {
        let entry = closure.by_digest(&instance.model.module).unwrap();
        let id = closure
            .declaration(&entry.name, DeclKind::Connector, &port.connector)
            .or_else(|| {
                compiled.modules[&instance.model.module]
                    .semantic
                    .imports
                    .iter()
                    .find(|i| i.name == port.connector && i.target.kind == DeclKind::Connector)
                    .map(|i| i.target.clone())
            })
            .ok_or_else(|| fail("PORT_CONNECTOR_UNKNOWN", &port.connector))?;
        Ok(id)
    };
    let mut result = vec![];
    for connection in &declaration.connects {
        if connection.ports.len() != 2 {
            return Err(fail(
                "SYSTEM_CONNECTION_ARITY_UNSUPPORTED",
                "matching elimination requires two ports",
            ));
        }
        let mut endpoints = vec![];
        let mut declared_connector = None;
        let mut region_interface = None;
        let mut kinds = None;
        for reference in &connection.ports {
            let key = (reference.instance.clone(), reference.member.clone());
            if !seen.insert(key.clone()) {
                return Err(fail("SYSTEM_PORT_OVERLAP", format!("{}.{}", key.0, key.1)));
            }
            let (instance, port) = ports
                .get(&key)
                .ok_or_else(|| fail("SYSTEM_PORT_UNKNOWN", format!("{}.{}", key.0, key.1)))?;
            let id = connector_id(instance, port)?;
            if declared_connector
                .as_ref()
                .is_some_and(|previous| previous != &id)
            {
                return Err(fail(
                    "PORT_CONNECTOR_MISMATCH",
                    "connection ports must share the same declared connector",
                ));
            }
            let connector = closure
                .by_digest(&id.module)
                .unwrap()
                .module
                .connectors
                .iter()
                .find(|c| c.name == id.name)
                .unwrap();
            if connector.members.len() != 2 {
                return Err(fail(
                    "PORT_TRACE_CLASS_UNSUPPORTED",
                    "one H1 equal and one normal balance member required",
                ));
            }
            let equal = connector
                .members
                .iter()
                .find(|m| !m.balance && m.name == port.equal_member)
                .ok_or_else(|| fail("PORT_MEMBER_UNASSIGNED", &port.equal_member))?;
            let balance = connector
                .members
                .iter()
                .find(|m| m.balance && m.name == port.balance_member)
                .ok_or_else(|| fail("PORT_MEMBER_UNASSIGNED", &port.balance_member))?;
            let conserved = balance
                .conserves
                .as_ref()
                .ok_or_else(|| fail("PORT_CONSERVATION_UNDECLARED", &balance.name))?;
            if equal.conserves.is_some() {
                return Err(fail(
                    "PORT_ROLE_INVALID",
                    "equal member cannot carry a balance declaration",
                ));
            }
            let model = compiled.model(instance.instance);
            for (symbol, kind) in [
                (port.field, &equal.quantity_kind),
                (port.flux, &balance.quantity_kind),
            ] {
                let definition = registries
                    .kinds
                    .by_name(kind.as_str())
                    .ok_or_else(|| fail("PORT_KIND_UNKNOWN", kind.as_str()))?;
                let ty = &model.symbols[symbol.index()].ty;
                if ty.dimension != Some(definition.dimension) {
                    return Err(fail("PORT_KIND_MISMATCH", kind.as_str()));
                }
                if ty
                    .quantity_kind
                    .as_ref()
                    .is_some_and(|k| k != &definition.id)
                {
                    return Err(fail("PORT_KIND_MISMATCH", kind.as_str()));
                }
            }
            if registries.kinds.by_name(conserved.as_str()).is_none() {
                return Err(fail("PORT_KIND_UNKNOWN", conserved.as_str()));
            }
            let flux_dimension = registries
                .kinds
                .by_name(balance.quantity_kind.as_str())
                .unwrap()
                .dimension;
            let conserved_dimension = registries
                .kinds
                .by_name(conserved.as_str())
                .unwrap()
                .dimension;
            // Flux density integrated over a physical surface and time has the conserved
            // quantity's dimension. Two-dimensional cases are per unit out-of-plane depth.
            let integrated = flux_dimension
                .checked_product(quantitas::Dimension::LENGTH.checked_powi(2).unwrap())
                .and_then(|d| d.checked_product(quantitas::Dimension::TIME))
                .map_err(|e| fail("PORT_CONSERVATION_DIMENSION", e.to_string()))?;
            if integrated != conserved_dimension {
                return Err(fail("PORT_CONSERVATION_DIMENSION", conserved.as_str()));
            }
            kinds = Some((
                equal.quantity_kind.clone(),
                balance.quantity_kind.clone(),
                conserved.clone(),
            ));
            declared_connector = Some(id);
            let local_region = &model.regions[port.region.index()];
            let authored = declaration
                .instances
                .iter()
                .find(|i| i.name == instance.name)
                .unwrap();
            let interface_name = &authored
                .arguments
                .iter()
                .find(|a| a.parameter == local_region.name)
                .ok_or_else(|| fail("SYSTEM_PORT_REGION_UNMAPPED", &local_region.name))?
                .value;
            if region_interface
                .as_ref()
                .is_some_and(|prior| prior != interface_name)
            {
                return Err(fail(
                    "SYSTEM_INTERFACE_MISMATCH",
                    "ports must map to the same interface",
                ));
            }
            region_interface = Some(interface_name.clone());
            let region = instance
                .region_map
                .iter()
                .find(|(r, _)| *r == port.region)
                .unwrap()
                .1;
            let domain = system.regions[region.index()]
                .domain
                .ok_or_else(|| fail("SYSTEM_INTERFACE_MISMATCH", "port has no domain"))?;
            let variable = system
                .variables
                .iter()
                .find(|v| v.owner == instance.instance && v.local == port.field)
                .unwrap()
                .id;
            let residual=system.residuals.iter().find(|r|matches!(r.origin,crate::composition::ResidualOrigin::Equation{instance:i,declaration:d,..} if i==instance.instance && d==port.equation)).unwrap().id;
            endpoints.push(ConnectionPort {
                instance: instance.instance,
                name: port.name.clone(),
                region,
                domain,
                variable,
                residual,
                orientation: port.orientation,
                origin: SourceLocator {
                    module: instance.model.module.clone(),
                    span: port.span,
                },
            });
        }
        let name = region_interface.unwrap();
        let interface = declaration
            .interfaces
            .iter()
            .find(|i| i.name == name)
            .ok_or_else(|| fail("SYSTEM_INTERFACE_UNKNOWN", &name))?;
        let actual = endpoints
            .iter()
            .map(|p| system.domains[p.domain.index()].name.as_str())
            .collect::<BTreeSet<_>>();
        if actual != interface.domains.iter().map(String::as_str).collect() {
            return Err(fail("SYSTEM_INTERFACE_MISMATCH", &name));
        }
        let (equal_kind, balance_kind, conserves) = kinds.unwrap();
        let mut set = ConnectionSet {
            schema: "scientia-connection-set/1".into(),
            interface: name,
            connector: declared_connector.unwrap(),
            equal_kind,
            balance_kind,
            conserves,
            ports: endpoints.try_into().unwrap(),
            equal_coefficients: [1, -1],
            balance_coefficients: [1, 1],
            origin: SourceLocator {
                module: closure.root_module().digest.clone(),
                span: connection.span,
            },
            identity: Digest::blake3(b"unset"),
        };
        set.identity = set.expected_identity();
        set.validate()?;
        result.push(set);
    }
    if seen.len() != ports.len() {
        return Err(fail(
            "SYSTEM_PORT_UNCLOSED",
            "every boundary port must belong to an admitted connection",
        ));
    }
    Ok(result)
}
