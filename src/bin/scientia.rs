use quantitas::UnitRegistry;
use scientia::{
    IncidenceSystem, SemanticModel, SemanticModule, SourceDiagnostic, compile_schedule,
    compile_semantics, compile_variational_form, derive_binding_slots, derive_coupling_graph,
    derive_variational_form, elaborate_module, factor_operator, format_scientific_module,
    infer_form_requirements, parse_scientific_module_diagnostics, semantic_arena_digest,
    semantic_digest,
};
use std::{env, fs, process::ExitCode};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(usage)?;
    let rest = args.collect::<Vec<_>>();
    let json = rest.iter().any(|arg| arg == "--json");
    let mut positional = rest.iter().filter(|arg| arg.as_str() != "--json");
    let file = positional.next().ok_or_else(usage)?;
    let selector = positional.next().map(String::as_str);
    let detail = positional.next().map(String::as_str);
    if positional.next().is_some() {
        return Err(usage());
    }
    if detail.is_some() && command != "explain" {
        return Err(usage());
    }
    let source = fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
    let module = parse_scientific_module_diagnostics(&source)
        .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
    match command.as_str() {
        "check" => {
            let (semantic, advisories) =
                elaborate_module(&module, &UnitRegistry::si_bootstrap())
                    .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "ok": true,
                        "models": module.models.len(),
                        "semantic_digest": semantic_digest(&module),
                        "semantic_arena_digest": semantic_arena_digest(&semantic),
                        "expressions": semantic.models.iter().map(|model| model.expressions.len()).sum::<usize>(),
                        "advisories": advisories,
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "ok: {} model(s), digest {}",
                    module.models.len(),
                    semantic_arena_digest(&semantic)
                );
                if !advisories.is_empty() {
                    println!("{}", render_diagnostics(&source, &advisories, false));
                }
            }
        }
        "parse" => println!(
            "{}",
            serde_json::to_string_pretty(&module).map_err(|e| e.to_string())?
        ),
        "fmt" => print!("{}", format_scientific_module(&module)),
        "freeze" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema":"scientia-scientific-lock/1",
                    "module":module.name,
                    "source_digest":semantic_digest(&module),
                    "semantic_digest":semantic_arena_digest(&semantic)
                }))
                .map_err(|e| e.to_string())?
            );
        }
        "inspect" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let models = module
                .models
                .iter()
                .map(|model| {
                    serde_json::json!({
                        "name": model.name,
                        "domains": model.domains.len(),
                        "fields": model.fields,
                        "parameters": model.parameters,
                        "constants": model.constants,
                        "sources": model.sources,
                        "properties": model.properties,
                        "constitutive_laws": model.constitutive_laws,
                        "equations": model.equations,
                        "forms": model.forms,
                        "initial_conditions": model.initial_conditions,
                        "boundary_conditions": model.boundary_conditions,
                        "interface_conditions": model.interface_conditions,
                        "observables": model.observables,
                        "invariants": model.invariants,
                        "verifications": model.verifications,
                    })
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema": "scientia-scientific-inspect/1",
                    "module": module.name,
                    "semantic_digest": semantic_digest(&module),
                    "semantic_arena_digest": semantic_arena_digest(&semantic),
                    "semantic_models": semantic.models,
                    "models": models,
                }))
                .map_err(|e| e.to_string())?
            );
        }
        "coupling" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let semantic_model = select_model(&semantic, selector)?;
            let model = module
                .models
                .iter()
                .find(|model| model.name == semantic_model.name)
                .ok_or_else(|| {
                    format!("source module has no model named `{}`", semantic_model.name)
                })?;
            println!(
                "{}",
                serde_json::to_string_pretty(&derive_coupling_graph(model))
                    .map_err(|e| e.to_string())?
            );
        }
        "structural" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let semantic_model = select_model(&semantic, selector)?;
            let model = module
                .models
                .iter()
                .find(|model| model.name == semantic_model.name)
                .ok_or_else(|| {
                    format!("source module has no model named `{}`", semantic_model.name)
                })?;
            let incidence = IncidenceSystem::from_model(model).map_err(|e| e.to_string())?;
            let output = match compile_schedule(&incidence) {
                Ok(schedule) => serde_json::json!({
                    "incidence": incidence,
                    "schedule": schedule,
                }),
                Err(error) => serde_json::json!({
                    "incidence": incidence,
                    "schedule_error": error.to_string(),
                }),
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?
            );
        }
        "explain" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let (semantic_model, edge_selector) = select_explain(&semantic, selector, detail)?;
            let model = module
                .models
                .iter()
                .find(|model| model.name == semantic_model.name)
                .ok_or_else(|| {
                    format!("source module has no model named `{}`", semantic_model.name)
                })?;
            let graph = derive_coupling_graph(model);
            let edges = graph
                .edges
                .iter()
                .filter(|edge| {
                    edge_selector.is_none_or(|name| edge.from == name || edge.to == name)
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema": "scientia-scientific-explain/1",
                    "model": model.name,
                    "selector": edge_selector,
                    "edges": edges,
                }))
                .map_err(|e| e.to_string())?
            );
        }
        "form" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let (model, form_name) = select_model_item(&semantic, selector, "form")?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &compile_variational_form(&semantic, &model.name, form_name)
                        .map_err(|e| e.to_string())?
                )
                .map_err(|e| e.to_string())?
            );
        }
        "derive-form" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let (model, equation_name) = select_model_item(&semantic, selector, "derive-form")?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &derive_variational_form(&semantic, &model.name, equation_name)
                        .map_err(|e| e.to_string())?
                )
                .map_err(|e| e.to_string())?
            );
        }
        "requirements" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let (model, form_name) = select_model_item(&semantic, selector, "requirements")?;
            let form = compile_variational_form(&semantic, &model.name, form_name)
                .map_err(|error| error.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &infer_form_requirements(&semantic, &form).map_err(|error| error.to_string())?
                )
                .map_err(|error| error.to_string())?
            );
        }
        "derive-requirements" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let (model, equation_name) =
                select_model_item(&semantic, selector, "derive-requirements")?;
            let form = derive_variational_form(&semantic, &model.name, equation_name)
                .map_err(|error| error.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &infer_form_requirements(&semantic, &form).map_err(|error| error.to_string())?
                )
                .map_err(|error| error.to_string())?
            );
        }
        "operator" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let (model, form_name) = select_model_item(&semantic, selector, "operator")?;
            let form = compile_variational_form(&semantic, &model.name, form_name)
                .map_err(|error| error.to_string())?;
            let requirements =
                infer_form_requirements(&semantic, &form).map_err(|error| error.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &factor_operator(&form, &requirements).map_err(|error| error.to_string())?
                )
                .map_err(|error| error.to_string())?
            );
        }
        "derive-operator" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let (model, equation_name) = select_model_item(&semantic, selector, "derive-operator")?;
            let form = derive_variational_form(&semantic, &model.name, equation_name)
                .map_err(|error| error.to_string())?;
            let requirements =
                infer_form_requirements(&semantic, &form).map_err(|error| error.to_string())?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &factor_operator(&form, &requirements).map_err(|error| error.to_string())?
                )
                .map_err(|error| error.to_string())?
            );
        }
        "elaborate" => {
            let (semantic, _advisories) = elaborate_module(&module, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&semantic).map_err(|error| error.to_string())?
            );
        }
        "slots" => {
            let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap())
                .map_err(|diagnostics| render_diagnostics(&source, &diagnostics, json))?;
            let manifests = derive_binding_slots(&compilation);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&manifests).map_err(|error| error.to_string())?
                );
            } else {
                for manifest in &manifests {
                    println!("model {} ({} slots):", manifest.model, manifest.slots.len());
                    for slot in &manifest.slots {
                        println!("  {:<10} {}", format!("{:?}", slot.status), slot.id);
                    }
                }
            }
        }
        _ => return Err(usage()),
    }
    Ok(())
}

fn usage() -> String {
    "usage: scientia <check|fmt|parse|elaborate|inspect|freeze|explain|coupling|structural|form|derive-form|requirements|derive-requirements|operator|derive-operator|slots> [--json] <model.res> [model|model:item] [detail]".into()
}

fn select_model<'a>(
    semantic: &'a SemanticModule,
    selector: Option<&str>,
) -> Result<&'a SemanticModel, String> {
    if let Some(name) = selector {
        return semantic
            .models
            .iter()
            .find(|model| model.name == name)
            .ok_or_else(|| format!("semantic module has no model named `{name}`"));
    }
    match semantic.models.as_slice() {
        [] => Err("module has no model".into()),
        [model] => Ok(model),
        models => Err(format!(
            "module has {} models; command requires an explicit model selector",
            models.len()
        )),
    }
}

fn select_model_item<'a>(
    semantic: &'a SemanticModule,
    selector: Option<&'a str>,
    command: &str,
) -> Result<(&'a SemanticModel, &'a str), String> {
    let selector = selector
        .ok_or_else(|| format!("{command} command requires an item or `model:item` selector"))?;
    if let Some((model_name, item)) = selector.split_once(':') {
        if model_name.is_empty() || item.is_empty() {
            return Err(format!(
                "{command} command requires a nonempty `model:item` selector"
            ));
        }
        return Ok((select_model(semantic, Some(model_name))?, item));
    }
    let model = select_model(semantic, None).map_err(|_| {
        format!("module has multiple models; {command} command requires a `model:item` selector")
    })?;
    Ok((model, selector))
}

fn select_explain<'a>(
    semantic: &'a SemanticModule,
    selector: Option<&'a str>,
    detail: Option<&'a str>,
) -> Result<(&'a SemanticModel, Option<&'a str>), String> {
    if let Some(detail) = detail {
        return Ok((select_model(semantic, selector)?, Some(detail)));
    }
    if let Some(selector) = selector
        && let Some(model) = semantic.models.iter().find(|model| model.name == selector)
    {
        return Ok((model, None));
    }
    Ok((select_model(semantic, None)?, selector))
}

fn render_diagnostics(source: &str, diagnostics: &[SourceDiagnostic], json: bool) -> String {
    if json {
        return serde_json::to_string_pretty(&serde_json::json!({
            "ok": false,
            "diagnostics": diagnostics,
        }))
        .unwrap_or_else(|error| error.to_string());
    }
    diagnostics
        .iter()
        .map(|diagnostic| {
            let (line, column) = line_column(source, diagnostic.span.start);
            format!(
                "{}:{}: {} [{}] {}",
                line,
                column,
                match diagnostic.severity {
                    scientia::SourceSeverity::Note => "note",
                    scientia::SourceSeverity::Warning => "warning",
                    scientia::SourceSeverity::Error => "error",
                },
                diagnostic.code,
                diagnostic.message
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn line_column(source: &str, byte_offset: usize) -> (usize, usize) {
    let offset = byte_offset.min(source.len());
    let prefix = &source[..offset];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let column = prefix
        .rsplit_once('\n')
        .map_or(prefix.chars().count(), |(_, tail)| tail.chars().count())
        + 1;
    (line, column)
}
