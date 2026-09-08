use malleus::{BufferBinding, Executable, Interpreter, OperandId};
use quantitas::UnitRegistry;
use scientia::*;
use std::collections::BTreeMap;

const SOURCE: &str = r#"
module w8.point;
model Example {
 domain body { dimension = 2; coordinates = cartesian; }
 field u: unknown scalar H1(order=1) on body;
 input field observed: Dimensionless on body;
 input value offset: Dimensionless;
 provider law(x: Dimensionless) -> Dimensionless { differentiability = analytic_provided; }
 property first = law(u + offset);
 constitutive nested = first * u;
 source f: VolumetricSource;
 equation balance on body { -div(grad(u)) = f; }
 objective misfit { minimize integrate(0.5 * (u-observed) * (u-observed)); }
 observable energy { integrate(nested * nested); }
 observable norm { integrate(u*u); }
}
"#;
fn model(source: &str) -> SemanticCompilation {
    compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap()
}
fn run(
    kernel: &malleus::StructuredKernel,
    supplied: &BTreeMap<OperandId, Vec<f64>>,
) -> Vec<Vec<f64>> {
    let executable = Executable::reference(malleus::validate(kernel.clone()).unwrap());
    let mut buffers = kernel
        .operands
        .iter()
        .enumerate()
        .map(|(i, operand)| {
            supplied
                .get(&OperandId::new(i))
                .cloned()
                .unwrap_or_else(|| vec![0.; operand.shape.iter().product::<usize>().max(1)])
        })
        .collect::<Vec<_>>();
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(i, values)| BufferBinding::new(OperandId::new(i), values))
        .collect::<Vec<_>>();
    Interpreter::run(&executable, &mut bindings).unwrap();
    buffers
}
fn values(
    node: &PointExpressionNode,
    supplied: &BTreeMap<SymbolId, Vec<f64>>,
    capture: f64,
) -> BTreeMap<OperandId, Vec<f64>> {
    let bundle = &node.kernels.bundles[0];
    let program = &node.factorization.integrals[0].primal;
    bundle
        .primal_inputs
        .iter()
        .map(|binding| {
            let input = program
                .inputs
                .iter()
                .find(|input| input.id == binding.input)
                .unwrap();
            (
                binding.operand,
                supplied
                    .get(&input.binding.symbol)
                    .cloned()
                    .unwrap_or_else(|| vec![capture]),
            )
        })
        .collect()
}
fn primal(node: &PointExpressionNode, supplied: &BTreeMap<OperandId, Vec<f64>>) -> Vec<f64> {
    let bundle = &node.kernels.bundles[0];
    run(&bundle.module.kernels[bundle.primal_kernel_index], supplied)[bundle.primal_output.index()]
        .clone()
}
fn symbol(model: &SemanticModel, name: &str) -> SymbolId {
    model
        .symbols
        .iter()
        .find(|symbol| symbol.name == name)
        .unwrap()
        .id
}
fn directional(
    node: &PointExpressionNode,
    supplied: &BTreeMap<OperandId, Vec<f64>>,
    parameter: bool,
    seed: f64,
) -> f64 {
    let bundle = &node.kernels.bundles[0];
    let contract = if parameter {
        &bundle.parameter
    } else {
        &bundle.jvp
    };
    let mut values = supplied.clone();
    for input in &contract.independent_operands {
        values.insert(input.derivative, vec![seed]);
    }
    run(&bundle.module.kernels[contract.kernel_index], &values)
        [contract.dependent_operands[0].derivative.index()][0]
}
#[test]
fn objective_primal_state_and_capture_products_are_executable() {
    let compilation = model(SOURCE);
    let module = &compilation.semantic;
    let model = &module.models[0];
    let compiled = compile_cell_functional(module, "Example", "misfit").unwrap();
    let inputs = values(
        &compiled.root,
        &BTreeMap::from([
            (symbol(model, "u"), vec![3.]),
            (symbol(model, "observed"), vec![1.]),
        ]),
        0.,
    );
    assert_eq!(primal(&compiled.root, &inputs), vec![2.]);
    assert_eq!(directional(&compiled.root, &inputs, false, 1.), 2.);
    assert_eq!(directional(&compiled.root, &inputs, true, 1.), -2.);
    let bundle = &compiled.root.kernels.bundles[0];
    let mut covector_inputs = inputs.clone();
    covector_inputs.insert(bundle.vjp.dependent_operands[0].derivative, vec![1.]);
    let covectors = run(
        &bundle.module.kernels[bundle.vjp.kernel_index],
        &covector_inputs,
    );
    assert_eq!(
        covectors[bundle.vjp.independent_operands[0].derivative.index()],
        vec![2.]
    );
    let reverse = &compiled.root.parameter_vjp[0];
    let mut reverse_inputs = reverse
        .primal_operands
        .iter()
        .map(|op| (op.derivative, inputs[&op.primal].clone()))
        .collect::<BTreeMap<_, _>>();
    reverse_inputs.insert(reverse.dependent_operands[0].derivative, vec![1.]);
    let covectors = run(&reverse.kernel, &reverse_inputs);
    assert_eq!(
        covectors[reverse.independent_operands[0].derivative.index()],
        vec![-2.]
    );
}
#[test]
fn provider_argument_dag_and_nested_constitutive_chain_are_explicit() {
    let compilation = model(SOURCE);
    let module = &compilation.semantic;
    let model = &module.models[0];
    let declaration = model
        .declarations
        .iter()
        .find(|d| d.name == "nested")
        .unwrap();
    let SemanticDeclarationKind::ConstitutiveLaw { value } = declaration.kind else {
        panic!()
    };
    let compiled = compile_point_expression(
        module,
        "Example",
        declaration.id,
        value,
        model.domains[0].id,
    )
    .unwrap();
    let [capture] = compiled.root.captures.as_slice() else {
        panic!("{:?}", compiled.root.captures)
    };
    assert_eq!(
        capture.disposition,
        PointCaptureDisposition::ProviderValueAndProductsRequired
    );
    let arg = &compiled.arguments[&capture.arguments[0]];
    let bindings = BTreeMap::from([
        (symbol(model, "u"), vec![3.]),
        (symbol(model, "offset"), vec![1.]),
    ]);
    let arg_inputs = values(arg, &bindings, 0.);
    let argument = primal(arg, &arg_inputs)[0];
    assert_eq!(argument, 4.);
    // Provider law(x)=x². Its derivative is supplied by the provider, and its argument is compiled.
    let root_inputs = values(&compiled.root, &bindings, argument * argument);
    assert_eq!(primal(&compiled.root, &root_inputs), vec![48.]);
    let arg_direction = directional(arg, &arg_inputs, false, 1.);
    let total = directional(&compiled.root, &root_inputs, false, 1.)
        + directional(
            &compiled.root,
            &root_inputs,
            true,
            2. * argument * arg_direction,
        );
    assert_eq!(total, 40.);
    assert_eq!(compiled.arguments.len(), 1);
}
#[test]
fn polynomial_degree_is_conservative_and_identity_binds_origin() {
    let compilation = model(SOURCE);
    let module = &compilation.semantic;
    let model = &module.models[0];
    let compiled = compile_cell_functional(module, "Example", "norm").unwrap();
    let mut degrees = PolynomialDegreeBindings::default();
    degrees.symbols.insert(symbol(model, "u"), 1);
    assert_eq!(
        infer_expression_polynomial_degree(model, compiled.expression, &degrees),
        Some(2)
    );
    let energy = compile_cell_functional(module, "Example", "energy").unwrap();
    assert_eq!(
        infer_expression_polynomial_degree(model, energy.expression, &degrees),
        None
    );
    let provider = energy.root.captures[0].definition.unwrap();
    degrees.provider_calls.insert(provider, 2);
    assert_eq!(
        infer_expression_polynomial_degree(model, energy.expression, &degrees),
        Some(6)
    );
    let other = model
        .declarations
        .iter()
        .find(|d| d.name == "misfit")
        .unwrap();
    assert_eq!(
        compile_point_expression(
            module,
            "Example",
            other.id,
            compiled.expression,
            compiled.domain
        )
        .unwrap_err()
        .code,
        "POINT_EXPRESSION_ORIGIN"
    );
    let changed = model_fn(SOURCE.replace("integrate(u*u)", "integrate(2*u*u)"));
    assert_ne!(
        compiled.artifact_digest,
        compile_cell_functional(&changed.semantic, "Example", "norm")
            .unwrap()
            .artifact_digest
    );
}
fn model_fn(source: String) -> SemanticCompilation {
    model(&source)
}
#[test]
fn unsupported_functionals_and_unavailable_provider_products_refuse() {
    let nested = model(&SOURCE.replace("integrate(u*u)", "integrate(integrate(u*u))"));
    assert_eq!(
        compile_cell_functional(&nested.semantic, "Example", "norm")
            .unwrap_err()
            .code,
        "POINT_CONSTRUCT_UNSUPPORTED"
    );
    let nondiff = model(&SOURCE.replace("analytic_provided", "none"));
    assert_eq!(
        compile_cell_functional(&nondiff.semantic, "Example", "energy")
            .unwrap_err()
            .code,
        "POINT_PROVIDER_DERIVATIVE_UNAVAILABLE"
    );
}

#[test]
fn elasticity_vector_norm_uses_generated_state_covectors() {
    let source = r#"module elasticity;
model Elasticity {
 domain body { dimension = 3; coordinates = cartesian; }
 field displacement: unknown vector(3) H1(order=1) on body;
 observable displacement_magnitude_sq { integrate(dot(displacement,displacement)); }
}"#;
    let compilation = model(source);
    let model = &compilation.semantic.models[0];
    let compiled = compile_cell_functional(
        &compilation.semantic,
        "Elasticity",
        "displacement_magnitude_sq",
    )
    .unwrap();
    let input = values(
        &compiled.root,
        &BTreeMap::from([(symbol(model, "displacement"), vec![1., 2., 3.])]),
        0.,
    );
    assert_eq!(primal(&compiled.root, &input), vec![14.]);
    let bundle = &compiled.root.kernels.bundles[0];
    let mut reverse = input;
    reverse.insert(bundle.vjp.dependent_operands[0].derivative, vec![1.]);
    let result = run(&bundle.module.kernels[bundle.vjp.kernel_index], &reverse);
    assert_eq!(
        result[bundle.vjp.independent_operands[0].derivative.index()],
        vec![2., 4., 6.]
    );
    let degrees = PolynomialDegreeBindings {
        symbols: BTreeMap::from([(symbol(model, "displacement"), 1)]),
        ..Default::default()
    };
    assert_eq!(
        infer_expression_polynomial_degree(model, compiled.expression, &degrees),
        Some(2)
    );
}

#[test]
fn provider_chain_reverse_product_and_nested_dag_are_complete() {
    let compilation = model(SOURCE);
    let model = &compilation.semantic.models[0];
    let declaration = model
        .declarations
        .iter()
        .find(|d| d.name == "nested")
        .unwrap();
    let SemanticDeclarationKind::ConstitutiveLaw { value } = declaration.kind else {
        panic!()
    };
    let compiled = compile_point_expression(
        &compilation.semantic,
        "Example",
        declaration.id,
        value,
        model.domains[0].id,
    )
    .unwrap();
    let bindings = BTreeMap::from([
        (symbol(model, "u"), vec![3.]),
        (symbol(model, "offset"), vec![1.]),
    ]);
    let input = values(&compiled.root, &bindings, 16.);
    let bundle = &compiled.root.kernels.bundles[0];
    let mut state_reverse = input.clone();
    state_reverse.insert(bundle.vjp.dependent_operands[0].derivative, vec![1.]);
    let direct = run(
        &bundle.module.kernels[bundle.vjp.kernel_index],
        &state_reverse,
    )[bundle.vjp.independent_operands[0].derivative.index()][0];
    let parameter = &compiled.root.parameter_vjp[0];
    let mut capture_reverse = parameter
        .primal_operands
        .iter()
        .map(|op| (op.derivative, input[&op.primal].clone()))
        .collect::<BTreeMap<_, _>>();
    capture_reverse.insert(parameter.dependent_operands[0].derivative, vec![1.]);
    let capture_covector = run(&parameter.kernel, &capture_reverse)
        [parameter.independent_operands[0].derivative.index()][0];
    let arg = &compiled.arguments[&compiled.root.captures[0].arguments[0]];
    let arg_bundle = &arg.kernels.bundles[0];
    let mut arg_reverse = values(arg, &bindings, 0.);
    arg_reverse.insert(
        arg_bundle.vjp.dependent_operands[0].derivative,
        vec![8. * capture_covector],
    );
    let chain = run(
        &arg_bundle.module.kernels[arg_bundle.vjp.kernel_index],
        &arg_reverse,
    )[arg_bundle.vjp.independent_operands[0].derivative.index()][0];
    assert_eq!(direct + chain, 40.);
    let nested = model_fn(SOURCE.replace("law(u + offset)", "law(law(u + offset))"));
    let energy = compile_cell_functional(&nested.semantic, "Example", "energy").unwrap();
    assert_eq!(
        energy.arguments.len(),
        2,
        "shared repeated nested expression arguments compile only once"
    );
    assert_eq!(
        energy.root.captures.len(),
        1,
        "repeated reference to first shares its provider capture"
    );
}

#[test]
fn identity_covers_capture_contract_support_and_executable_payload() {
    let compilation = model(SOURCE);
    let compiled = compile_cell_functional(&compilation.semantic, "Example", "energy").unwrap();
    compiled.validate_identity().unwrap();
    let roundtrip: PointExpressionKernels =
        serde_json::from_str(&serde_json::to_string(&compiled).unwrap()).unwrap();
    assert_eq!(compiled, roundtrip);
    roundtrip.validate_identity().unwrap();
    let mut changed = compiled.clone();
    changed.root.captures[0].disposition = PointCaptureDisposition::ExternalInput;
    assert_eq!(
        changed.validate_identity().unwrap_err().code,
        "POINT_IDENTITY_MISMATCH"
    );
    let mut changed = compiled.clone();
    changed.root.parameter_vjp[0].kernel.name.push_str("forged");
    assert!(changed.validate_identity().is_err());
    let mut changed = compiled;
    changed.domain = DomainId(42);
    assert!(changed.validate_identity().is_err());
}

#[test]
fn spatial_degree_never_guesses_unbound_data_or_coordinate_names() {
    let compilation = model(SOURCE);
    let model = &compilation.semantic.models[0];
    let compiled = compile_cell_functional(&compilation.semantic, "Example", "energy").unwrap();
    let arg = compiled.arguments.values().next().unwrap();
    let mut degrees = PolynomialDegreeBindings::default();
    degrees.symbols.insert(symbol(model, "u"), 1);
    assert_eq!(
        infer_expression_polynomial_degree(model, arg.expression, &degrees),
        None
    );
    degrees.symbols.insert(symbol(model, "offset"), 0);
    assert_eq!(
        infer_expression_polynomial_degree(model, arg.expression, &degrees),
        Some(1)
    );
    // Declared provider output is still unknown even when all of its arguments are polynomial.
    assert_eq!(
        infer_expression_polynomial_degree(model, compiled.expression, &degrees),
        None
    );
}

#[test]
fn tensor_constitutive_definitions_inline_without_frozen_inputs() {
    let source = r#"module tensor;
model Tensor {
 domain body { dimension = 2; coordinates = cartesian; }
 field u: unknown vector(2) H1(order=1) on body;
 constitutive strain = sym_grad(u);
 constitutive stress = 2 * strain;
 observable energy { integrate(inner(strain, stress)); }
}"#;
    let compilation = model(source);
    let model = &compilation.semantic.models[0];
    let declaration = model
        .declarations
        .iter()
        .find(|d| d.name == "stress")
        .unwrap();
    let SemanticDeclarationKind::ConstitutiveLaw { value } = declaration.kind else {
        panic!()
    };
    let compiled = compile_point_expression(
        &compilation.semantic,
        "Tensor",
        declaration.id,
        value,
        model.domains[0].id,
    )
    .unwrap();
    assert!(
        compiled.root.captures.is_empty(),
        "all constitutive arithmetic belongs in the kernel"
    );
    let input = values(
        &compiled.root,
        &BTreeMap::from([(symbol(model, "u"), vec![1., 2., 2., 4.])]),
        0.,
    );
    assert_eq!(primal(&compiled.root, &input), vec![2., 4., 4., 8.]);
    // Explicit conjugating inner products retain FC4's typed real64 refusal.
    assert_eq!(
        compile_cell_functional(&compilation.semantic, "Tensor", "energy")
            .unwrap_err()
            .code,
        "POINT_FACTORIZATION_UNSUPPORTED"
    );
}

#[test]
fn support_and_nested_provider_provenance_are_checked() {
    let noncartesian = model(
        &SOURCE
            .replace("coordinates = cartesian", "coordinates = cylindrical")
            .replace("dimension = 2", "dimension = 3"),
    );
    assert_eq!(
        compile_cell_functional(&noncartesian.semantic, "Example", "norm")
            .unwrap_err()
            .code,
        "POINT_COORDINATES_UNSUPPORTED"
    );
    let compilation = model(SOURCE);
    let model = &compilation.semantic.models[0];
    let compiled = compile_cell_functional(&compilation.semantic, "Example", "energy").unwrap();
    let source_decl = model
        .declarations
        .iter()
        .find(|d| d.name == "first")
        .unwrap();
    let SemanticDeclarationKind::Property { value } = source_decl.kind else {
        panic!()
    };
    assert_eq!(compiled.root.captures[0].definition, Some(value));
    assert_eq!(
        compiled.root.form.receipt.source_declaration,
        compiled.declaration
    );
    assert_eq!(
        compiled.root.form.source_semantic_digest.hex,
        semantic_arena_digest(&compilation.semantic)
    );
}
