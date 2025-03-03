use ark_bn254::Fr;
use ark_ff::{BigInteger, Field, PrimeField, Zero, One};
use sha3::{Keccak256, Digest};
use std::marker::PhantomData;

// ## Gate and Circuit Definitions

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    Add,
    Mul,
}

#[derive(Debug, Clone)]
pub struct Gate {
    left: usize,
    right: usize,
    ty: GateType,
}

#[derive(Debug, Clone)]
pub struct Circuit {
    layers: Vec<Vec<Gate>>,
    input_size: usize,
}

impl Circuit {
    pub fn new(layers: Vec<Vec<Gate>>, input_size: usize) -> Self {
        Circuit { layers, input_size }
    }

    // Evaluate the circuit with given inputs
    pub fn evaluate(&self, inputs: Vec<Fr>) -> (Vec<Vec<Fr>>, Fr) {
        assert_eq!(inputs.len(), self.input_size, "Input size mismatch");
        let mut intermediate_values = vec![inputs.clone()];
        let mut current_inputs = inputs;

        for layer in &self.layers {
            let mut layer_outputs = Vec::new();
            for gate in layer {
                let result = match gate.ty {
                    GateType::Add => current_inputs[gate.left] + current_inputs[gate.right],
                    GateType::Mul => current_inputs[gate.left] * current_inputs[gate.right],
                };
                layer_outputs.push(result);
            }
            current_inputs = layer_outputs;
            intermediate_values.push(current_inputs.clone());
        }

        (intermediate_values, current_inputs[0])
    }

    // Multilinear extension of the witness (w_i) at a given layer
    pub fn w_i(&self, layer_index: usize, values: &[Vec<Fr>]) -> MultilinearPoly {
        MultilinearPoly::new(values[layer_index].clone())
    }

    // Multilinear extension of the addition gate indicator
    pub fn add_i(&self, layer_index: usize) -> MultilinearPoly {
        let layer = &self.layers[layer_index];
        let input_size = if layer_index == 0 {
            self.input_size
        } else {
            self.layers[layer_index - 1].len()
        };
        let num_vars = (input_size as f64).log2().ceil() as usize;
        let poly_size = 1 << num_vars;
        let mut coeffs = vec![Fr::zero(); poly_size];
        for (i, gate) in layer.iter().enumerate() {
            if matches!(gate.ty, GateType::Add) {
                coeffs[i] = Fr::one();
            }
        }
        MultilinearPoly::new(coeffs)
    }

    // Multilinear extension of the multiplication gate indicator
    pub fn mul_i(&self, layer_index: usize) -> MultilinearPoly {
        let layer = &self.layers[layer_index];
        let input_size = if layer_index == 0 {
            self.input_size
        } else {
            self.layers[layer_index - 1].len()
        };
        let num_vars = (input_size as f64).log2().ceil() as usize;
        let poly_size = 1 << num_vars;
        let mut coeffs = vec![Fr::zero(); poly_size];
        for (i, gate) in layer.iter().enumerate() {
            if matches!(gate.ty, GateType::Mul) {
                coeffs[i] = Fr::one();
            }
        }
        MultilinearPoly::new(coeffs)
    }
}

// ## Multilinear Polynomial

#[derive(Clone)]
pub struct MultilinearPoly {
    coefficients: Vec<Fr>,
}

impl MultilinearPoly {
    fn new(coefficients: Vec<Fr>) -> Self {
        assert!(coefficients.len().is_power_of_two(), "Coefficients length must be a power of 2");
        Self { coefficients }
    }

    // Evaluate the multilinear polynomial at a point
    fn evaluate(&self, point: &[Fr]) -> Fr {
        let num_vars = (self.coefficients.len() as f64).log2() as usize;
        assert_eq!(point.len(), num_vars, "Point size must match number of variables");
        let mut result = Fr::zero();
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            let mut term = coeff;
            for (j, &r) in point.iter().enumerate() {
                if (i >> j) & 1 == 1 {
                    term *= r;
                } else {
                    term *= Fr::one() - r;
                }
            }
            result += term;
        }
        result
    }

    // Compute the univariate polynomial by fixing all variables except one
    fn univariate(&self, fixed_point: &[Fr], var_idx: usize) -> Vec<Fr> {
        let num_vars = (self.coefficients.len() as f64).log2() as usize;
        let degree = num_vars - 1; // Max degree of the univariate polynomial
        let mut coeffs = vec![Fr::zero(); degree + 1];
        for x in 0..=1 {
            let mut point = fixed_point.to_vec();
            point[var_idx] = Fr::from(x as u64);
            let eval = self.evaluate(&point);
            coeffs[x] = eval;
        }
        coeffs
    }
}

// Transcript Management

 struct Transcript<K: HashTrait, F: PrimeField> {
    hash_function: K,
    _field: PhantomData<F>,
}

impl<K: HashTrait, F: PrimeField> Transcript<K, F> {
    fn new(hash_function: K) -> Self {
        Self { hash_function, _field: PhantomData }
    }

    fn absorb(&mut self, data: &[u8]) {
        self.hash_function.update(data);
    }

    fn squeeze(&mut self) -> F {
        let hash_result = self.hash_function.finalize_reset();
        F::from_be_bytes_mod_order(&hash_result)
    }
}

trait HashTrait: Clone {
    fn update(&mut self, data: &[u8]);
    fn finalize_reset(&mut self) -> Vec<u8>;
}

impl HashTrait for Keccak256 {
    fn update(&mut self, data: &[u8]) {
        Digest::update(self, data);
    }

    fn finalize_reset(&mut self) -> Vec<u8> {
        let hash = self.clone().finalize();
        self.reset();
        hash.to_vec()
    }
}

// ## Sumcheck Protocol Structures

pub struct SumcheckRound {
    univariate_coeffs: Vec<Fr>,
    random_challenge: Fr,
}

pub struct LayerProof {
    sumcheck_rounds: Vec<SumcheckRound>,
    next_layer_evaluation: Fr
}

// ## Prover Implementation

pub struct GKRProver {
    circuit: Circuit,
    transcript: Transcript<Keccak256, Fr>,
    values: Vec<Vec<Fr>>,
}

impl GKRProver {
    pub fn new(circuit: Circuit, inputs: Vec<Fr>) -> Self {
        let (values, _) = circuit.evaluate(inputs);
        Self {
            circuit,
            transcript: Transcript::new(Keccak256::new()),
            values,
        }
    }

    pub fn prove(&mut self) -> Vec<LayerProof> {
        let mut proofs = Vec::new();
        for layer in (0..self.circuit.layers.len()).rev() {
            let sumcheck_proof = self.run_sumcheck(layer);
            proofs.push(sumcheck_proof);
        }
        proofs
    }

    fn run_sumcheck(&mut self, layer: usize) -> LayerProof {
        let input_size = if layer == 0 {
            self.circuit.input_size
        } else {
            self.circuit.layers[layer - 1].len()
        };
        let num_vars = (input_size as f64).log2().ceil() as usize;
        let w = self.circuit.w_i(layer + 1, &self.values);
        let add = self.circuit.add_i(layer);
        let mul = self.circuit.mul_i(layer);

        let mut rounds = Vec::new();
        let mut current_point = vec![Fr::zero(); num_vars];
        let mut current_sum = self.compute_initial_sum(&w, &add, &mul);

        self.transcript.absorb(&current_sum.into_bigint().to_bytes_be());

        for var_idx in 0..num_vars {
            let f = |point: &[Fr]| {
                let w_b = w.evaluate(point);
                let add_b = add.evaluate(point);
                let mul_b = mul.evaluate(point);
                add_b * w_b + mul_b * w_b * w_b // Simplified for single-point evaluation
            };
            let univariate_coeffs = self.compute_univariate(&f, &current_point, var_idx);
            self.transcript.absorb(
                &univariate_coeffs
                    .iter()
                    .flat_map(|c| c.into_bigint().to_bytes_be())
                    .collect::<Vec<_>>(),
            );
            let challenge = self.transcript.squeeze();
            rounds.push(SumcheckRound {
                univariate_coeffs,
                random_challenge: challenge,
            });
            current_sum = self.evaluate_univariate(&rounds.last().unwrap().univariate_coeffs, challenge);
            
            current_point[var_idx] = challenge;
        }
        let next_layer_evaluation = w.evaluate(&current_point);

        LayerProof {
            sumcheck_rounds: rounds,
            next_layer_evaluation, 
        }
    }

    fn compute_initial_sum(&self, w: &MultilinearPoly, add: &MultilinearPoly, mul: &MultilinearPoly) -> Fr {
        let num_vars = (w.coefficients.len() as f64).log2() as usize;
        let mut sum = Fr::zero();
        for i in 0..(1 << num_vars) {
            let mut point = Vec::new();
            for j in 0..num_vars {
                point.push(Fr::from(((i >> j) & 1) as u64));
            }
            let w_b = w.evaluate(&point);
            let add_b = add.evaluate(&point);
            let mul_b = mul.evaluate(&point);
            sum += add_b * w_b + mul_b * w_b * w_b;
        }
        sum
    }

    fn compute_univariate(&self, f: &impl Fn(&[Fr]) -> Fr, point: &[Fr], var_idx: usize) -> Vec<Fr> {
        let mut coeffs = vec![];
        for x in 0..=1 {
            let mut eval_point = point.to_vec();
            eval_point[var_idx] = Fr::from(x as u64);
            coeffs.push(f(&eval_point));
        }
        coeffs
    }

    fn evaluate_univariate(&self, coeffs: &[Fr], x: Fr) -> Fr {
        let mut result = Fr::zero();
        for (i, &coeff) in coeffs.iter().enumerate() {
            result += coeff * x.pow(&[i as u64]);
        }
        result
    }
}

// ## Verifier Implementation

pub struct GKRVerifier {
    transcript: Transcript<Keccak256, Fr>,
    circuit: Circuit,
}

impl GKRVerifier {
    pub fn new(circuit: Circuit) -> Self {
        Self {
            transcript: Transcript::new(Keccak256::new()),
            circuit,
        }
    }

    pub fn verify(&mut self, inputs: &[Fr], claimed_output: Fr, proof: &[LayerProof]) -> bool {
        let mut current_claim = claimed_output;
        for (layer_idx, layer_proof) in proof.iter().enumerate().rev() {
            let input_size = if layer_idx == 0 {
                self.circuit.input_size
            } else {
                self.circuit.layers[layer_idx - 1].len()
            };
            let num_vars = (input_size as f64).log2().ceil() as usize;

            self.transcript.absorb(&current_claim.into_bigint().to_bytes_be());
            let mut challenges = Vec::new();
            let mut current_eval = current_claim;

            for round in &layer_proof.sumcheck_rounds {
                self.transcript.absorb(
                    &round.univariate_coeffs
                        .iter()
                        .flat_map(|c| c.into_bigint().to_bytes_be())
                        .collect::<Vec<_>>(),
                );
                let challenge = self.transcript.squeeze();
                if self.evaluate_univariate(&round.univariate_coeffs, challenge) != current_eval {
                    return false;
                }
                challenges.push(challenge);
                current_eval = round.univariate_coeffs[0];
            }

            if challenges.len() != num_vars {
                return false;
            }

            // Verify the claim at the challenge point
            let add = self.circuit.add_i(layer_idx);
            let mul = self.circuit.mul_i(layer_idx);
            let w_next = if layer_idx + 1 < self.circuit.layers.len() {
                self.circuit.w_i(layer_idx + 1, &vec![vec![current_claim]])
            } else {
                MultilinearPoly::new(vec![current_claim])
            };
            // let w_eval = w_next.evaluate(&challenges);
            let add_eval = add.evaluate(&challenges);
            let mul_eval = mul.evaluate(&challenges);
            let expected_claim = add_eval * layer_proof.next_layer_evaluation + mul_eval * layer_proof.next_layer_evaluation * layer_proof.next_layer_evaluation;

            if expected_claim != current_claim {
                return false;
            }

            current_claim = layer_proof.next_layer_evaluation;
        }

        // Final check against inputs
        let w_0 = self.circuit.w_i(0, &vec![inputs.to_vec()]);
        let input_eval = w_0.evaluate(&vec![self.transcript.squeeze(); (self.circuit.input_size as f64).log2().ceil() as usize]);
        input_eval == current_claim
    }

    fn evaluate_univariate(&self, coeffs: &[Fr], x: Fr) -> Fr {
        let mut result = Fr::zero();
        for (i, &coeff) in coeffs.iter().enumerate() {
            result += coeff * x.pow(&[i as u64]);
        }
        result
    }
}















#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::PrimeField;

    // Helper function to create a sample circuit: (x0 + x1) * (x2 + x3)
    fn create_sample_circuit() -> Circuit {
        let layer0 = vec![
            Gate { left: 0, right: 1, ty: GateType::Add }, // x0 + x1
            Gate { left: 2, right: 3, ty: GateType::Add }, // x2 + x3
        ];
        let layer1 = vec![
            Gate { left: 0, right: 1, ty: GateType::Mul }, // (x0 + x1) * (x2 + x3)
        ];
        Circuit::new(vec![layer0, layer1], 4)
    }

    // Helper function to create sample inputs: [1, 2, 3, 4]
    fn sample_inputs() -> Vec<Fr> {
        vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)]
    }

    #[test]
    fn test_circuit_evaluation() {
        let circuit = create_sample_circuit();
        let inputs = sample_inputs();
        let (intermediate_values, final_output) = circuit.evaluate(inputs.clone());

        // Verify input layer
        assert_eq!(intermediate_values[0], inputs);
        // Verify layer 0: 1+2=3, 3+4=7
        assert_eq!(intermediate_values[1], vec![Fr::from(3), Fr::from(7)]);
        // Verify layer 1: 3*7=21
        assert_eq!(intermediate_values[2], vec![Fr::from(21)]);
        // Verify final output
        assert_eq!(final_output, Fr::from(21));
    }

    #[test]
    fn test_multilinear_polynomials() {
        let circuit = create_sample_circuit();
        let inputs = sample_inputs();
        let (intermediate_values, _) = circuit.evaluate(inputs.clone());

        // Test w_i for layer 0 (input layer)
        let w0 = circuit.w_i(0, &intermediate_values);
        assert_eq!(w0.evaluate(&[Fr::from(0), Fr::from(0)]), Fr::from(1)); // x0
        assert_eq!(w0.evaluate(&[Fr::from(1), Fr::from(0)]), Fr::from(2)); // x1
        assert_eq!(w0.evaluate(&[Fr::from(0), Fr::from(1)]), Fr::from(3)); // x2
        assert_eq!(w0.evaluate(&[Fr::from(1), Fr::from(1)]), Fr::from(4)); // x3

        // Test w_i for layer 1 (intermediate layer)
        let w1 = circuit.w_i(1, &intermediate_values);
        assert_eq!(w1.evaluate(&[Fr::from(0)]), Fr::from(3)); // Gate 0: 1+2=3
        assert_eq!(w1.evaluate(&[Fr::from(1)]), Fr::from(7)); // Gate 1: 3+4=7

        // Test add_i for layer 0 (addition gates)
        let add0 = circuit.add_i(0);
        assert_eq!(add0.evaluate(&[Fr::from(0)]), Fr::one()); // Gate 0 is add
        assert_eq!(add0.evaluate(&[Fr::from(1)]), Fr::one()); // Gate 1 is add

        // Test mul_i for layer 0 (no multiplication gates)
        let mul0 = circuit.mul_i(0);
        assert_eq!(mul0.evaluate(&[Fr::from(0)]), Fr::zero());
        assert_eq!(mul0.evaluate(&[Fr::from(1)]), Fr::zero());

        // Test add_i for layer 1 (no addition gates)
        let add1 = circuit.add_i(1);
        assert_eq!(add1.evaluate(&[Fr::from(0)]), Fr::zero());

        // Test mul_i for layer 1 (multiplication gate)
        let mul1 = circuit.mul_i(1);
        assert_eq!(mul1.evaluate(&[Fr::from(0)]), Fr::one());
    }

    #[test]
    fn test_prover_verifier() {
        let circuit = create_sample_circuit();
        let inputs = sample_inputs();
        let (_, expected_output) = circuit.evaluate(inputs.clone());

        // Generate proof with prover
        let mut prover = GKRProver::new(circuit.clone(), inputs.clone());
        let proof = prover.prove();

        // Verify proof with verifier
        let mut verifier = GKRVerifier::new(circuit);
        let is_valid = verifier.verify(&inputs, expected_output, &proof);
        assert!(is_valid, "Verification should pass for a valid proof");
    }

    #[test]
    fn test_sumcheck_consistency() {
        
        let circuit = create_sample_circuit();
        let inputs = sample_inputs();
        let mut prover = GKRProver::new(circuit.clone(), inputs.clone());
        let proof = prover.prove();
        assert!(!proof.is_empty(), "Proof should contain sumcheck data");
    }
}



