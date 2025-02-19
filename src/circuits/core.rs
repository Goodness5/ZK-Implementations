use ark_bn254::Fr;
use ark_ff::{BigInteger, Field, PrimeField, Zero, One};
use sha3::{Keccak256, Digest};
use std::marker::PhantomData;

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

pub struct Circuit<T: PrimeField> {
    layers: Vec<Vec<Gate>>,
    _phantom: PhantomData<T>,
}

struct LayerProof {
    sumcheck_rounds: Vec<SumcheckRound>,
    final_evaluation: Fr,
}

struct SumcheckRound {
    coefficients: Vec<Fr>,
    random_challenge: Fr,
}

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

impl<T: PrimeField> Circuit<T> {
    pub fn new(layers: Vec<Vec<Gate>>) -> Self {
        Circuit { layers, _phantom: PhantomData }
    }

    pub fn build_binary_tree_circuit(num_layers: usize, gate_type: GateType) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        let mut current_size = 2usize.pow(num_layers as u32);
        
        for _ in 0..num_layers {
            current_size /= 2;
            layers.push(Self::create_binary_tree_layer(current_size * 2, gate_type));
        }
        
        Circuit::new(layers)
    }

    fn create_binary_tree_layer(current_size: usize, gate_type: GateType) -> Vec<Gate> {
        let mut layer = Vec::with_capacity(current_size / 2);
        for i in 0..(current_size / 2) {
            layer.push(Gate {
                left: 2 * i,
                right: 2 * i + 1,
                ty: gate_type,
            });
        }
        layer
    }

    pub fn evaluate(&self, initial_inputs: Vec<T>) -> T {
        let mut current_inputs = initial_inputs;
        for layer in &self.layers {
            current_inputs = Self::evaluate_layer(current_inputs, layer);
        }
        current_inputs[0]
    }

    fn evaluate_layer(inputs: Vec<T>, layer: &[Gate]) -> Vec<T> {
        layer.iter()
            .map(|gate| Self::evaluate_gate(&inputs, gate))
            .collect()
    }

    fn evaluate_gate(inputs: &[T], gate: &Gate) -> T {
        let left = inputs[gate.left];
        let right = inputs[gate.right];
        match gate.ty {
            GateType::Add => left + right,
            GateType::Mul => left * right,
        }
    }
}

impl Circuit<Fr> {
    pub fn prove(&self, inputs: &[Fr]) -> (Vec<LayerProof>, Fr) {
        let mut transcript = Transcript::new(Keccak256::new());
        let mut current_values = inputs.to_vec();
        let mut proofs = Vec::new();

        for layer in &self.layers {
            let outputs = Self::evaluate_layer(current_values.clone(), layer);
            let (layer_proof, _) = self.prove_layer(&current_values, &outputs, layer, &mut transcript);
            proofs.push(layer_proof);
            current_values = outputs;
        }

        (proofs, current_values[0])
    }

    fn prove_layer(
        &self,
        inputs: &[Fr],
        outputs: &[Fr],
        layer: &[Gate],
        transcript: &mut Transcript<Keccak256, Fr>
    ) -> (LayerProof, Fr) {
        let mut rounds = Vec::new();
        let mut current_poly = self.layer_to_poly(inputs, outputs, layer);
        let mut final_challenge = Fr::zero();

        for _ in 0..current_poly.len() {
            let coeffs = current_poly.clone();
            transcript.absorb(&Self::serialize_coefficients(&coeffs));
            
            let challenge = transcript.squeeze();
            current_poly = self.reduce_polynomial(&coeffs, challenge);
            
            rounds.push(SumcheckRound {
                coefficients: coeffs,
                random_challenge: challenge,
            });
            final_challenge = challenge;
        }

        (LayerProof { sumcheck_rounds: rounds, final_evaluation: current_poly[0] }, final_challenge)
    }

    // fn layer_to_poly(&self, inputs: &[Fr], outputs: &[Fr], layer: &[Gate]) -> Vec<Fr> {
    //     let mut poly = vec![Fr::zero(); inputs.len() + outputs.len()];
        
    //     for (i, gate) in layer.iter().enumerate() {
    //         let input_idx = 2 * i;
    //         match gate.ty {
    //             GateType::Add => {
    //                 poly[input_idx] = Fr::one();
    //                 poly[input_idx + 1] = Fr::one();
    //                 poly[inputs.len() + i] = -Fr::one();
    //             }
    //             GateType::Mul => {
    //                 poly[input_idx] = inputs[gate.right];
    //                 poly[input_idx + 1] = inputs[gate.left];
    //                 poly[inputs.len() + i] = -Fr::one();
    //             }
    //         }
    //     }
    //     poly
    // }

    fn reduce_polynomial(&self, poly: &[Fr], r: Fr) -> Vec<Fr> {
        let mut reduced = Vec::with_capacity(poly.len() / 2);
        for i in 0..(poly.len() / 2) {
            reduced.push(poly[2 * i] + r * poly[2 * i + 1]);
        }
        reduced
    }

    fn serialize_coefficients(coeffs: &[Fr]) -> Vec<u8> {
        coeffs.iter()
            .flat_map(|c| c.into_bigint().to_bytes_be())
            .collect()
    }

    pub fn verify(&self, proofs: &[LayerProof], final_output: Fr, transcript: &mut Transcript<Keccak256, Fr>) -> bool {
        let mut current_eval = final_output;

        for (layer_proof, layer) in proofs.iter().zip(&self.layers).rev() {
            if !self.verify_layer(layer_proof, current_eval, transcript) {
                return false;
            }
            current_eval = layer_proof.sumcheck_rounds[0].coefficients[0];
        }

        current_eval == Fr::zero()
    }

    fn verify_layer(
        &self,
        proof: &LayerProof,
        expected_eval: Fr,
        transcript: &mut Transcript<Keccak256, Fr>
    ) -> bool {
        let mut eval = expected_eval;
        
        for round in proof.sumcheck_rounds.iter().rev() {
            let mut local_transcript = Transcript::new(Keccak256::new());
            local_transcript.absorb(&Self::serialize_coefficients(&round.coefficients));
            let challenge:Fr = local_transcript.squeeze();
            
            if challenge != round.random_challenge {
                return false;
            }
            
            let mut reconstructed = Fr::zero();
            for i in 0..(round.coefficients.len() / 2) {
                reconstructed += round.coefficients[2 * i] + challenge * round.coefficients[2 * i + 1];
            }
            
            if reconstructed != eval {
                return false;
            }
            
            eval = reconstructed;
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{MontConfig, Fp64};

    // #[derive(MontConfig)]
    // #[modulus = "17"]
    // #[generator = "3"]
    // struct FqConfig;
    // type Fq = Fp64<ark_ff::MontBackend<FqConfig, 1>>;

    // #[test]
    // fn test_basic_proof() {
    //     let layers = vec![
    //         vec![
    //             Gate { left: 0, right: 1, ty: GateType::Add },
    //             Gate { left: 2, right: 3, ty: GateType::Add },
    //         ],
    //         vec![
    //             Gate { left: 0, right: 1, ty: GateType::Add },
    //         ],
    //     ];
        
    //     let circuit = Circuit::<Fr>::new(layers);
    //     let inputs = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(4)];
        
    //     let (proofs, output) = circuit.prove(&inputs);
    //     let mut transcript = Transcript::new(Keccak256::new());
    //     assert!(circuit.verify(&proofs, output, &mut transcript));
    //     assert_eq!(output, Fr::from(10)); // (1+2) + (3+4) = 10
    // }
}