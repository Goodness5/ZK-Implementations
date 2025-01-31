use ark_ff::{BigInteger, PrimeField};
use sha3::{Keccak256, Digest};
use std::marker::PhantomData;

trait MultilinearPolynomial<F: PrimeField> {
    fn evaluate(&self, point: &[F]) -> F;
    fn num_vars(&self) -> usize;
}

struct Transcript<K: HashTrait, F: PrimeField> {
    hash_function: K,
    _field: PhantomData<F>,
}

impl<K: HashTrait, F: PrimeField> Transcript<K, F> {
    fn new(hash_function: K) -> Self {
        Self {
            hash_function,
            _field: PhantomData,
        }
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

struct Proof<F: PrimeField> {
    rounds: Vec<(F, F)>,
    final_evaluation: F,
}


fn prove<F: PrimeField, P: MultilinearPolynomial<F>>(
    polynomial: &P,
    sum: F,
    transcript: &mut Transcript<impl HashTrait, F>,
) -> Result<Proof<F>, &'static str> {
    let num_vars = polynomial.num_vars();
    let mut rounds = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);

    // Absorb the sum into the transcript
    let sum_bytes = sum.into_bigint().to_bytes_be();
    transcript.absorb(&sum_bytes);

    let mut current_sum = sum;

    for round in 0..num_vars {
        // Compute gi(0) and gi(1)
        let gi0 = compute_partial_sum(polynomial, &challenges, round, F::zero())?;
        let gi1 = compute_partial_sum(polynomial, &challenges, round, F::one())?;

        if gi0 + gi1 != current_sum {
            return Err("Sum check failed");
        }

        rounds.push((gi0, gi1));

        // Absorb gi0 and gi1 into the transcript
        transcript.absorb(&gi0.into_bigint().to_bytes_be());
        transcript.absorb(&gi1.into_bigint().to_bytes_be());

        // Generate challenge
        let challenge = transcript.squeeze();
        challenges.push(challenge);

        // Update current_sum for next round
        current_sum = gi0 + (gi1 - gi0) * challenge;
    }

    Ok(Proof {
        rounds,
        final_evaluation: current_sum,
    })
}

fn compute_partial_sum<F: PrimeField, P: MultilinearPolynomial<F>>(
    polynomial: &P,
    challenges: &[F],
    current_var: usize,
    x: F,
) -> Result<F, &'static str> {
    let num_vars = polynomial.num_vars();
    let mut assignment = challenges.to_vec();
    assignment.resize(current_var + 1, F::zero());
    assignment[current_var] = x;

    let remaining_vars = num_vars - (current_var + 1);
    let mut sum = F::zero();

    for bits in 0..(1 << remaining_vars) {
        let mut full_assignment = assignment.clone();
        for i in 0..remaining_vars {
            let bit = (bits >> i) & 1;
            full_assignment.push(if bit == 1 { F::one() } else { F::zero() });
        }
        sum += polynomial.evaluate(&full_assignment);
    }

    Ok(sum)
}

fn verify<F: PrimeField, P: MultilinearPolynomial<F>>(
    polynomial: &P,
    claimed_sum: F,
    proof: &Proof<F>,
    transcript: &mut Transcript<impl HashTrait, F>,
) -> Result<bool, &'static str> {
    let num_vars = polynomial.num_vars();
    if proof.rounds.len() != num_vars {
        return Ok(false);
    }

    // Absorb the sum into the transcript
    let sum_bytes = claimed_sum.into_bigint().to_bytes_be();
    transcript.absorb(&sum_bytes);

    let mut current_sum = claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for (round, &(gi0, gi1)) in proof.rounds.iter().enumerate() {
        if gi0 + gi1 != current_sum {
            return Ok(false);
        }

        transcript.absorb(&gi0.into_bigint().to_bytes_be());
        transcript.absorb(&gi1.into_bigint().to_bytes_be());

        let challenge = transcript.squeeze();
        challenges.push(challenge);

        current_sum = gi0 + (gi1 - gi0) * challenge;
    }

    // Verify final evaluation
    let full_assignment = challenges;
    let expected = polynomial.evaluate(&full_assignment);
    Ok(proof.final_evaluation == expected)
}


struct SimplePoly;

impl<F: PrimeField> MultilinearPolynomial<F> for SimplePoly {
    fn evaluate(&self, point: &[F]) -> F {
        let x1 = point.get(0).cloned().unwrap_or(F::zero());
        let x2 = point.get(1).cloned().unwrap_or(F::zero());
        let x3 = point.get(2).cloned().unwrap_or(F::zero());
        x1 * x2 + x3
    }

    fn num_vars(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_sumcheck() {
        let poly = SimplePoly;
        let sum = Fr::from(6);

        // Prover
        let mut prover_transcript = Transcript::new(Keccak256::new());
        let proof = prove(&poly, sum, &mut prover_transcript).unwrap();

        // Verifier
        let mut verifier_transcript = Transcript::new(Keccak256::new());
        let is_valid = verify(&poly, sum, &proof, &mut verifier_transcript).unwrap();
        
        assert!(is_valid);
    }
}