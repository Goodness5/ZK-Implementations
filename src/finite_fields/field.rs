use ark_ff::PrimeField;
use ark_std::{One, UniformRand};
// use ark_test_curves::bls12_381::Fq2 as F;

use super::interface::FFPolynomialTrait;


#[derive(Clone, Debug)]
pub struct Polynomial <T: PrimeField>{
    coefficient: Vec<(T, T)>,
    degree: T::BigInt

}

impl<T: PrimeField> FFPolynomialTrait<T> for Polynomial<T> {

    fn degree(&self) -> usize {
        self.coefficient.len() -1
        
    }


    fn evaluate(&self, x: T) -> T {
        self.coefficient
            .iter()
            // The pow func takes a generic with type of u64 so this into_bigint is used
            .map(|&(coef, exp)| coef * x.pow(exp.into_bigint())) 
            .sum()
    }


 fn multiply_field_polynomials<S: PrimeField>(a: &[S], b: &[S]) -> Vec<S> {
        let mut result = vec![S::zero(); a.len() + b.len() - 1];
    
        for (i, &a_coef) in a.iter().enumerate() {
            for (j, &b_coef) in b.iter().enumerate() {
                result[i + j] += a_coef * b_coef;
            }
        }
    
        result
    }
    
    fn interpolate(&self) -> Vec<T> {
        let n = self.coefficient.len();
        let mut result = vec![T::zero(); n]; // Initialize coefficients of the result polynomial to zero.
    
        for i in 0..n {
            let (x_i, y_i) = self.coefficient[i];
            let mut lagrange_polynomial = vec![T::one()]; // Start with the constant polynomial 1.
    
            for j in 0..n {
                if i != j {
                    let (x_j, _) = self.coefficient[j];
    
                    // Compute Lagrange basis: (x - x_j) / (x_i - x_j).
                    let denominator = x_i - x_j; // x_i - x_j
                    let denominator_inv = denominator.inverse().expect("Denominator not invertible!"); // (x_i - x_j)^(-1)
    
                    // Multiply the current Lagrange polynomial by (x - x_j).
                    lagrange_polynomial = Self::multiply_field_polynomials(
                        &lagrange_polynomial,
                        &[x_j.neg(), T::one()],
                    );
    
                    lagrange_polynomial
                        .iter_mut()
                        .for_each(|coef| *coef *= denominator_inv);
                }
            }
    
            // Scale the Lagrange polynomial by y_i and add it to the result.
            for (k, coef) in lagrange_polynomial.iter().enumerate() {
                result[k] += *coef * y_i;
            }
        }
    
        result
    }
    
}