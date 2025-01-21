use ark_ff::PrimeField;

pub trait FFPolynomialTrait<T: PrimeField> {
    fn evaluate(&self, x: T) -> T;
    fn degree(&self) -> usize;
    fn interpolate(&self) -> Vec<T>;
    fn multiply_field_polynomials<S: PrimeField>(a: &[S], b: &[S]) -> Vec<S>;
}