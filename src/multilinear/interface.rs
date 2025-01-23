use ark_ff::PrimeField;

// pub trait MultilinearPolynomialTrait<T: PrimeField> {
//     fn evaluate(&self, x: T) -> T;
//     fn degree(&self) -> usize;
//     fn interpolate(&self) -> Vec<T>;
//     fn multiply_field_polynomials<S: PrimeField>(a: &[S], b: &[S]) -> Vec<S>;
// }

pub trait MultilinearPolynomialTrait<T> {
    // Partially evaluates the polynomial at a single point.
    fn partial_evaluate(&self, x: T) -> T;

    // Partially evaluates the polynomial at multiple points.
    fn partial_evaluate_vec(&self, x: Vec<T>) -> T;

    // Returns the degree of the polynomial.
    fn degree(&self) -> usize;

    // Given a polynomial points this function interpolates and evaluates at given point x simultaneously
    fn eval_and_interpolate(&self, x: Vec<T>, coeffs: &mut Vec<T>, index: usize) -> T;


    fn normal_interpolation(&self, points: Vec<Vec<T>>, values: Vec<T>, degree: usize) -> MultilinearPolynomial<T>;
}
