mod polynomials;

use zk_implementations::polynomials::univalent::{self, PolynomialTrait};

// use crate::polynomials::univalent;

fn main() {
    let polynomial = univalent::Polynomial::new(vec![(1, 2), (3, 18)]); // Example terms
    println!("{}", polynomial.represent());
    print!("{}", polynomial.evaluate(5))
}