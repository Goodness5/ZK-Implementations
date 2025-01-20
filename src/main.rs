mod polynomials;
mod shamirs_secret_sharing;



use zk_implementations::{polynomials::univalent::{self, PolynomialTrait}, shamirs_secret_sharing::core};

// use crate::polynomials::univalent;

fn main() {
    let polynomial = univalent::Polynomial::new(vec![(1, 2), (3, 18), (6, 6), (-1, 4)]);
    println!("{}", polynomial.represent());
    print!("{}", polynomial.evaluate(5));
    print!("{}", polynomial.interpolate());
    core::secret_sharing_scheme();
}