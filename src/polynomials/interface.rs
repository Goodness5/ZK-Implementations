use super::univalent::Polynomial;



pub trait PolynomialTrait {

    
    
    fn evaluate(&self, x: usize) -> usize;
    fn degree(&self) -> usize;
    fn interpolate(&self) -> Polynomial;
    // fn evaluate(x: Vec<usize, usize>) -> Polynomial;
}