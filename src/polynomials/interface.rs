use super::univalent::Polynomial;



pub trait PolynomialTrait {

    
    
    fn evaluate(&self, x: isize) -> isize;
    fn degree(&self) -> usize;
    fn interpolate(&self) -> String;
   
}