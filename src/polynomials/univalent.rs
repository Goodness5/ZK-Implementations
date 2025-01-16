pub use crate::polynomials::interface::PolynomialTrait;
   
    

    
    pub struct Polynomial {
        
        terms: Vec<(usize, usize)>,
    }
    
    impl Polynomial {
        pub fn new(terms: Vec<(usize, usize)>) -> Polynomial {
            Polynomial { terms }
        }
        
        fn init_poly(&mut self, terms: Vec<(usize, usize)>) {
            self.terms = terms;
            
            self.terms.sort_by(|a, b| b.1.cmp(&a.1));
        }

       pub fn represent(&self) -> String {
            self.terms.iter()
                .map(|&(coef, pow)| {
                    if coef == 1 && pow == 0 {
                        "1".to_string()  
                    } else if coef == 1 {
                        format!("x^{}", pow)  
                    } else {
                        format!("{}x^{}", coef, pow)  
                    }
                })
                .collect::<Vec<_>>().join(" + ")
        }
        
    }
    
    


    impl PolynomialTrait for Polynomial {

    
        fn evaluate(&self, x: usize) -> usize{
            
            self.terms.iter()
            .map(|&(coef, pow)| coef * x.pow(pow.try_into().unwrap()))
            .sum() // Sum all term values
        }

        fn degree(&self) -> usize {
            
        return  0;

        }

        fn interpolate(&self) -> Polynomial {
            // amplify the vectors
            // for i in self.terms
            return Polynomial::new([].to_vec());
        }
    }