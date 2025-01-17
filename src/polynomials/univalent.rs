pub use crate::polynomials::interface::PolynomialTrait;
   
    

    
    pub struct Polynomial {
        
        terms: Vec<(isize, isize)>,
    }
    
    impl Polynomial {
        pub fn new(terms: Vec<(isize, isize)>) -> Polynomial {
            Polynomial { terms }
        }
        
        pub fn init_poly(&mut self, terms: Vec<(isize, isize)>) {
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

    
        fn evaluate(&self, x: isize) -> isize{
            
            self.terms.iter()
            .map(|&(coef, pow)| coef * x.pow(pow.try_into().unwrap()))
            .sum() // Sum all term values
        }

        fn degree(&self) -> usize {
            
        return  0;

        }

        
            fn interpolate(&self) -> String {
                let mut result_terms = Vec::new();
        
                for (i, &(xi, yi)) in self.terms.iter().enumerate() {
                    let mut term_poly = vec![(yi, 0)]; // Start with the constant term y_i
        
                    // Multiply by (x - xj)/(xi - xj) for all j != i
                    for (j, &(xj, _)) in self.terms.iter().enumerate() {
                        if i != j {
                            // For each term in term_poly, we need to multiply by (x - xj) and divide by (xi - xj)
                            let mut new_term_poly = Vec::new();
        
                            // Multiply by (x - xj)
                            for &(coef, pow) in &term_poly {
                                new_term_poly.push((coef, pow + 1));
                                new_term_poly.push((-coef * xj, pow));
                            }
        
                            // Divide all terms by (xi - xj)
                            let xi_xj_diff = (xi - xj).abs(); // Ensure positive division
                            term_poly = new_term_poly.into_iter()
                                .map(|(coef, pow)| (coef / xi_xj_diff, pow))
                                .collect();
                        }
                    }
        
                    // Add the terms from this i-th term polynomial to the result polynomial
                    for (coef, pow) in term_poly {
                        if let Some(pos) = result_terms.iter_mut().find(|&&mut (c, p)| p == pow) {
                            pos.0 += coef; // Add coefficients of like powers
                        } else {
                            result_terms.push((coef, pow));
                        }
                    }
                }
        
                Polynomial::new(result_terms).represent()
            }
        }