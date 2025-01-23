#[derive(Debug, Clone)]
pub struct MultilinearPolynomial<T> {
    pub coefficients: Vec<T>, 
}


impl<T> MultilinearPolynomialTrait<T> for MultilinearPolynomial<T>{
    fn partial_evaluate(&self, x: T) -> T {
        // Evaluate the polynomial at the single point `x`.
        let mut result = T::default();
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            let term = coeff * x.pow(i as u32); 
            result = result + term;
        }
        result
    }

    fn partial_evaluate_vec(&self, x: Vec<T>) -> T {
        // Evaluate the polynomial at a set of points.
        let mut result = T::default();
        for (i, &coeff) in self.coefficients.iter().enumerate() {
            let mut term = coeff;
            for j in 0..x.len() {
                term = term * x[j]; 
            }
            result = result + term;
        }
        result
    }

    fn degree(&self) -> usize {
        
        self.coefficients.len() - 1
    }

    fn normal_interpolation(
        &self,
        points: Vec<Vec<T>>,  
        values: Vec<T>,       
             
    ) -> MultilinearPolynomial<T> {

        // Base case: if there's only one point, the polynomial is constant.
        if points.len() == 1 {
            return MultilinearPolynomial {
                coefficients: vec![values[0]; 1 << degree], 
            };
        }

        // Split points and values into two halves.
        let mid = points.len() / 2;
        let left_points = points[0..mid].to_vec();
        let right_points = points[mid..].to_vec();
        let left_values = values[0..mid].to_vec();
        let right_values = values[mid..].to_vec();

        // Recursively interpolate the left and right halves.
        let left_poly = self.normal_interpolation(left_points, left_values, degree - 1);
        let right_poly = self.normal_interpolation(right_points, right_values, degree - 1);

        // Combine the left and right polynomials into a single polynomial.
        let mut coefficients = vec![T::default(); 1 << degree];
        for i in 0..coefficients.len() {
            // Determine the variable (x_i) for this step based on the parity of i.
            let variable = if i < (coefficients.len() / 2) { T::default() } else { T::default() + T::default() - T::default() };

            // Combine the coefficients.
            coefficients[i] = left_poly.coefficients[i % (coefficients.len() / 2)] * (T::default() + T::default() - variable)
                + right_poly.coefficients[i % (coefficients.len() / 2)] * variable;
        }

        MultilinearPolynomial { coefficients }
    }
    

    fn eval_and_interpolate(&self, x: Vec<T>, coeffs: &mut Vec<T>, index: usize) -> T {
        if x.is_empty() {
            return self.coefficients[index]; 
        }

        // Split the vector: `x = [x_0, x_rest]`
        let (first_x, rest_x) = (x[0], &x[1..]);

        // Recursive calls: interpolate and evaluate on subtrees.
        // let eval_left = self.eval_and_interpolate(rest_x.to_vec(), coeffs, index);
        // let eval_right = self.eval_and_interpolate(rest_x.to_vec(), coeffs, index + (1 << (rest_x.len()))); 
        // Straight-line evaluation and interpolation (using `x_0`).
        // let result = eval_left * (T::default() + T::default() - first_x) + eval_right * first_x;

        // Interpolate by storing results (optional, could be optimized).
        coeffs[index] = result;

        result
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_partial_evaluate() {
    //     let poly = MultilinearPolynomial { coefficients: vec![1, 2, 3] }; 
    //     assert_eq!(poly.partial_evaluate(2), 1 + 2 * 2 + 3 * 2 * 2); 
    // }

    // #[test]
    // fn test_partial_evaluate_vec() {
    //     let poly = MultilinearPolynomial { coefficients: vec![1, 2, 3] }; 
    //     assert_eq!(poly.partial_evaluate_vec(vec![1, 2]), 1 + 2 * 1 * 2 + 3 * 1 * 2); 
    // }

    // #[test]
    // fn test_degree() {
    //     let poly = MultilinearPolynomial { coefficients: vec![1, 0, 3] };
    //     assert_eq!(poly.degree(), 2); 
    // }
}
