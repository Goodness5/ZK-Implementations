use rand::Rng;
use std::io;

/// Interpolates the given points to calculate polynomial coefficients.
pub fn interpolate_to_coefficients(points: &[(f64, f64)]) -> Vec<f64> {
    let n = points.len();
    let mut coefficients = vec![0.0; n];

    for i in 0..n {
        let (x_i, y_i) = points[i];
        let mut term_coefficients = vec![1.0];

        for j in 0..n {
            if i != j {
                let (x_j, _) = points[j];
                term_coefficients = multiply_polynomials(&term_coefficients, &[-x_j, 1.0]);
                term_coefficients = term_coefficients.iter().map(|c| c / (x_i - x_j)).collect();
            }
        }

        for k in 0..term_coefficients.len() {
            coefficients[k] += y_i * term_coefficients[k];
        }
    }

    coefficients
}


pub fn multiply_polynomials(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; p1.len() + p2.len() - 1];

    for (i, &c1) in p1.iter().enumerate() {
        for (j, &c2) in p2.iter().enumerate() {
            result[i + j] += c1 * c2;
        }
    }

    result
}


pub fn evaluate_polynomial(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, &coef| acc * x + coef)
}


pub fn generate_points(threshold: f64) -> (Vec<(f64, f64)>, f64) {
    let mut rng = rand::thread_rng();

    
    let secret: f64 = rng.gen_range(1.0..threshold);

    
    let a1: f64 = rng.gen_range(1.0..threshold);
    let a2: f64 = rng.gen_range(1.0..threshold);

    
    let polynomial = move |x: f64| a2 * x * x + a1 * x + secret;

    
    let points: Vec<(f64, f64)> = (1..=4)
        .map(|_| {
            let x = rng.gen_range(1.0..10.0);
            let y = polynomial(x);
            (x, y)
        })
        .collect();

    (points, secret)
}


pub fn secret_sharing_scheme() {
    let mut input = String::new();

    println!("Welcome to Shamir's Secret Sharing Scheme!");
    println!("Please enter a positive threshold value:");

    // Get user input for the threshold
    io::stdin().read_line(&mut input).unwrap();
    let threshold: f64 = match input.trim().parse() {
        Ok(value) if value > 0.0 => value,
        _ => {
            println!("Invalid threshold. Please enter a positive number.");
            return;
        }
    };

    
    let (points, secret) = generate_points(threshold);

    println!("\nGenerated Secret: {:.2}", secret);
    println!("Generated Points:");
    for (x, y) in &points {
        println!("  (x: {:.2}, y: {:.2})", x, y);
    }

    // Interpolate the polynomial
    let coefficients = interpolate_to_coefficients(&points);
    println!("\nInterpolated Polynomial Coefficients: {:?}", coefficients);

    // Verify the reconstruction of the secret
    let reconstructed_secret = evaluate_polynomial(&coefficients, 0.0);
    println!(
        "\nReconstructed Secret (evaluated at x=0): {:.2}",
        reconstructed_secret
    );

    // Check if the reconstructed secret matches
    if (secret - reconstructed_secret).abs() < 1e-6 {
        println!("\nSuccess! The secret has been reconstructed correctly.");
    } else {
        println!("\nError: The reconstructed secret does not match the original.");
    }
}

