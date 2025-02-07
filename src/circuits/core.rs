use ark_ff::{BigInteger, PrimeField};
use sha3::{Keccak256, Digest};
use std::{marker::PhantomData, vec};


struct Gate<T> {
    pub left: T,
    pub right: T,
    pub output: T,
    pub ty: GateType,
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    Add,
    Mul,
}

struct circuit <T>{
    layers: u64,
    inputs: Vec<T>,
    gates: Vec<Gate<T>>,

}

impl <T: PrimeField> Gate <T> {
    fn evaluate_gate(&self) -> T {
        match self.ty {
            GateType::Add => self.left + self.right,
            GateType::Mul => self.left * self.right,
        }
    }

    

    fn new(left: T, right: T, output: T, ty: GateType) -> Self {
        Self {
            left,
            right,
            output,
            ty,
        }
    }



    }

impl <T: PrimeField> circuit <T> {
    fn new(layers: u64, inputs: Vec<T>, gates: Vec<Gate<T>>) -> Self {
        Self {
            layers,
            inputs,
            gates,
        }
    }
    fn construct_circuit(&self, layers: u64, inputs: Vec<T>) -> Vec<Vec<Gate<T>>> {
        // let mut circuit = vec![];
        // let mut current_layer = vec![self.clone()];
        // let mut current_input = inputs;
        // for _ in 0..layers {
        //     for gate in current_layer {
        //         current_input.push(gate.evaluat_gate(current_input));
        //     }
        //     circuit.push(current_layer);
        // }
        // circuit
        todo!()
    }


    fn evaluate_circuit(&self, circuit: Vec<Vec<Gate<T>>>, inputs: Vec<T>) -> T {
        // let mut current_input = inputs;
        // for layer in circuit {
        //     let mut next_input = vec![];
        //     for gate in layer {
        //         next_input.push(gate.evaluate_gate(current_input));
        //     }
        //     current_input = next_input;
        // }
        // current_input[0]
        todo!()
    }

}




