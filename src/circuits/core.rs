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

impl <T: PrimeField> Gate <T> {
    fn evaluate(&self) -> T {
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


    }




