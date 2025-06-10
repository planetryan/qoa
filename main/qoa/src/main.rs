use num_bigint::BigUint;
use num_traits::{Zero, One};

fn main() {
    let mut a: BigUint = Zero::zero();
    let mut b: BigUint = One::one();

    let mut index = 1;

    // Store
    let mut sequence = vec![a.clone(), b.clone()];

    println!("F(0) = {}", a);
    println!("F(1) = {}", b);

    while b.to_string().len() < 10_000 {
        let next = &a + &b;
        a = b;
        b = next;

        index += 1;
        println!("F({}) = {}", index, &b);
        sequence.push(b.clone());
    }

    println!("\nReached Fibonacci number with 10,000 digits at index F({})", index);

    // verify sequence
    println!("\nVerifying sequence...");
    for i in 2..sequence.len() {
        if &sequence[i] != &sequence[i - 1] + &sequence[i - 2] {
            eprintln!("Verification failed at index {}", i);
            return;
        }
    }

    println!("All Fibonacci numbers verified up to F({})!", index);
}
