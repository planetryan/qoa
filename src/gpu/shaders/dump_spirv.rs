use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

fn main() -> std::io::Result<()> {
    let path_in = Path::new("hadamard.spv");
    let mut file_in = File::open(&path_in)?;
    let mut buffer = Vec::new();
    file_in.read_to_end(&mut buffer)?;

    let path_out = Path::new("dumped.spv");
    let mut file_out = File::create(&path_out)?;
    file_out.write_all(&buffer)?;

    println!("Dumped hadamard.spv to dumped.spv successfully.");
    Ok(())
}
