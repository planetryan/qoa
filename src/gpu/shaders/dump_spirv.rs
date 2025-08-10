use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("usage: {} <input_spv> <output_spv>", args[0]);
        std::process::exit(1);
    }

    let path_in = Path::new(&args[1]);
    let path_out = Path::new(&args[2]);

    let mut file_in = File::open(&path_in)?;
    let mut buffer = Vec::new();
    file_in.read_to_end(&mut buffer)?;

    let mut file_out = File::create(&path_out)?;
    file_out.write_all(&buffer)?;

    println!("dumped {} to {} successfully.", args[1], args[2]);
    Ok(())
}
