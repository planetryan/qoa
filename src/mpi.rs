// fitxategi honek MPI taldea ematen du (`qoa run --mpi`, `mpi` ezaugarriarekin konpilatuta): src/mpi_shim.c-ko
// funtzioak `Team` trait-aren atzean. prozesu bakoitzak `World` bakarra du; erortzean MPI amaitzen da.
// unsafe hemen: MPI liburutegiari deiak (kanpoko C funtzioak)
#![allow(unsafe_code)]

use crate::team::Team;

unsafe extern "C" {
    fn qoa_mpi_init() -> i32;
    fn qoa_mpi_finalize();
    fn qoa_mpi_rank() -> i32;
    fn qoa_mpi_size() -> i32;
    fn qoa_mpi_barrier();
    fn qoa_mpi_send(p: *const u8, n: u64, to: i32);
    fn qoa_mpi_recv_len(from: i32) -> u64;
    fn qoa_mpi_recv_data(p: *mut u8, n: u64, from: i32);
    fn qoa_mpi_exchange(s: *const u8, r: *mut u8, n: u64, partner: i32);
}

pub struct World {
    rank: usize,
    size: usize,
}

impl World {
    pub fn init() -> Result<World, String> {
        // SAFETY: shim-aren funtzioak MPI-ren dei zuzenak dira; init behin, beste ezer baino lehen
        unsafe {
            if qoa_mpi_init() != 0 {
                return Err(crate::ts!("MPI did not start", "MPI ez da abiarazi").into());
            }
            Ok(World {
                rank: qoa_mpi_rank() as usize,
                size: qoa_mpi_size() as usize,
            })
        }
    }
}

impl Drop for World {
    fn drop(&mut self) {
        // SAFETY: MPI hasieratuta dago; finalize-k bere burua egiaztatzen du
        unsafe { qoa_mpi_finalize() }
    }
}

impl Team for World {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn send(&self, to: usize, data: &[u8]) {
        // SAFETY: buffer baliozkoa, `data.len()` byte
        unsafe { qoa_mpi_send(data.as_ptr(), data.len() as u64, to as i32) }
    }

    fn recv(&self, from: usize) -> Vec<u8> {
        // SAFETY: luzera lehenik, gero hainbeste byte buffer berrian
        unsafe {
            let n = qoa_mpi_recv_len(from as i32) as usize;
            let mut out = vec![0u8; n];
            qoa_mpi_recv_data(out.as_mut_ptr(), n as u64, from as i32);
            out
        }
    }

    fn exchange(&self, partner: usize, send: &[u8], recv: &mut [u8]) {
        assert_eq!(send.len(), recv.len());
        // SAFETY: bi buffer baliozko, luzera berekoak
        unsafe {
            qoa_mpi_exchange(
                send.as_ptr(),
                recv.as_mut_ptr(),
                send.len() as u64,
                partner as i32,
            )
        }
    }

    fn barrier(&self) {
        // SAFETY: dei kolektibo zuzena
        unsafe { qoa_mpi_barrier() }
    }
}
