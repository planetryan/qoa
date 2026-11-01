// fitxategi honek prozesu-talde bat deskribatzen du (`qoa run --mpi`): rank bakoitzak bere zatia kalkulatzen
// du eta emaitzak 0. rank-era biltzen dira. `Team` trait-ak behar diren eragiketak ematen ditu (bidali,
// jaso, trukatu, hesia), eta haien gainean bilketa eta zabalkundea.
//
// bi inplementazio: `LocalTeam` (hari batzuk prozesu bakarrean, kanalekin: probetarako, MPI gabe) eta
// `crate::mpi::World` (benetako MPI, liburutegia exekuzioan kargatuta, src/mpi.rs).

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Barrier, Mutex};

pub trait Team: Send + Sync {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    // datuak `to`-ri (blokeatu gabe, jasotzailea prest egon arte bufferrean)
    fn send(&self, to: usize, data: &[u8]);
    // `from`-ek bidalitako hurrengo mezua
    fn recv(&self, from: usize) -> Vec<u8>;
    // bikote batekin tamaina bereko buffer bat trukatu (biek aldi berean deitzen dute)
    fn exchange(&self, partner: usize, send: &[u8], recv: &mut [u8]);
    fn barrier(&self);
}

impl std::fmt::Debug for dyn Team {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Team(rank {} of {})", self.rank(), self.size())
    }
}

impl dyn Team {
    // rank guztien datuak 0. rank-ean, rank-en ordenan (besteek None)
    pub fn gather(&self, data: Vec<u8>) -> Option<Vec<Vec<u8>>> {
        if self.rank() == 0 {
            let mut all = vec![data];
            for r in 1..self.size() {
                all.push(self.recv(r));
            }
            Some(all)
        } else {
            self.send(0, &data);
            None
        }
    }

    // 0. rank-aren datuak guztiei
    pub fn broadcast(&self, data: Vec<u8>) -> Vec<u8> {
        if self.rank() == 0 {
            for r in 1..self.size() {
                self.send(r, &data);
            }
            data
        } else {
            self.recv(0)
        }
    }

    // rank bakoitzaren f64 zerrenda guztiei, rank-en ordenan (baturak ordena finkoan egin daitezen)
    pub fn allgather_f64(&self, xs: &[f64]) -> Vec<Vec<f64>> {
        let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        let all = match self.gather(bytes) {
            Some(all) => {
                let joined = encode_list(&all);
                self.broadcast(joined)
            }
            None => self.broadcast(Vec::new()),
        };
        decode_list(&all)
            .into_iter()
            .map(|b| {
                b.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| f64::from_le_bytes(*c))
                    .collect()
            })
            .collect()
    }
}

// buffer-zerrenda bat byte bakarretan: luzera (u64) eta edukia, bakoitzeko
pub fn encode_list(items: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for it in items {
        out.extend((it.len() as u64).to_le_bytes());
        out.extend(it);
    }
    out
}

pub fn decode_list(mut bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while bytes.len() >= 8 {
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        out.push(bytes[8..8 + n].to_vec());
        bytes = &bytes[8 + n..];
    }
    out
}

// prozesu bakarreko taldea: rank bakoitza hari bat da, eta mezuak kanaletatik doaz
pub struct LocalTeam {
    rank: usize,
    size: usize,
    to: Vec<Sender<Vec<u8>>>,
    from: Vec<Mutex<Receiver<Vec<u8>>>>,
    barrier: Arc<Barrier>,
}

impl LocalTeam {
    // `size` kide, bakoitza bere harian erabiltzeko
    pub fn new(size: usize) -> Vec<LocalTeam> {
        // kanalak[nondik][nora]
        let mut senders: Vec<Vec<Option<Sender<Vec<u8>>>>> =
            (0..size).map(|_| vec![None; size]).collect();
        let mut receivers: Vec<Vec<Option<Receiver<Vec<u8>>>>> = (0..size)
            .map(|_| (0..size).map(|_| None).collect())
            .collect();
        for (a, row) in senders.iter_mut().enumerate() {
            for (b, slot) in row.iter_mut().enumerate() {
                let (tx, rx) = channel();
                *slot = Some(tx);
                receivers[b][a] = Some(rx);
            }
        }
        let barrier = Arc::new(Barrier::new(size));
        senders
            .into_iter()
            .zip(receivers)
            .enumerate()
            .map(|(rank, (to, from))| LocalTeam {
                rank,
                size,
                to: to.into_iter().map(Option::unwrap).collect(),
                from: from.into_iter().map(|r| Mutex::new(r.unwrap())).collect(),
                barrier: barrier.clone(),
            })
            .collect()
    }
}

impl Team for LocalTeam {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn send(&self, to: usize, data: &[u8]) {
        self.to[to].send(data.to_vec()).expect("taldekidea bizirik");
    }

    fn recv(&self, from: usize) -> Vec<u8> {
        self.from[from]
            .lock()
            .unwrap()
            .recv()
            .expect("taldekidea bizirik")
    }

    fn exchange(&self, partner: usize, send: &[u8], recv: &mut [u8]) {
        self.send(partner, send);
        recv.copy_from_slice(&self.recv(partner));
    }

    fn barrier(&self) {
        self.barrier.wait();
    }
}
