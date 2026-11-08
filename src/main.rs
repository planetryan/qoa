// qoa komando-lerroko tresna. komando bakoitzak gauza bat egiten du eta zuzenean esaten du zer egin duen.
// programak .qoa (iturburua), .qg (glifo-sintaxia), .qasm (openqasm, zuzenean inportatua) edo .qx (konpilatua)
// izan daitezke.
// laguntza-testuak eta mezuak euskaraz edo ingelesez daude (ikus src/lang.rs).
// komando-lerroko tresnak ez du `unsafe`-rik, eta ezin du izan
#![forbid(unsafe_code)]

mod visualizer;

use clap::{Parser, Subcommand, ValueEnum};
use qoa::asm;
use qoa::isa::{self, Status};
use qoa::program::{Op, Program};
use qoa::report::{self, ReportInput, Style};
use qoa::state::format_bytes;
use qoa::vm::{self, RunConfig};
use qoa::{t, ts};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use visualizer::SpectrumDirection;

const QOA_AUTHOR: &str = "Ryan (@planetryan on github)";

#[derive(Parser, Debug)]
#[command(
    name = "qoa",
    author = QOA_AUTHOR,
    version,
    about = ts!(
        "QOA (Quantum Optical Assembly): an assembly language and simulator for qubits and optical modes.",
        "QOA (Quantum Optical Assembly): qubit eta modu optikoetarako mihiztadura-lengoaia eta simulagailua."
    ),
    long_about = ts!(
        "QOA (Quantum Optical Assembly): an assembly language and simulator for qubits and optical modes.\nIt runs on a classical CPU: every result is computed, not measured on quantum hardware.\n\nRun `qoa isa` for the instruction reference. Set QOA_LANG=eu or QOA_LANG=en to choose the language.",
        "QOA (Quantum Optical Assembly): qubit eta modu optikoetarako mihiztadura-lengoaia eta simulagailua.\nCPU klasiko batean exekutatzen da: emaitza guztiak kalkulatuak dira, ez hardware kuantikoan neurtuak.\n\nAginduen erreferentziarako: `qoa isa`. Hizkuntza aukeratzeko: QOA_LANG=eu edo QOA_LANG=en."
    )
)]
struct Cli {
    #[arg(long, visible_alias = "bm", help = ts!(
        "measure this machine: every part of QOA, with a score per part and in total (the same as `qoa benchmark`)",
        "makina hau neurtu: QOA-ren zati guztiak, puntuazio batekin zati bakoitzeko eta guztira (`qoa benchmark` bera)"
    ))]
    benchmark: bool,
    #[arg(long, requires = "benchmark", help = ts!(
        "with --benchmark: smaller problems, about 10 seconds instead of half a minute",
        "--benchmark-ekin: problema txikiagoak, 10 segundo inguru minutu erdiaren ordez"
    ))]
    quick: bool,
    #[arg(long = "no-gpu", requires = "benchmark", help = ts!(
        "with --benchmark: skip the GPU part",
        "--benchmark-ekin: GPU zatia saltatu"
    ))]
    no_gpu: bool,
    #[arg(long, requires = "benchmark", help = ts!(
        "with --benchmark: print the result as json",
        "--benchmark-ekin: emaitza json gisa inprimatu"
    ))]
    json: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(visible_aliases = ["bench", "bm"], about = ts!(
        "Measure this machine: every part of QOA, with a score per part and in total",
        "Makina hau neurtu: QOA-ren zati guztiak, puntuazio batekin zati bakoitzeko eta guztira"
    ))]
    Benchmark {
        #[arg(long, help = ts!(
            "smaller problems, about 10 seconds instead of half a minute",
            "problema txikiagoak, 10 segundo inguru minutu erdiaren ordez"
        ))]
        quick: bool,
        #[arg(long = "no-gpu", help = ts!("skip the GPU part", "GPU zatia saltatu"))]
        no_gpu: bool,
        #[arg(long, help = ts!("print the result as json", "emaitza json gisa inprimatu"))]
        json: bool,
    },
    #[command(about = ts!("Run a program (.qoa, .qg, .qasm or .qx) and show its results", "Programa bat exekutatu (.qoa, .qg, .qasm edo .qx) eta emaitzak erakutsi"))]
    Run {
        #[arg(help = ts!("program file", "programa-fitxategia"))]
        file: PathBuf,
        #[arg(long, help = ts!(
            "how many times to run the program (default 1024; ignored if it measures nothing)",
            "programa zenbat aldiz exekutatu (lehenetsia 1024; ez da kontuan hartzen ezer neurtzen ez badu)"
        ))]
        shots: Option<u64>,
        #[arg(long, help = ts!(
            "random seed, to repeat a run exactly (default: a fresh random seed, which is printed)",
            "ausazko hazia, exekuzio bat zehazki errepikatzeko (lehenetsia: hazi berri bat, inprimatu egiten dena)"
        ))]
        seed: Option<u64>,
        #[arg(long, default_value_t = 0.0, value_parser = probability, help = ts!(
            "depolarizing noise: after each gate, each qubit it touched gets a random X, Y or Z with this probability",
            "depolarizazio-zarata: ate bakoitzaren ondoren, ukitutako qubit bakoitzak ausazko X, Y edo Z bat jasotzen du probabilitate honekin"
        ))]
        noise: f64,
        #[arg(long, default_value_t = 0.0, value_parser = probability, help = ts!(
            "photon loss: after each optical operation, each photon in the touched modes is lost with this probability",
            "fotoi-galera: eragiketa optiko bakoitzaren ondoren, ukitutako moduetako fotoi bakoitza probabilitate honekin galtzen da"
        ))]
        loss: f64,
        #[arg(long, value_name = "FILE", conflicts_with_all = ["noise", "loss"], help = ts!(
            "a real device's noise model (json, from tools/run_on_ibm.py --save-model): after each native gate its depolarizing error and T1/T2 relaxation, and readout errors on measurements. The program must use only the device's native gates. Or a tabletop bench's model (format qoa-bench-1, docs/bench.md): detector efficiency and dark clicks, heralded-source efficiency and two-photon emission, pair visibility",
            "benetako gailu baten zarata-eredua (json, tools/run_on_ibm.py --save-model-etik): ate natibo bakoitzaren ondoren haren depolarizazio-errorea eta T1/T2 erlaxazioa, eta irakurketa-erroreak neurketetan. programak gailuaren ate natiboak bakarrik erabil ditzake. edo mahai gaineko banku baten eredua (qoa-bench-1 formatua, docs/bench.eu.md): detektagailuen eraginkortasuna eta iluneko klikak, iturri iragarriaren eraginkortasuna eta bi fotoiko igorpena, bikoteen ikusgarritasuna"
        ))]
        noise_model: Option<PathBuf>,
        #[arg(long, help = ts!("also show the exact state at the end of the first shot", "lehen shot-aren amaierako egoera zehatza ere erakutsi"))]
        state: bool,
        #[arg(long, default_value_t = 16, help = ts!("how many amplitudes to show (0 = all)", "zenbat anplitude erakutsi (0 = guztiak)"))]
        top: usize,
        #[arg(long, help = ts!("print the result record as json instead of text", "emaitza-erregistroa json gisa inprimatu, testuaren ordez"))]
        json: bool,
        #[arg(long, default_value_t = 100_000_000, help = ts!(
            "stop a run after this many executed instructions (guards against endless loops)",
            "exekuzioa gelditu hainbeste agindu exekutatu ondoren (amaigabeko begizten aurkako babesa)"
        ))]
        max_steps: u64,
        #[arg(long, help = ts!(
            "apply qubit gates on a GPU through Vulkan (64-bit floats); measurements, optics and noise stay on the cpu. Fails if no usable device is found",
            "qubit-ateak GPU batean aplikatu vulkan bidez (64 biteko koma higikorra); neurketak, optika eta zarata cpu-an geratzen dira. huts egiten du gailu erabilgarririk ez badago"
        ))]
        gpu: bool,
        #[arg(long, value_enum, value_name = "API", help = ts!(
            "which GPU interface --gpu uses: vulkan, opencl (AMD Instinct, NVIDIA, Intel, POCL), or auto (Vulkan, then OpenCL). Implies --gpu. Each MPI process on a node takes its own GPU (QOA_GPU_DEVICE overrides)",
            "--gpu-k zein GPU interfaze erabiltzen duen: vulkan, opencl (AMD Instinct, NVIDIA, Intel, POCL), edo auto (vulkan, gero OpenCL). --gpu dakar. nodo bateko MPI prozesu bakoitzak bere GPUa hartzen du (QOA_GPU_DEVICE-k gainidazten du)"
        ))]
        gpu_api: Option<GpuApiArg>,
        #[arg(long, value_enum, default_value = "double", help = ts!(
            "number precision for the qubit state: double (64-bit floats) or quad (128-bit double-double, ~32 digits, qubit programs only)",
            "qubit-egoeraren zenbakien doitasuna: double (64 biteko koma higikorra) edo quad (128 biteko double-double, ~32 digitu, qubit-programak bakarrik)"
        ))]
        precision: PrecisionArg,
        #[arg(long, help = ts!(
            "spread the shots over MPI processes (start with mpirun or srun); needs a QOA built with --features mpi. Rank 0 prints the result",
            "shot-ak MPI prozesuen artean banatu (mpirun edo srun-ekin abiarazi); --features mpi-rekin konpilatutako QOA behar du. 0. rank-ak inprimatzen du emaitza"
        ))]
        mpi: bool,
        #[arg(long, requires = "mpi", help = ts!(
            "with --mpi: split one state over the processes (their number a power of two) instead of giving each a copy, for states larger than one node's memory. Qubit programs only",
            "--mpi-rekin: egoera bakar bat prozesuen artean banatu (haien kopurua bi-ren berretura) bakoitzari kopia bat eman ordez, nodo baten memoria baino handiagoak diren egoeretarako. qubit-programak bakarrik"
        ))]
        distribute: bool,
        #[arg(short, long, conflicts_with_all = ["json", "state"], help = ts!(
            "print only what the program itself writes, byte for byte, with no report; qoa exits with the program's EXIT status. For programs used as tools, like an assembler",
            "programak berak idazten duena bakarrik inprimatu, bytez byte, txostenik gabe; qoa programaren EXIT egoerarekin amaitzen da. tresna gisa erabiltzen diren programetarako, mihiztatzaile bat adibidez"
        ))]
        quiet: bool,
        #[arg(long, value_name = "COMMAND", conflicts_with_all = ["noise", "loss", "noise_model", "gpu", "gpu_api", "mpi", "quiet", "state"], help = ts!(
            "run on a bench instead of simulating: start this controller program (words split at spaces, no shell), send it the program's optics as one line of json and read the counts back (protocol qoa-device-1, docs/bench.md). qoa bench-sim is a stand-in controller that answers by simulation",
            "simulatu ordez banku batean exekutatu: kontrolagailu-programa hau abiarazi (hitzak zuriuneetan banatuta, shell-ik gabe), programaren optika json lerro gisa bidali eta kontaketak irakurri (qoa-device-1 protokoloa, docs/bench.eu.md). qoa bench-sim simulazioz erantzuten duen ordezko kontrolagailu bat da"
        ))]
        device: Option<String>,
        #[arg(long, value_name = "TARGET", requires = "device", help = ts!(
            "with --device: the bench's profile, bench-b (default) or bench-a; the program must pass it",
            "--device-rekin: bankuaren profila, bench-b (lehenetsia) edo bench-a; programak bete behar du"
        ))]
        target: Option<String>,
        #[arg(last = true, value_name = "ARGS", help = ts!(
            "arguments for the program, after --; it reads them with ARGC and ARGBYTE",
            "programarentzako argumentuak, -- ondoren; ARGC eta ARGBYTE-rekin irakurtzen ditu"
        ))]
        args: Vec<String>,
    },
    #[command(about = ts!("Assemble a program and report what it needs, without running it", "Programa bat mihiztatu eta zer behar duen esan, exekutatu gabe"))]
    Check {
        #[arg(help = ts!("program file", "programa-fitxategia"))]
        file: PathBuf,
        #[arg(long, value_name = "TARGET", help = ts!(
            "also check that the program runs on this hardware: bench-a (classical-light bench) or bench-b (single photons, a pair source, click detectors); each instruction it cannot run is an error, with the line and why (docs/bench.md)",
            "egiaztatu programa hardware honetan dabilela ere: bench-a (argi klasikoko bankua) edo bench-b (fotoi bakarrak, bikote-iturri bat, klik-detektagailuak); exekutatu ezin duen agindu bakoitza errore bat da, lerroa eta zergatia esanda (docs/bench.eu.md)"
        ))]
        target: Option<String>,
    },
    #[command(about = ts!("Assemble a program into a .qx binary", "Programa bat .qx bitar batean mihiztatu"))]
    Compile {
        #[arg(help = ts!("program file", "programa-fitxategia"))]
        file: PathBuf,
        #[arg(short, long, help = ts!("output file (default: same name with .qx)", "irteera-fitxategia (lehenetsia: izen bera .qx-rekin)"))]
        output: Option<PathBuf>,
        #[arg(long, value_name = "ASSEMBLER", help = ts!(
            "assemble with this QOA program instead of the built-in assembler, for example src/qoa/asm.qoa (QOA's assembler written in QOA). It runs on the QOA machine, reads the source on its input, gets the file name as its argument and writes the .qx",
            "mihiztatzaile integratuaren ordez QOA programa honekin mihiztatu, adibidez src/qoa/asm.qoa (QOAn idatzitako QOA-ren mihiztatzailea). QOA makinan exekutatzen da, iturburua sarreran irakurtzen du, fitxategi-izena argumentu gisa jasotzen du eta .qx idazten du"
        ))]
        assembler: Option<PathBuf>,
        #[arg(long, conflicts_with = "assembler", help = ts!(
            "assemble with QOA's own assembler written in QOA, built into this binary (src/qoa/asm.qoa; see qoa tools). The .qx is byte for byte what the built-in assembler writes",
            "QOAn idatzitako QOA-ren mihiztatzailearekin mihiztatu, binario honetan sartua (src/qoa/asm.qoa; ikus qoa tools). .qx-a mihiztatzaile integratuak idazten dituen byte berberak dira"
        ))]
        self_hosted: bool,
    },
    #[command(about = ts!("Convert a program to another form: glyph, OpenQASM 2.0 or 3, IonQ json, canonical QOA, a listing, json or a dot graph", "Programa bat beste forma batera bihurtu: glifoa, OpenQASM 2.0 edo 3, IonQ json, QOA kanonikoa, zerrenda, json edo dot grafoa"))]
    Export {
        #[arg(help = ts!("program file", "programa-fitxategia"))]
        file: PathBuf,
        #[arg(long, value_enum, required_unless_present = "to_glyph", help = ts!(
            "target: qasm (OpenQASM 2.0), qasm3 (OpenQASM 3: for, if/else and while blocks on measured bits), ionq (IonQ JSON), glyph (glyph syntax), qoa (canonical plain QOA, also from a .qx binary), listing (qoa with each operation's address and source line), json (the program as data), dot (control-flow graph for Graphviz)",
            "helburua: qasm (OpenQASM 2.0), qasm3 (OpenQASM 3: for, if/else eta while blokeak bit neurtuekin), ionq (IonQ JSON), glyph (glifo-sintaxia), qoa (QOA arrunt kanonikoa, .qx bitar batetik ere), listing (qoa eragiketa bakoitzaren helbide eta iturburu-lerroarekin), json (programa datu gisa), dot (kontrol-fluxuaren grafoa graphviz-erako)"
        ))]
        to: Option<ExportFormat>,
        #[arg(long, conflicts_with = "to", help = ts!("same as --to glyph", "--to glyph-en berdina"))]
        to_glyph: bool,
        #[arg(short, long, help = ts!("output file (default: print to the terminal)", "irteera-fitxategia (lehenetsia: terminalean inprimatu)"))]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 1024, help = ts!("shots to request (IonQ only)", "eskatu beharreko shot-ak (IonQ bakarrik)"))]
        shots: u64,
        #[arg(long, default_value_t = 100, help_heading = "glyph", help = ts!(
            "glyph: widest line in characters; statements fill lines up to it (0 = one statement per line)",
            "glifoa: lerro zabalena karakteretan; adierazpenek lerroak betetzen dituzte hura arte (0 = adierazpen bat lerroko)"
        ))]
        width: usize,
        #[arg(long, default_value = "Q,C,L", help_heading = "glyph", help = ts!(
            "glyph: names of the qubit, bit and mode registers",
            "glifoa: qubit, bit eta modu erregistroen izenak"
        ))]
        names: String,
        #[arg(long, help_heading = "glyph", help = ts!(
            "write instructions with their full names and forms (MEASURE q0 -> c0) instead of the short ones (Mq0.c0); for glyph, qoa and listing",
            "idatzi aginduak beren izen eta forma osoekin (MEASURE q0 -> c0), laburren ordez (Mq0.c0); glyph, qoa eta listing-entzat"
        ))]
        long_names: bool,
        #[arg(long, help_heading = "glyph", help = ts!(
            "glyph: one statement per element instead of ranges like H/Q[0..3]",
            "glifoa: adierazpen bat elementu bakoitzeko, H/Q[0..3] bezalako tarteen ordez"
        ))]
        no_ranges: bool,
        #[arg(long, help_heading = "glyph", help = ts!(
            "glyph: write indices in hexadecimal, like Q[0x3]",
            "glifoa: idatzi indizeak hamaseitarrean, Q[0x3] bezala"
        ))]
        hex: bool,
        #[arg(long, help_heading = "glyph", help = ts!(
            "glyph: as dense as possible: no ; or spaces where the boundary is clear, bits as #C0",
            "glifoa: ahalik eta trinkoen: ; eta hutsunerik gabe muga argia denean, bitak #C0 gisa"
        ))]
        dense: bool,
        #[arg(long, help_heading = "glyph", help = ts!(
            "glyph: convert a .qoa source line by line, keeping its comments, labels and layout",
            "glifoa: .qoa iturburu bat lerroz lerro bihurtu, haren iruzkinak, etiketak eta antolaketa gordeta"
        ))]
        keep_source: bool,
    },
    #[command(about = ts!(
        "Convert an OpenQASM file (.qasm: all of 2.0; of 3, bits, if/else, while, for and gate modifiers) or a glyph-syntax file (.qg) into plain QOA source",
        "OpenQASM fitxategi bat (.qasm: 2.0 osoa; 3-tik, bitak, if/else, while, for eta ate-modifikatzaileak) edo glifo-sintaxiko bat (.qg) QOA iturburu arrunt bihurtu"
    ))]
    Import {
        #[arg(help = ts!(".qasm or .qg file", ".qasm edo .qg fitxategia"))]
        file: PathBuf,
        #[arg(short, long, help = ts!("output file (default: print to the terminal)", "irteera-fitxategia (lehenetsia: terminalean inprimatu)"))]
        output: Option<PathBuf>,
    },
    #[command(about = ts!(
        "Seal source files: add a last line with the SHA-256 of the lines above it, so QOA refuses to run a sealed file that was changed or damaged afterwards. Sealing again after an edit updates it",
        "Iturburu-fitxategiak zigilatu: azken lerro bat gehitu haren gaineko lerroen SHA-256-arekin, QOA-k ondoren aldatu edo hondatutako fitxategi zigilatu bat exekuta ez dezan. aldaketa baten ondoren berriz zigilatzeak eguneratzen du"
    ))]
    Seal {
        #[arg(required = true, help = ts!(".qoa, .qg or .qasm files", ".qoa, .qg edo .qasm fitxategiak"))]
        files: Vec<PathBuf>,
    },
    #[command(name = "bench-sim", about = ts!(
        "A stand-in bench controller: answer the device protocol (qoa-device-1) on standard input and output by simulation, for qoa run --device \"qoa bench-sim\" (docs/bench.md)",
        "Ordezko banku-kontrolagailu bat: gailu-protokoloari (qoa-device-1) sarrera eta irteera estandarretan simulazioz erantzun, qoa run --device \"qoa bench-sim\"-entzat (docs/bench.eu.md)"
    ))]
    BenchSim {
        #[arg(long, value_name = "FILE", help = ts!(
            "the bench model to simulate (format qoa-bench-1); without it, an ideal bench",
            "simulatu beharreko banku-eredua (qoa-bench-1 formatua); gabe, banku ideal bat"
        ))]
        model: Option<PathBuf>,
    },
    #[command(about = ts!(
        "List the tools written in QOA that are built into this binary, or write their sources out",
        "Binario honetan sartutako QOAn idatzitako tresnak zerrendatu, edo haien iturburuak idatzi"
    ))]
    Tools {
        #[arg(long, value_name = "DIR", help = ts!(
            "write each tool's source into this folder: the plain QOA file and its glyph form (for the assembler, asm.qoa and asm.qg)",
            "tresna bakoitzaren iturburua karpeta honetan idatzi: QOA arrunteko fitxategia eta haren glifo-forma (mihiztatzailearentzat, asm.qoa eta asm.qg)"
        ))]
        write: Option<PathBuf>,
    },
    #[command(about = ts!("Show the instruction reference, or one instruction in detail", "Aginduen erreferentzia erakutsi, edo agindu bat zehatz"))]
    Isa {
        #[arg(help = ts!("instruction or alias to look up", "bilatu beharreko agindua edo ezizena"))]
        name: Option<String>,
        #[arg(long, help = ts!("print the full reference as markdown (this is docs/ISA.md)", "erreferentzia osoa markdown gisa inprimatu (hau da docs/ISA.md)"))]
        markdown: bool,
    },
    #[command(about = ts!(
        "Render an audio file (.wav) into a spectrum video with ffmpeg. An art tool: it does not visualize a quantum state",
        "Audio-fitxategi bat (.wav) espektro-bideo bihurtu ffmpeg-ekin. Arte-tresna bat da: ez du egoera kuantikorik bistaratzen"
    ))]
    Visual {
        #[arg(help = ts!(
            "input: any audio or video file ffmpeg can read (mp3, flac, wav, ...), or - for stdin",
            "sarrera: ffmpeg-ek irakur dezakeen edozein audio edo bideo fitxategi (mp3, flac, wav, ...), edo - stdin-erako"
        ))]
        input: PathBuf,
        #[arg(help = ts!(
            "output file; its extension picks the container (.mp4, .mkv, .flac, .wav, ...)",
            "irteera-fitxategia; luzapenak edukiontzia aukeratzen du (.mp4, .mkv, .flac, .wav, ...)"
        ))]
        output: PathBuf,
        #[arg(long, value_name = "FORMAT", help = ts!(
            "the input is raw samples in this ffmpeg sample format (u8, s16le, s24le, s32le, f32le, f64le, ...); use with --rate and --channels",
            "sarrera lagin gordinak dira ffmpeg-en lagin-formatu honetan (u8, s16le, s24le, s32le, f32le, f64le, ...); --rate eta --channels-ekin"
        ))]
        raw: Option<String>,
        #[arg(long, requires = "raw", default_value_t = 48000, help = ts!("sample rate of raw input, in Hz", "sarrera gordinaren lagin-maiztasuna, Hz-tan"))]
        rate: u32,
        #[arg(long, requires = "raw", default_value_t = 1, help = ts!("channels of raw input", "sarrera gordinaren kanalak"))]
        channels: u32,
        #[arg(long, conflicts_with_all = ["video_only", "keep_audio"], help = ts!(
            "write only the processed audio, no video",
            "audio prozesatua bakarrik idatzi, bideorik gabe"
        ))]
        audio_only: bool,
        #[arg(long, conflicts_with = "keep_audio", help = ts!("write only the video, no audio track", "bideoa bakarrik idatzi, audio-pistarik gabe"))]
        video_only: bool,
        #[arg(long, help = ts!(
            "keep the original audio unmodified; the video still follows the processed light",
            "jatorrizko audioa aldatu gabe gorde; bideoak argi prozesatuari jarraitzen dio"
        ))]
        keep_audio: bool,
        #[arg(long, default_value_t = 1e5, help = ts!(
            "photons per sample of a full-scale tone: fewer photons, more quantum noise",
            "eskala osoko tonu baten fotoiak lagin bakoitzeko: fotoi gutxiago, zarata kuantiko gehiago"
        ))]
        photons: f64,
        #[arg(long, default_value_t = 0.0, help = ts!("fraction of the light lost on the way (0 to <1)", "bidean galtzen den argiaren zatia (0tik <1era)"))]
        loss: f64,
        #[arg(long, default_value_t = 0.0, value_name = "DB", help = ts!(
            "squeeze the light's measured quadrature by this many dB (below shot noise, like LIGO)",
            "argiaren koadratura neurtua dB hauetan estutu (tiro-zarataren azpitik, LIGO bezala)"
        ))]
        squeeze: f64,
        #[arg(long, default_value_t = 0.0, value_name = "RAD", allow_negative_numbers = true, help = ts!(
            "Kerr self-phase: the brightest mode picks up this many radians",
            "kerr fase propioa: modu distiratsuenak hainbeste radian hartzen ditu"
        ))]
        kerr: f64,
        #[arg(long, default_value_t = 0.0, value_name = "RAD", allow_negative_numbers = true, help = ts!(
            "beamsplitter mesh between neighbouring bands, at this angle",
            "ondoko banden arteko habe-zatitzaile sarea, angelu honetan"
        ))]
        mix: f64,
        #[arg(long, value_enum, default_value = "homodyne", help = ts!(
            "how the light is detected: mean (exact average), homodyne (one shot, real quantum noise) or count (photon counting)",
            "argia nola detektatzen den: mean (batez besteko zehatza), homodyne (shot bat, benetako zarata kuantikoa) edo count (fotoi-zenbaketa)"
        ))]
        detect: DetectArg,
        #[arg(long, default_value_t = 1024, help = ts!(
            "modes per block and channel (MDCT size, even, 16 or more)",
            "modu-kopurua bloke eta kanal bakoitzeko (MDCT tamaina, bikoitia, 16 edo gehiago)"
        ))]
        block: usize,
        #[arg(long, default_value = "1920x1080", help = ts!("resolution, e.g. 1920x1080 or 3840x2160", "bereizmena, adib. 1920x1080 edo 3840x2160"))]
        resolution: String,
        #[arg(long, value_enum, default_value = "row", help = ts!(
            "the light's quantum states (exact Wigner functions of the brightest modes): row (4 panels over the art), full (8 large panels) or off",
            "argiaren egoera kuantikoak (modu distiratsuenen wigner funtzio zehatzak): row (4 panel artearen gainean), full (8 panel handi) edo off"
        ))]
        phase_space: PhaseSpaceArg,
        #[arg(long, default_value_t = 60, help = ts!("frames per second", "irudi segundoko"))]
        fps: u32,
        #[arg(long, conflicts_with = "rtl", help = ts!("spectrum runs left to right", "espektroa ezkerretik eskuinera"))]
        ltr: bool,
        #[arg(long, conflicts_with = "ltr", help = ts!("spectrum runs right to left", "espektroa eskuinetik ezkerrera"))]
        rtl: bool,
        #[arg(long, help = ts!(
            "random seed: the same seed and audio give the same video (default: a fresh seed, which is printed)",
            "ausazko hazia: hazi eta audio berak bideo bera ematen dute (lehenetsia: hazi berri bat, inprimatzen dena)"
        ))]
        seed: Option<u64>,
        #[arg(long = "ffmpeg-flag", value_name = "FLAG", num_args = 0.., action = clap::ArgAction::Append, help = ts!("extra ffmpeg flags", "ffmpeg-en aukera gehigarriak"))]
        ffmpeg_flags: Vec<String>,
        #[arg(last = true, allow_hyphen_values = true, help = ts!("arguments passed to ffmpeg after --", "-- ondoren ffmpeg-i pasatzen zaizkion argumentuak"))]
        ffmpeg_args: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum GpuApiArg {
    Auto,
    Vulkan,
    Opencl,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum PrecisionArg {
    Double,
    Quad,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum PhaseSpaceArg {
    Row,
    Full,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum DetectArg {
    Mean,
    Homodyne,
    Count,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum ExportFormat {
    // openqasm 2.0 (qubit-zatia)
    Qasm,
    // openqasm 3.0: begiztak, if blokeak eta while begiztak bit neurtuen arabera
    Qasm3,
    // ionq-ren json formatua
    Ionq,
    // glifo-sintaxia
    Glyph,
    // QOA arrunt kanonikoa (desmihiztatzea)
    Qoa,
    // QOA kanonikoa helbide eta iturburu-lerroekin
    Listing,
    // programa json gisa
    Json,
    // kontrol-fluxuaren grafoa (graphviz)
    Dot,
}

fn probability(s: &str) -> Result<f64, String> {
    let v: f64 = s
        .parse()
        .map_err(|_| t!("'{}' is not a number", "'{}' ez da zenbaki bat", s))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(t!(
            "{} is not a probability (0 to 1)",
            "{} ez da probabilitate bat (0tik 1era)",
            v
        ))
    }
}

// clap-ek berak idazten dituen testu finkoak (izenburuak, "[default: ...]", errore ohikoenak) euskaraz.
// luzeenak lehenago, zatiak bata bestearen barruan daudelako
const CLAP_EU: &[(&str, &str)] = &[
    (
        "Print help (see a summary with '-h')",
        "Laguntza inprimatu (laburpena: '-h')",
    ),
    (
        "Print help (see more with '--help')",
        "Laguntza inprimatu (gehiago: '--help')",
    ),
    (
        "Print this message or the help of the given subcommand(s)",
        "Mezu hau edo azpikomando baten laguntza inprimatu",
    ),
    ("Print help", "Laguntza inprimatu"),
    ("Print version", "Bertsioa inprimatu"),
    ("Usage:", "Erabilera:"),
    ("Arguments:", "Argumentuak:"),
    ("Options:", "Aukerak:"),
    ("Commands:", "Komandoak:"),
    ("[default: ", "[lehenetsia: "),
    ("[possible values: ", "[balio posibleak: "),
    ("[aliases: ", "[ezizenak: "),
    ("error: ", "errorea: "),
    ("tip: ", "aholkua: "),
    ("unexpected argument '", "ustekabeko argumentua: '"),
    ("' found", "'"),
    ("the argument ", ""),
    ("unrecognized subcommand", "azpikomando ezezaguna"),
    (
        "the following required arguments were not provided:",
        "argumentu hauek falta dira:",
    ),
    ("invalid value", "balio baliogabea"),
    ("for '", "honentzat: '"),
    ("a value is required", "balio bat behar da"),
    ("but none was supplied", "baina ez da bat ere eman"),
    ("cannot be used with", "ezin da honekin erabili:"),
    (
        "For more information, try '--help'.",
        "Informazio gehiagorako: '--help'.",
    ),
    // rust-en zenbaki-irakurketaren erroreak (adib. `--shots abc`)
    ("invalid digit found in string", "ez da zenbaki oso bat"),
    ("cannot parse integer from empty string", "hutsik dago"),
    (
        "number too large to fit in target type",
        "zenbakia handiegia da",
    ),
    ("invalid float literal", "ez da zenbaki bat"),
    ("to pass ", ""),
    (" as a value, use ", " balio gisa pasatzeko, erabili "),
];

// programaren EXIT egoera (`qoa run`), qoa bera amaitzean itzultzeko
static EXIT_STATUS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

// irteera estandarraren irakurlea joan bada (`qoa ... | head`), isilik amaitu, unix tresna batek bezala
// (128 + SIGPIPE), panic edo errore-mezurik gabe
fn closed_pipe(e: std::io::Error) -> std::io::Error {
    if e.kind() == std::io::ErrorKind::BrokenPipe {
        std::process::exit(141);
    }
    e
}

struct Out<W>(W);

impl<W: std::io::Write> std::io::Write for Out<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf).map_err(closed_pipe)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush().map_err(closed_pipe)
    }
}

fn main() -> ExitCode {
    // println!-ek panic egiten du irteera ezin bada idatzi: irteera itxita badago, isilik amaitu; beste
    // edozein arrazoirengatik (disko betea), errore-mezu argi batekin, panic-ik gabe
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("");
        if let Some(why) = msg.strip_prefix("failed printing to stdout: ") {
            if why.contains("Broken pipe") || why.contains("pipe is being closed") {
                std::process::exit(141);
            }
            eprintln!(
                "{}",
                t!(
                    "error: cannot write the output: {}",
                    "errorea: ezin da irteera idatzi: {}",
                    why
                )
            );
            std::process::exit(1);
        }
        default(info)
    }));
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        // ingelesez clap-en irteera bere horretan (koloreekin); euskaraz testu finkoak itzulita
        Err(e) if qoa::lang::is_eu() => {
            let mut text = e.render().to_string();
            for (en, eu) in CLAP_EU {
                text = text.replace(en, eu);
            }
            if e.use_stderr() {
                eprint!("{}", text);
            } else {
                print!("{}", text);
            }
            return ExitCode::from(e.exit_code() as u8);
        }
        Err(e) => e.exit(),
    };
    let command = match (cli.command, cli.benchmark) {
        (Some(c), _) => c,
        (None, true) => Command::Benchmark {
            quick: cli.quick,
            no_gpu: cli.no_gpu,
            json: cli.json,
        },
        (None, false) => {
            use clap::CommandFactory;
            let _ = Cli::command().print_help();
            return ExitCode::from(2);
        }
    };
    let result = match command {
        Command::Benchmark {
            quick,
            no_gpu,
            json,
        } => cmd_benchmark(quick, no_gpu, json),
        Command::Run {
            file,
            shots,
            seed,
            noise,
            loss,
            noise_model,
            state,
            top,
            json,
            max_steps,
            gpu,
            gpu_api,
            precision,
            mpi,
            distribute,
            quiet,
            args,
            device,
            target,
        } => cmd_run(
            &file,
            shots,
            seed,
            noise,
            loss,
            noise_model.as_deref(),
            state,
            top,
            json,
            max_steps,
            match (gpu, gpu_api) {
                (_, Some(GpuApiArg::Vulkan)) => Some(qoa::gpu::Api::Vulkan),
                (_, Some(GpuApiArg::Opencl)) => Some(qoa::gpu::Api::OpenCl),
                (_, Some(GpuApiArg::Auto)) | (true, None) => Some(qoa::gpu::Api::Auto),
                (false, None) => None,
            },
            match precision {
                PrecisionArg::Double => qoa::vm::Precision::Double,
                PrecisionArg::Quad => qoa::vm::Precision::Quad,
            },
            mpi,
            distribute,
            quiet,
            args,
            device.map(|cmd| (cmd, target.unwrap_or_else(|| "bench-b".to_string()))),
        ),
        Command::Check { file, target } => cmd_check(&file, target.as_deref()),
        Command::BenchSim { model } => cmd_bench_sim(model.as_deref()),
        Command::Compile {
            file,
            output,
            assembler,
            self_hosted,
        } => cmd_compile(&file, output, assembler.as_deref(), self_hosted),
        Command::Tools { write } => cmd_tools(write.as_deref()),
        Command::Export {
            file,
            to,
            to_glyph,
            output,
            shots,
            width,
            names,
            long_names,
            no_ranges,
            hex,
            dense,
            keep_source,
        } => {
            let to = to.unwrap_or(ExportFormat::Glyph);
            let _ = to_glyph;
            if keep_source {
                cmd_keep_source(&file, to, output)
            } else {
                glyph_options(to, width, &names, long_names, no_ranges, hex)
                    .map(|g| qoa::disasm::GlyphOptions { dense, ..g })
                    .and_then(|g| {
                        if dense && to != ExportFormat::Glyph {
                            return Err(t!(
                                "--dense only applies to --to glyph",
                                "--dense --to glyph-ekin bakarrik dabil"
                            ));
                        }
                        cmd_export(&file, to, output, shots, &g)
                    })
            }
        }
        Command::Import { file, output } => cmd_import(&file, output),
        Command::Isa { name, markdown } => cmd_isa(name, markdown),
        Command::Seal { files } => files.iter().try_for_each(|f| cmd_seal(f)),
        Command::Visual {
            input,
            output,
            raw,
            rate,
            channels,
            audio_only,
            video_only,
            keep_audio,
            photons,
            loss,
            squeeze,
            kerr,
            mix,
            detect,
            block,
            resolution,
            fps,
            phase_space,
            ltr,
            rtl,
            seed,
            ffmpeg_flags,
            ffmpeg_args,
        } => {
            // audio-fitxategi batek (wav, flac, mp3...) ezin du bideorik izan: orduan audioa bakarrik,
            // ffmpeg-ek huts egin beharrean
            let audio_file = output
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
                .is_some_and(|e| {
                    [
                        "wav", "flac", "mp3", "ogg", "oga", "opus", "m4a", "aac", "aiff", "aif",
                        "wma", "alac", "caf", "au",
                    ]
                    .contains(&e.as_str())
                });
            let conflict = (audio_file && (video_only || keep_audio)).then(|| {
                t!(
                    "{} is an audio file and cannot hold video; write the video to .mp4, .mkv or .webm",
                    "{} audio-fitxategi bat da eta ezin du bideorik izan; idatzi bideoa .mp4, .mkv edo .webm-ra",
                    output.display()
                )
            });
            let mode = if audio_only || audio_file {
                visualizer::Mode::AudioOnly
            } else if video_only {
                visualizer::Mode::VideoOnly
            } else if keep_audio {
                visualizer::Mode::KeepAudio
            } else {
                visualizer::Mode::Both
            };
            let light = qoa::light::Light {
                photons,
                loss,
                squeeze_db: squeeze,
                kerr,
                mix,
                detect: match detect {
                    DetectArg::Mean => qoa::light::Detect::Mean,
                    DetectArg::Homodyne => qoa::light::Detect::Homodyne,
                    DetectArg::Count => qoa::light::Detect::Count,
                },
            };
            let raw = raw.map(|format| visualizer::Raw {
                format,
                rate,
                channels,
            });
            if let Some(why) = conflict {
                Err(why)
            } else {
                cmd_visual(
                    &input,
                    &output,
                    raw,
                    mode,
                    light,
                    block,
                    &resolution,
                    fps,
                    match phase_space {
                        PhaseSpaceArg::Row => visualizer::PhaseSpace::Row,
                        PhaseSpaceArg::Full => visualizer::PhaseSpace::Full,
                        PhaseSpaceArg::Off => visualizer::PhaseSpace::Off,
                    },
                    ltr,
                    rtl,
                    seed,
                    &ffmpeg_flags,
                    &ffmpeg_args,
                )
            }
        }
    };
    match result {
        Ok(()) => ExitCode::from(EXIT_STATUS.load(std::sync::atomic::Ordering::Relaxed)),
        Err(message) => {
            // mihiztatzailearen erroreek beren formatua dute; besteei aurrizkia jarri. mezu hutsa: programak
            // (`--assembler`) bere erroreak jada inprimatu ditu, eta haren EXIT egoera itzultzen da
            if message.is_empty() {
            } else if message.starts_with(ts!("error:", "errorea:")) {
                eprint!("{}", qoa::term::for_stderr(&message));
            } else {
                eprintln!(
                    "{} {}",
                    ts!("error:", "errorea:"),
                    qoa::term::for_stderr(&message)
                );
            }
            match EXIT_STATUS.load(std::sync::atomic::Ordering::Relaxed) {
                0 => ExitCode::FAILURE,
                status => ExitCode::from(status),
            }
        }
    }
}

// fitxategia programa bihurtu, luzapenaren arabera
fn load(path: &Path) -> Result<Program, String> {
    let name = path.display().to_string();
    let read_err = |e: std::io::Error| t!("cannot read {}: {}", "ezin da {} irakurri: {}", name, e);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "qx" | "qexe" | "oexe" => {
            let data = qoa::state::read_file(path).map_err(read_err)?;
            Program::from_bytes(&data).map_err(|e| format!("{}: {}", name, e))
        }
        _ => {
            let src = qoa::state::read_text(path).map_err(read_err)?;
            load_source(&ext, &src, &name)
        }
    }
}

// QOA_REQUIRE_SEAL=1: zigilurik gabeko iturburuak ez dira onartzen (zigilu-lerro osoa galdu edo hondatu bada ere)
fn require_seal(src: &[u8], name: &str) -> Result<(), String> {
    if std::env::var_os("QOA_REQUIRE_SEAL").is_some_and(|v| !v.is_empty() && v != "0")
        && qoa::seal::find(src, qoa::seal::Style::for_name(name)).is_none()
    {
        return Err(t!(
            "{} is not sealed, and QOA_REQUIRE_SEAL is set: only sealed sources run here (`qoa seal {}` seals it)",
            "{} ez dago zigilatuta, eta QOA_REQUIRE_SEAL ezarrita dago: hemen iturburu zigilatuak bakarrik exekutatzen dira (`qoa seal {}`-k zigilatzen du)",
            name,
            name
        ));
    }
    Ok(())
}

// iturburu bat programa bihurtu, luzapenaren arabera (qg, qasm, edo QOA arrunta)
fn load_source(ext: &str, src: &str, name: &str) -> Result<Program, String> {
    require_seal(src.as_bytes(), name)?;
    match ext {
        "qg" => qoa::glyph::assemble(src, name).map_err(|d| {
            format!(
                "{}{}\n",
                asm::render(&d, src, name),
                t!(
                    "{} error(s) in {}",
                    "{} errore {} fitxategian",
                    d.len(),
                    name
                )
            )
        }),
        "qasm" => {
            let qoa_src = qoa::qasm::to_qoa(src, name).map_err(|e| format!("{}: {}", name, e))?;
            asm::assemble(&qoa_src, name).map_err(|d| {
                t!(
                    "{}: the imported program did not assemble (this is a QOA bug, please report it):\n{}",
                    "{}: inportatutako programa ez da mihiztatu (QOA-ren akats bat da, jakinarazi mesedez):\n{}",
                    name,
                    asm::render(&d, &qoa_src, &format!("{} (import)", name))
                )
            })
        }
        _ => asm::assemble(src, name).map_err(|d| {
            format!(
                "{}{}\n",
                asm::render(&d, src, name),
                t!(
                    "{} error(s) in {}",
                    "{} errore {} fitxategian",
                    d.len(),
                    name
                )
            )
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    file: &Path,
    shots: Option<u64>,
    seed: Option<u64>,
    noise: f64,
    loss: f64,
    noise_model: Option<&Path>,
    show_state: bool,
    top: usize,
    json: bool,
    max_steps: u64,
    gpu: Option<qoa::gpu::Api>,
    precision: qoa::vm::Precision,
    mpi: bool,
    distribute: bool,
    quiet: bool,
    args: Vec<String>,
    device: Option<(String, String)>,
) -> Result<(), String> {
    let device = match device {
        Some((cmd, target)) => Some((cmd, parse_target(&target)?)),
        None => None,
    };
    let team = join_team(mpi)?;
    let rank = team.as_ref().map_or(0, |t| t.rank());
    let program = load(file)?;
    // QOA_GPU_DEVICE=cpu: prozesu hau CPUan, --gpu-rekin ere (MPI lan misto batean, GPUak eta CPUak batera)
    let cpu_here =
        std::env::var("QOA_GPU_DEVICE").is_ok_and(|v| v.trim().eq_ignore_ascii_case("cpu"));
    let gpu = match gpu.filter(|_| !cpu_here) {
        Some(api) => {
            let g = qoa::gpu::Gpu::open_with(api, qoa::gpu::local_index())
                .map_err(|e| format!("--gpu: {e}"))?;
            Some(std::sync::Arc::new(g))
        }
        None => None,
    };
    // gailu baten kalibrazioa (qoa-noise-1) ala mahai gaineko banku bat (qoa-bench-1)
    let (model, bench) = match noise_model {
        Some(path) => match qoa::noise::load_any(path)? {
            qoa::noise::AnyModel::Device(m) => (Some(std::sync::Arc::new(m)), None),
            qoa::noise::AnyModel::Bench(b) => (None, Some(std::sync::Arc::new(b))),
        },
        None => (None, None),
    };
    let config = RunConfig {
        shots: shots.unwrap_or(1024),
        seed: seed.unwrap_or_else(rand::random::<u64>),
        noise,
        loss,
        max_steps,
        gpu,
        model,
        bench,
        precision,
        team: team.clone(),
        distribute,
        args,
        input: None,
    };
    if quiet {
        // programaren irteera bakarrik, bufferrean (byte bakoitza idazketa bat ez izateko)
        let stdout = std::io::stdout();
        let terminal = std::io::IsTerminal::is_terminal(&stdout);
        let mut out = std::io::BufWriter::new(Out(qoa::term::Screen::new(stdout.lock(), terminal)));
        let output = vm::run(&program, &config, &mut out);
        std::io::Write::flush(&mut out).map_err(|e| {
            t!(
                "cannot write the program's output: {}",
                "ezin da programaren irteera idatzi: {}",
                e
            )
        })?;
        if rank != 0 {
            return Ok(());
        }
        let output = output.map_err(|e| format!("{}: {}", file.display(), e))?;
        EXIT_STATUS.store(
            output.exit_status.unwrap_or(0),
            std::sync::atomic::Ordering::Relaxed,
        );
        return Ok(());
    }
    let start = Instant::now();
    // json moduan programaren irteera bildu egiten da; testu moduan zuzenean ateratzen da
    let mut captured: Vec<u8> = Vec::new();
    let mut printed = Counting {
        inner: Out(qoa::term::Screen::new(
            std::io::stdout(),
            std::io::IsTerminal::is_terminal(&std::io::stdout()),
        )),
        bytes: 0,
    };
    let output = if let Some((cmd, target)) = &device {
        qoa::device::run(&program, &config, cmd, *target)
    } else if json {
        vm::run(&program, &config, &mut captured)
    } else {
        vm::run(&program, &config, &mut printed)
    };
    // taldean, 0. rank-ak bakarrik du emaitza osoa (eta errorea jakinarazten du)
    if rank != 0 {
        return Ok(());
    }
    let output = output.map_err(|e| format!("{}: {}", file.display(), e))?;
    let elapsed = start.elapsed();
    let rep = report::build(ReportInput {
        program: &program,
        config: &config,
        seed_given: seed.is_some(),
        shots_given: shots.is_some(),
        output: &output,
        elapsed,
        show_state,
        top,
    });
    if json {
        let mut value = serde_json::to_value(&rep).map_err(|e| e.to_string())?;
        value["program_output"] =
            serde_json::Value::String(String::from_utf8_lossy(&captured).into_owned());
        println!(
            "{}",
            serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
        );
    } else {
        // programak zerbait inprimatu badu, txostena lerro huts batez bereizi
        if printed.bytes > 0 {
            println!();
        }
        print!(
            "{}",
            qoa::term::for_stdout(&report::render(&rep, &Style::detect()))
        );
    }
    EXIT_STATUS.store(
        output.exit_status.unwrap_or(0),
        std::sync::atomic::Ordering::Relaxed,
    );
    Ok(())
}

// errendimendu-proba: atal bakoitza abiaraztean aurrerapena stderr-era, emaitza stdout-era
fn cmd_benchmark(quick: bool, no_gpu: bool, json: bool) -> Result<(), String> {
    let style = Style::detect();
    let report = qoa::bench::run(quick, !no_gpu, |name| {
        eprint!("\r{} {:<12}", ts!("measuring", "neurtzen"), name);
    });
    eprint!("\r{:<40}\r", "");
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else {
        print!("{}", qoa::bench::render(&report, &style));
    }
    Ok(())
}

// `--mpi`: MPI taldean sartu (ezaugarriarekin konpilatuta), bestela zer egin azaldu
#[cfg(feature = "mpi")]
fn join_team(mpi: bool) -> Result<Option<std::sync::Arc<dyn qoa::team::Team>>, String> {
    if !mpi {
        return Ok(None);
    }
    let world = qoa::mpi::World::init()?;
    Ok(Some(std::sync::Arc::new(world)))
}

#[cfg(not(feature = "mpi"))]
fn join_team(mpi: bool) -> Result<Option<std::sync::Arc<dyn qoa::team::Team>>, String> {
    if mpi {
        return Err(ts!(
            "--mpi: this QOA was built without MPI (`--no-default-features`). Build it with the default features or `--features mpi` (nothing else is needed: the MPI library is loaded at run time), then start it with mpirun or srun",
            "--mpi: QOA hau MPI gabe konpilatu da (`--no-default-features`). konpilatu ezaugarri lehenetsiekin edo `--features mpi`-rekin (ez da beste ezer behar: MPI liburutegia exekuzioan kargatzen da), eta abiarazi mpirun edo srun-ekin"
        )
        .into());
    }
    Ok(None)
}

// idatzitako byteak zenbatzen dituen idazlea (programak zerbait inprimatu duen jakiteko)
struct Counting<W: std::io::Write> {
    inner: W,
    bytes: usize,
}

impl<W: std::io::Write> std::io::Write for Counting<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes += n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn cmd_check(file: &Path, target: Option<&str>) -> Result<(), String> {
    let target = target.map(parse_target).transpose()?;
    let p = load(file)?;
    if let Some(target) = target {
        let problems = qoa::target::check(&p, target);
        if !problems.is_empty() {
            let mut msg = t!(
                "{} cannot run on {} (docs/bench.md):\n",
                "{} ezin da {}-n exekutatu (docs/bench.eu.md):\n",
                p.name,
                target.name()
            );
            for e in problems {
                msg.push_str(&format!("  {}\n", e));
            }
            return Err(msg.trim_end().to_string());
        }
    }
    let dim = p.state_dim().unwrap_or(0);
    let style = Style::detect();
    let label = |s: &str| {
        let padded = format!("{:<9}", s);
        if style.color {
            format!("\x1b[2m{}\x1b[0m", padded)
        } else {
            padded
        }
    };
    let gates = p
        .ops
        .iter()
        .filter(|o| !o.gate_qubits().is_empty() && o.optical_modes().is_empty())
        .count();
    let optical = p
        .ops
        .iter()
        .filter(|o| !o.optical_modes().is_empty())
        .count();
    let readouts = p
        .ops
        .iter()
        .filter(|o| {
            o.is_diagonal_readout() || matches!(o, Op::Measure { .. } | Op::EntangleSwap { .. })
        })
        .count();
    let other = p.ops.len() - gates - optical - readouts;
    println!(
        "{}  {}",
        label(ts!("program", "programa")),
        t!(
            "{}: assembles without errors",
            "{}: errorerik gabe mihiztatzen da",
            p.name
        )
    );
    let mut system = t!("{} qubit(s)", "{} qubit", p.n_qubits);
    if p.n_modes > 0 {
        system.push_str(&if p.cutoff_declared {
            t!(
                ", {} mode(s) of 0..{} photons",
                ", {} modu, 0..{} fotoikoak",
                p.n_modes,
                p.cutoff - 1
            )
        } else {
            t!(
                ", {} mode(s) of 0..{} photons (default cutoff)",
                ", {} modu, 0..{} fotoikoak (muga lehenetsia)",
                p.n_modes,
                p.cutoff - 1
            )
        });
    }
    system.push_str(&t!(
        " = {} amplitudes, {}",
        " = {} anplitude, {}",
        dim,
        format_bytes(dim * 16)
    ));
    println!("{}  {}", label(ts!("system", "sistema")), system);
    let (limit, source) = qoa::state::memory_limit();
    if dim * 16 > limit {
        println!(
            "{}  {}",
            label(""),
            t!(
                "too large to simulate here ({}: {}; QOA_MAX_MEMORY changes it)",
                "handiegia hemen simulatzeko ({}: {}; QOA_MAX_MEMORY-k aldatzen du)",
                source,
                format_bytes(limit)
            )
        );
    }
    println!(
        "{}  {}",
        label(ts!("ops", "eragik.")),
        t!(
            "{} in total: {} qubit gate(s), {} optical, {} measurement(s), {} classical/control/output",
            "{} guztira: {} qubit-ate, {} optiko, {} neurketa, {} klasiko/kontrol/irteera",
            p.ops.len(),
            gates,
            optical,
            readouts,
            other
        )
    );
    let how = if !p.has_readout() {
        t!(
            "it measures nothing: `qoa run` shows the exact final state",
            "ez du ezer neurtzen: `qoa run`-ek amaierako egoera zehatza erakusten du"
        )
    } else if p.sampling_split().is_some() {
        t!(
            "every measurement is at the end: one simulation, every shot sampled from it (without --noise/--loss)",
            "neurketa guztiak amaieran daude: simulazio bakarra, eta shot guztiak hartatik laginduta (--noise/--loss gabe)"
        )
    } else {
        t!(
            "mid-circuit measurement, branching or randomness: each shot is simulated from the start",
            "erdiko neurketa, adarkatzea edo ausazkotasuna: shot bakoitza hasieratik simulatzen da"
        )
    };
    println!("{}  {}", label("shots"), how);
    let inspects = p.ops.iter().any(|o| {
        matches!(
            o,
            Op::DumpState | Op::Expect { .. } | Op::BellCheck { .. } | Op::MeanPhoton { .. }
        )
    });
    if inspects {
        println!(
            "{}  {}",
            label(ts!("note", "oharra")),
            t!(
                "uses simulator-only inspection (DUMPSTATE/EXPECT/BELLCHECK/MEANPHOTON); a real device cannot do this",
                "simulagailuko ikuskapena erabiltzen du (DUMPSTATE/EXPECT/BELLCHECK/MEANPHOTON); benetako gailu batek ezin du hori egin"
            )
        );
    }
    if let Some(target) = target {
        println!(
            "{}  {}",
            label(ts!("target", "helburua")),
            t!(
                "runs on {} (docs/bench.md)",
                "{}-n dabil (docs/bench.eu.md)",
                target.name()
            )
        );
    }
    Ok(())
}

fn cmd_compile(
    file: &Path,
    output: Option<PathBuf>,
    assembler: Option<&Path>,
    self_hosted: bool,
) -> Result<(), String> {
    let (p, bytes) = match assembler {
        _ if self_hosted => {
            let tool = qoa::tools::find("asm").expect("asm tresna binarioan dago");
            assemble_with(&tool.program()?, &tool.path(), file)?
        }
        None => {
            let p = load(file)?;
            let bytes = p.to_bytes();
            (p, bytes)
        }
        Some(a) => assemble_with(&load(a)?, &a.display().to_string(), file)?,
    };
    let out = output.unwrap_or_else(|| file.with_extension("qx"));
    qoa::state::write_file(&out, &bytes).map_err(|e| {
        t!(
            "cannot write {}: {}",
            "ezin da {} idatzi: {}",
            out.display(),
            e
        )
    })?;
    println!(
        "{}",
        t!(
            "wrote {} ({} operations, {} bytes)",
            "{} idatzita ({} eragiketa, {} byte)",
            out.display(),
            p.ops.len(),
            bytes.len()
        )
    );
    Ok(())
}

// QOA programa batekin mihiztatu (adib. src/qoa/asm.qoa): iturburua sarrera gisa, fitxategi-izena argumentu
// gisa, eta programak idatzitako byteak .qx-a dira. haren mezuak errore-irteerara doaz zuzenean
fn assemble_with(tool: &Program, label: &str, file: &Path) -> Result<(Program, Vec<u8>), String> {
    let name = file.display().to_string();
    let source = qoa::state::read_file(file)
        .map_err(|e| t!("cannot read {}: {}", "ezin da {} irakurri: {}", name, e))?;
    require_seal(&source, &name)?;
    let config = RunConfig {
        shots: 1,
        seed: 0,
        max_steps: u64::MAX,
        args: vec![name.clone()],
        input: Some(std::sync::Arc::new(source)),
        ..RunConfig::default()
    };
    let mut bytes = Vec::new();
    let output = vm::run(tool, &config, &mut bytes).map_err(|e| format!("{}: {}", label, e))?;
    match output.exit_status.unwrap_or(0) {
        0 => {}
        status => {
            // mihiztatzaileak bere erroreak jada inprimatu ditu
            EXIT_STATUS.store(status, std::sync::atomic::Ordering::Relaxed);
            return Err(String::new());
        }
    }
    let p = Program::from_bytes(&bytes).map_err(|e| {
        t!(
            "{} did not write a QOA binary for {}: {}",
            "{}-k ez du QOA bitarrik idatzi {}-rentzat: {}",
            label,
            name,
            e
        )
    })?;
    Ok((p, bytes))
}

fn parse_target(name: &str) -> Result<qoa::target::Target, String> {
    qoa::target::Target::parse(name).ok_or_else(|| {
        let names: Vec<&str> = qoa::target::Target::ALL.iter().map(|t| t.name()).collect();
        t!(
            "unknown target '{}' (known: {})",
            "'{}' helburu ezezaguna (ezagunak: {})",
            name,
            names.join(", ")
        )
    })
}

// ordezko banku-kontrolagailua: lanak sarrera estandarrean, erantzunak irteera estandarrean
fn cmd_bench_sim(model: Option<&Path>) -> Result<(), String> {
    let model = match model {
        None => None,
        Some(path) => match qoa::noise::load_any(path)? {
            qoa::noise::AnyModel::Bench(b) => Some(b),
            qoa::noise::AnyModel::Device(_) => {
                return Err(t!(
                    "{} is a device calibration; bench-sim needs a bench model (format {})",
                    "{} gailu baten kalibrazioa da; bench-sim-ek banku-eredu bat behar du ({} formatua)",
                    path.display(),
                    qoa::noise::BENCH_FORMAT
                ));
            }
        },
    };
    let stdin = std::io::stdin();
    qoa::device::serve(model.as_ref(), stdin.lock(), std::io::stdout().lock())
}

// binarioan sartutako tresnak: zerrenda, edo iturburuak karpeta batean (QOA arrunta eta glifoak)
fn cmd_tools(write: Option<&Path>) -> Result<(), String> {
    let Some(dir) = write else {
        for tool in qoa::tools::TOOLS {
            println!(
                "{:<6} {} ({} {})\n       {}",
                tool.name,
                tool.path(),
                tool.source.lines().count(),
                t!("lines", "lerro"),
                if qoa::lang::is_eu() {
                    tool.about_eu
                } else {
                    tool.about_en
                }
            );
        }
        println!(
            "{}",
            t!(
                "\nqoa compile --self-hosted file.qoa assembles with the built-in assembler written in QOA; qoa tools --write DIR writes the sources",
                "\nqoa compile --self-hosted fitxategia.qoa-k QOAn idatzitako mihiztatzaile integratuarekin mihiztatzen du; qoa tools --write KARPETA-k iturburuak idazten ditu"
            )
        );
        return Ok(());
    };
    std::fs::create_dir_all(dir).map_err(|e| {
        t!(
            "cannot create {}: {}",
            "ezin da {} sortu: {}",
            dir.display(),
            e
        )
    })?;
    for tool in qoa::tools::TOOLS {
        for (file, text) in [
            (tool.file.to_string(), tool.source.to_string()),
            (tool.glyph_file(), tool.glyph_source()?),
        ] {
            let path = dir.join(file);
            qoa::state::write_file(&path, text.as_bytes()).map_err(|e| {
                t!(
                    "cannot write {}: {}",
                    "ezin da {} idatzi: {}",
                    path.display(),
                    e
                )
            })?;
            println!("{}", t!("wrote {}", "{} idatzita", path.display()));
        }
    }
    Ok(())
}

fn write_or_print(output: Option<PathBuf>, text: &str) -> Result<(), String> {
    match output {
        Some(path) => {
            qoa::state::write_file(&path, text.as_bytes()).map_err(|e| {
                t!(
                    "cannot write {}: {}",
                    "ezin da {} idatzi: {}",
                    path.display(),
                    e
                )
            })?;
            eprintln!("{}", t!("wrote {}", "{} idatzita", path.display()));
        }
        None => print!("{}", qoa::term::for_stdout(text)),
    }
    Ok(())
}

// `qoa export`-en glifo-aukerak, egiaztatuta
fn glyph_options(
    to: ExportFormat,
    width: usize,
    names: &str,
    long_names: bool,
    no_ranges: bool,
    hex: bool,
) -> Result<qoa::disasm::GlyphOptions, String> {
    let glyph_flags = width != 100 || names != "Q,C,L" || no_ranges || hex;
    if glyph_flags && to != ExportFormat::Glyph {
        return Err(t!(
            "--width, --names, --no-ranges and --hex only apply to --to glyph",
            "--width, --names, --no-ranges eta --hex --to glyph-ekin bakarrik dabiltza"
        ));
    }
    if long_names
        && !matches!(
            to,
            ExportFormat::Glyph | ExportFormat::Qoa | ExportFormat::Listing
        )
    {
        return Err(t!(
            "--long-names only applies to --to glyph, qoa and listing",
            "--long-names --to glyph, qoa eta listing-ekin bakarrik dabil"
        ));
    }
    let parts: Vec<&str> = names.split(',').map(str::trim).collect();
    let [qubits, bits, modes] = parts[..] else {
        return Err(t!(
            "--names takes three names separated by commas, like Q,C,L",
            "--names-ek komaz bereizitako hiru izen hartzen ditu, Q,C,L bezala"
        ));
    };
    Ok(qoa::disasm::GlyphOptions {
        width,
        qubits: qubits.into(),
        bits: bits.into(),
        modes: modes.into(),
        long_names,
        ranges: !no_ranges,
        hex,
        dense: false,
    })
}

fn cmd_export(
    file: &Path,
    to: ExportFormat,
    output: Option<PathBuf>,
    shots: u64,
    glyph: &qoa::disasm::GlyphOptions,
) -> Result<(), String> {
    let p = load(file)?;
    let local = |r: Result<String, String>| r.map_err(|e| format!("{}: {}", p.name, e));
    let result = match to {
        ExportFormat::Qasm => qoa::qasm::from_program(&p),
        ExportFormat::Qasm3 => qoa::qasm::from_program_v3(&p),
        ExportFormat::Ionq => qoa::ionq::from_program(&p, shots),
        ExportFormat::Glyph => {
            return write_or_print(output, &local(qoa::disasm::to_glyph(&p, glyph))?);
        }
        ExportFormat::Qoa => {
            let text = if glyph.long_names {
                qoa::disasm::to_qoa_long(&p)
            } else {
                qoa::disasm::to_qoa(&p)
            };
            return write_or_print(output, &local(text)?);
        }
        ExportFormat::Listing => {
            let text = if glyph.long_names {
                qoa::disasm::listing_long(&p)
            } else {
                qoa::disasm::listing(&p)
            };
            return write_or_print(output, &local(text)?);
        }
        ExportFormat::Json => return write_or_print(output, &local(qoa::disasm::to_json(&p))?),
        ExportFormat::Dot => return write_or_print(output, &local(qoa::disasm::to_dot(&p))?),
    };
    match result {
        Ok(text) => write_or_print(output, &text),
        Err(errors) => {
            let target = match to {
                ExportFormat::Qasm => "OpenQASM 2.0",
                ExportFormat::Qasm3 => "OpenQASM 3",
                _ => "IonQ",
            };
            let mut msg = t!(
                "{} cannot be exported to {}:\n",
                "{} ezin da {}-ra esportatu:\n",
                p.name,
                target
            );
            for e in errors {
                msg.push_str(&format!("  {}\n", e));
            }
            Err(msg.trim_end().to_string())
        }
    }
}

// iturburu bat zigilatu: aurreko zigilua ordezten da, eta programa mihiztatzen dela egiaztatzen da idatzi aurretik
fn cmd_seal(file: &Path) -> Result<(), String> {
    let name = file.display().to_string();
    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if matches!(ext.as_str(), "qx" | "qexe" | "oexe") {
        return Err(t!(
            "{}: a .qx file already carries its own SHA-256; seal its source instead",
            "{}: .qx fitxategi batek bere SHA-256-a badu jada; zigilatu haren iturburua",
            name
        ));
    }
    let src = qoa::state::read_text(file)
        .map_err(|e| t!("cannot read {}: {}", "ezin da {} irakurri: {}", name, e))?;
    let sealed = qoa::seal::seal(src.as_bytes(), qoa::seal::Style::for_name(&name));
    let text = String::from_utf8(sealed).expect("utf-8 + ascii");
    load_source(&ext, &text, &name)?;
    qoa::state::write_file(file, text.as_bytes())
        .map_err(|e| t!("cannot write {}: {}", "ezin da {} idatzi: {}", name, e))?;
    let last = text.trim_end().rsplit('\n').next().unwrap_or("");
    println!(
        "{}",
        t!(
            "sealed {}: {}",
            "{} zigilatuta: {}",
            qoa::term::for_stdout(&name),
            last.trim_start_matches("// ")
        )
    );
    Ok(())
}

// `qoa export --to glyph --keep-source`: .qoa iturburua glifora, lerroz lerro
fn cmd_keep_source(file: &Path, to: ExportFormat, output: Option<PathBuf>) -> Result<(), String> {
    let name = file.display().to_string();
    if to != ExportFormat::Glyph || !name.to_ascii_lowercase().ends_with(".qoa") {
        return Err(t!(
            "--keep-source converts a .qoa source to --to glyph",
            "--keep-source-k .qoa iturburu bat --to glyph-era bihurtzen du"
        ));
    }
    let src = qoa::state::read_text(file)
        .map_err(|e| t!("cannot read {}: {}", "ezin da {} irakurri: {}", name, e))?;
    let text = qoa::convert::qoa_to_glyph(&src, &name)?;
    write_or_print(output, &text)
}

fn cmd_import(file: &Path, output: Option<PathBuf>) -> Result<(), String> {
    let name = file.display().to_string();
    let src = qoa::state::read_text(file)
        .map_err(|e| t!("cannot read {}: {}", "ezin da {} irakurri: {}", name, e))?;
    let is_glyph = file
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("qg"));
    let text = if is_glyph {
        qoa::glyph::expand_file(&src, &name)
            .map_err(|d| asm::render(&d, &src, &name))?
            .source
    } else {
        qoa::qasm::to_qoa(&src, &name).map_err(|e| format!("{}: {}", name, e))?
    };
    write_or_print(output, &text)
}

fn cmd_isa(name: Option<String>, markdown: bool) -> Result<(), String> {
    if markdown {
        print!("{}", isa::markdown_reference());
        return Ok(());
    }
    let label = |s: &str| format!("{:<9}", s);
    if let Some(n) = name {
        let spec = isa::lookup(&n).ok_or_else(|| {
            t!(
                "no instruction or alias named '{}'",
                "ez dago '{}' izeneko agindu edo ezizenik",
                n
            )
        })?;
        if let Some(why) = spec.removed_reason() {
            println!(
                "{}",
                t!("{}: removed in QOA 0.4", "{}: QOA 0.4-n kendua", spec.name)
            );
            println!("{}  {}", label(ts!("why", "zergatik")), why);
            return Ok(());
        }
        println!("{}  {}", label(ts!("usage", "erabilera")), spec.usage());
        println!("{}  {}", label(ts!("short", "laburra")), isa::short(spec));
        println!("{}  {}", label(ts!("does", "egiten du")), spec.summary());
        if spec.status == Status::SimOnly {
            println!(
                "{}  {}",
                label(ts!("note", "oharra")),
                t!(
                    "simulator only: a real device cannot do this without disturbing the state",
                    "simulagailuan bakarrik: benetako gailu batek ezin du hau egin egoera aldatu gabe"
                )
            );
        }
        if spec.broadcast {
            println!(
                "{}  {}",
                label(ts!("ranges", "tarteak")),
                t!(
                    "the first operand may be a range, e.g. {} q0..q3",
                    "lehen eragigaia tarte bat izan daiteke, adib. {} q0..q3",
                    spec.name
                )
            );
        }
        if !spec.aliases.is_empty() {
            println!(
                "{}  {}",
                label(ts!("aliases", "ezizenak")),
                spec.aliases.join(", ")
            );
        }
        let mut kinds: Vec<char> = spec.operand_specs().iter().map(|o| o.kind).collect();
        kinds.dedup();
        for k in kinds {
            println!(
                "{}  {}",
                label(ts!("operand", "eragigaia")),
                isa::kind_help(k)
            );
        }
        println!("{}  {}", label(ts!("group", "taldea")), spec.group.title());
        return Ok(());
    }
    for group in isa::Group::ALL {
        let specs: Vec<&isa::Spec> = isa::SPECS
            .iter()
            .filter(|s| s.group == group && !s.is_removed())
            .collect();
        if specs.is_empty() {
            continue;
        }
        println!("{}", group.title());
        for s in specs {
            let marker = if s.status == Status::SimOnly {
                ts!(" [simulator only]", " [simulagailuan bakarrik]")
            } else {
                ""
            };
            let short = isa::short(s);
            let short = if short == s.name {
                String::new()
            } else {
                short.to_string()
            };
            println!("  {:<5} {:<46} {}{}", short, s.usage(), s.summary(), marker);
        }
        println!();
    }
    let removed = isa::SPECS.iter().filter(|s| s.is_removed()).count();
    println!(
        "{}",
        t!(
            "The first column is each instruction's short name (the same instruction, preferred). {} instructions from QOA 0.3 were removed; `qoa isa NAME` says why. `qoa isa --markdown` prints all of this as docs.",
            "Lehen zutabea agindu bakoitzaren izen laburra da (agindu bera, hobetsia). QOA 0.3-ko {} agindu kendu dira; `qoa isa IZENA`-k zergatik esaten du. `qoa isa --markdown`-ek hau guztia dokumentazio gisa inprimatzen du.",
            removed
        )
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_visual(
    input: &Path,
    output: &Path,
    raw: Option<visualizer::Raw>,
    mode: visualizer::Mode,
    light: qoa::light::Light,
    block: usize,
    resolution: &str,
    fps: u32,
    phase_space: visualizer::PhaseSpace,
    ltr: bool,
    rtl: bool,
    seed: Option<u64>,
    ffmpeg_flags: &[String],
    ffmpeg_args: &[String],
) -> Result<(), String> {
    let (width, height) = resolution
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
        .ok_or_else(|| {
            t!(
                "resolution '{}' should look like 1920x1080",
                "'{}' bereizmenak 1920x1080 itxura izan behar du",
                resolution
            )
        })?;
    let direction = if ltr {
        SpectrumDirection::Ltr
    } else if rtl {
        SpectrumDirection::Rtl
    } else {
        SpectrumDirection::None
    };
    visualizer::run(&visualizer::Options {
        input,
        output,
        raw,
        mode,
        light,
        block,
        width,
        height,
        fps,
        phase_space,
        direction,
        seed: seed.unwrap_or_else(rand::random::<u64>),
        ffmpeg_flags,
        ffmpeg_args,
    })
    .map_err(|e| format!("visual: {}", e))
}
