// qoa komando-lerroko tresna. komando bakoitzak gauza bat egiten du eta zuzenean esaten du zer egin duen.
// programak .qoa (iturburua), .qg (glifo-sintaxia), .qasm (openqasm, zuzenean inportatua) edo .qx (konpilatua)
// izan daitezke.
// laguntza-testuak eta mezuak euskaraz edo ingelesez daude (ikus src/lang.rs).

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
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
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
    },
    #[command(about = ts!("Assemble a program and report what it needs, without running it", "Programa bat mihiztatu eta zer behar duen esan, exekutatu gabe"))]
    Check {
        #[arg(help = ts!("program file", "programa-fitxategia"))]
        file: PathBuf,
    },
    #[command(about = ts!("Assemble a program into a .qx binary", "Programa bat .qx bitar batean mihiztatu"))]
    Compile {
        #[arg(help = ts!("program file", "programa-fitxategia"))]
        file: PathBuf,
        #[arg(short, long, help = ts!("output file (default: same name with .qx)", "irteera-fitxategia (lehenetsia: izen bera .qx-rekin)"))]
        output: Option<PathBuf>,
    },
    #[command(about = ts!("Convert a program to OpenQASM 2.0 or IonQ json", "Programa bat OpenQASM 2.0 edo IonQ json bihurtu"))]
    Export {
        #[arg(help = ts!("program file", "programa-fitxategia"))]
        file: PathBuf,
        #[arg(long, value_enum, required_unless_present = "to_glyph", help = ts!(
            "target: qasm (OpenQASM 2.0), ionq (IonQ JSON), glyph (glyph syntax), qoa (canonical plain QOA, also from a .qx binary), listing (qoa with each operation's address and source line), json (the program as data), dot (control-flow graph for Graphviz)",
            "helburua: qasm (OpenQASM 2.0), ionq (IonQ JSON), glyph (glifo-sintaxia), qoa (QOA arrunt kanonikoa, .qx bitar batetik ere), listing (qoa eragiketa bakoitzaren helbide eta iturburu-lerroarekin), json (programa datu gisa), dot (kontrol-fluxuaren grafoa graphviz-erako)"
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
            "glyph: write MEASURE and CNOT instead of M and CX",
            "glifoa: idatzi MEASURE eta CNOT, M eta CX ordez"
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
    },
    #[command(about = ts!(
        "Convert an OpenQASM 2.0 file (.qasm) or a glyph-syntax file (.qg) into plain QOA source",
        "OpenQASM 2.0 fitxategi bat (.qasm) edo glifo-sintaxiko bat (.qg) QOA iturburu arrunt bihurtu"
    ))]
    Import {
        #[arg(help = ts!(".qasm or .qg file", ".qasm edo .qg fitxategia"))]
        file: PathBuf,
        #[arg(short, long, help = ts!("output file (default: print to the terminal)", "irteera-fitxategia (lehenetsia: terminalean inprimatu)"))]
        output: Option<PathBuf>,
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
        #[arg(help = ts!("input audio file", "sarrerako audio-fitxategia"))]
        input: PathBuf,
        #[arg(help = ts!("output video file", "irteerako bideo-fitxategia"))]
        output: PathBuf,
        #[arg(long, default_value = "1920x1080", help = ts!("resolution, e.g. 1920x1080", "bereizmena, adib. 1920x1080"))]
        resolution: String,
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
enum ExportFormat {
    // openqasm 2.0 (qubit-zatia)
    Qasm,
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

fn main() -> ExitCode {
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
    let result = match cli.command {
        Command::Run {
            file,
            shots,
            seed,
            noise,
            loss,
            state,
            top,
            json,
            max_steps,
            gpu,
        } => cmd_run(
            &file, shots, seed, noise, loss, state, top, json, max_steps, gpu,
        ),
        Command::Check { file } => cmd_check(&file),
        Command::Compile { file, output } => cmd_compile(&file, output),
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
        } => {
            let to = to.unwrap_or(ExportFormat::Glyph);
            let _ = to_glyph;
            glyph_options(to, width, &names, long_names, no_ranges, hex)
                .and_then(|g| cmd_export(&file, to, output, shots, &g))
        }
        Command::Import { file, output } => cmd_import(&file, output),
        Command::Isa { name, markdown } => cmd_isa(name, markdown),
        Command::Visual {
            input,
            output,
            resolution,
            fps,
            ltr,
            rtl,
            seed,
            ffmpeg_flags,
            ffmpeg_args,
        } => cmd_visual(
            &input,
            &output,
            &resolution,
            fps,
            ltr,
            rtl,
            seed,
            &ffmpeg_flags,
            &ffmpeg_args,
        ),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            // mihiztatzailearen erroreek beren formatua dute; besteei aurrizkia jarri
            if message.starts_with(ts!("error:", "errorea:")) {
                eprint!("{}", message);
            } else {
                eprintln!("{} {}", ts!("error:", "errorea:"), message);
            }
            ExitCode::FAILURE
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
            let data = std::fs::read(path).map_err(read_err)?;
            Program::from_bytes(&data).map_err(|e| format!("{}: {}", name, e))
        }
        "qg" => {
            let src = std::fs::read_to_string(path).map_err(read_err)?;
            qoa::glyph::assemble(&src, &name).map_err(|d| {
                format!(
                    "{}{}\n",
                    asm::render(&d, &src, &name),
                    t!(
                        "{} error(s) in {}",
                        "{} errore {} fitxategian",
                        d.len(),
                        name
                    )
                )
            })
        }
        "qasm" => {
            let src = std::fs::read_to_string(path).map_err(read_err)?;
            let qoa_src = qoa::qasm::to_qoa(&src, &name).map_err(|e| format!("{}: {}", name, e))?;
            asm::assemble(&qoa_src, &name).map_err(|d| {
                t!(
                    "{}: the imported program did not assemble (this is a QOA bug, please report it):\n{}",
                    "{}: inportatutako programa ez da mihiztatu (QOA-ren akats bat da, jakinarazi mesedez):\n{}",
                    name,
                    asm::render(&d, &qoa_src, &format!("{} (import)", name))
                )
            })
        }
        _ => {
            let src = std::fs::read_to_string(path).map_err(read_err)?;
            asm::assemble(&src, &name).map_err(|d| {
                format!(
                    "{}{}\n",
                    asm::render(&d, &src, &name),
                    t!(
                        "{} error(s) in {}",
                        "{} errore {} fitxategian",
                        d.len(),
                        name
                    )
                )
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_run(
    file: &Path,
    shots: Option<u64>,
    seed: Option<u64>,
    noise: f64,
    loss: f64,
    show_state: bool,
    top: usize,
    json: bool,
    max_steps: u64,
    gpu: bool,
) -> Result<(), String> {
    let program = load(file)?;
    let gpu = if gpu {
        let g = qoa::gpu::Gpu::open().map_err(|e| format!("--gpu: {e}"))?;
        Some(std::sync::Arc::new(g))
    } else {
        None
    };
    let config = RunConfig {
        shots: shots.unwrap_or(1024),
        seed: seed.unwrap_or_else(rand::random::<u64>),
        noise,
        loss,
        max_steps,
        gpu,
    };
    let start = Instant::now();
    // json moduan programaren irteera bildu egiten da; testu moduan zuzenean ateratzen da
    let mut captured: Vec<u8> = Vec::new();
    let mut printed = Counting {
        inner: std::io::stdout(),
        bytes: 0,
    };
    let output = if json {
        vm::run(&program, &config, &mut captured)
    } else {
        vm::run(&program, &config, &mut printed)
    };
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
        print!("{}", report::render(&rep, &Style::detect()));
    }
    Ok(())
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

fn cmd_check(file: &Path) -> Result<(), String> {
    let p = load(file)?;
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
    if dim > qoa::state::MAX_AMPLITUDES {
        println!(
            "{}  {}",
            label(""),
            t!(
                "too large to simulate (the limit is {})",
                "handiegia simulatzeko ({} da muga)",
                format_bytes(qoa::state::MAX_AMPLITUDES * 16)
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
    Ok(())
}

fn cmd_compile(file: &Path, output: Option<PathBuf>) -> Result<(), String> {
    let p = load(file)?;
    let out = output.unwrap_or_else(|| file.with_extension("qx"));
    let bytes = p.to_bytes();
    std::fs::write(&out, &bytes).map_err(|e| {
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

fn write_or_print(output: Option<PathBuf>, text: &str) -> Result<(), String> {
    match output {
        Some(path) => {
            std::fs::write(&path, text).map_err(|e| {
                t!(
                    "cannot write {}: {}",
                    "ezin da {} idatzi: {}",
                    path.display(),
                    e
                )
            })?;
            eprintln!("{}", t!("wrote {}", "{} idatzita", path.display()));
        }
        None => print!("{}", text),
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
    let glyph_flags = width != 100 || names != "Q,C,L" || long_names || no_ranges || hex;
    if glyph_flags && to != ExportFormat::Glyph {
        return Err(t!(
            "--width, --names, --long-names, --no-ranges and --hex only apply to --to glyph",
            "--width, --names, --long-names, --no-ranges eta --hex --to glyph-ekin bakarrik dabiltza"
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
        ExportFormat::Ionq => qoa::ionq::from_program(&p, shots),
        ExportFormat::Glyph => {
            return write_or_print(output, &local(qoa::disasm::to_glyph(&p, glyph))?);
        }
        ExportFormat::Qoa => return write_or_print(output, &local(qoa::disasm::to_qoa(&p))?),
        ExportFormat::Listing => return write_or_print(output, &local(qoa::disasm::listing(&p))?),
        ExportFormat::Json => return write_or_print(output, &local(qoa::disasm::to_json(&p))?),
        ExportFormat::Dot => return write_or_print(output, &local(qoa::disasm::to_dot(&p))?),
    };
    match result {
        Ok(text) => write_or_print(output, &text),
        Err(errors) => {
            let target = match to {
                ExportFormat::Qasm => "OpenQASM 2.0",
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

fn cmd_import(file: &Path, output: Option<PathBuf>) -> Result<(), String> {
    let name = file.display().to_string();
    let src = std::fs::read_to_string(file)
        .map_err(|e| t!("cannot read {}: {}", "ezin da {} irakurri: {}", name, e))?;
    let is_glyph = file
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("qg"));
    let text = if is_glyph {
        qoa::glyph::expand(&src)
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
            println!("  {:<46} {}{}", s.usage(), s.summary(), marker);
        }
        println!();
    }
    let removed = isa::SPECS.iter().filter(|s| s.is_removed()).count();
    println!(
        "{}",
        t!(
            "{} instructions from QOA 0.3 were removed; `qoa isa NAME` says why. `qoa isa --markdown` prints all of this as docs.",
            "QOA 0.3-ko {} agindu kendu dira; `qoa isa IZENA`-k zergatik esaten du. `qoa isa --markdown`-ek hau guztia dokumentazio gisa inprimatzen du.",
            removed
        )
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_visual(
    input: &Path,
    output: &Path,
    resolution: &str,
    fps: u32,
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
    visualizer::render_video(&visualizer::Options {
        input,
        output,
        width,
        height,
        fps,
        direction,
        seed: seed.unwrap_or_else(rand::random::<u64>),
        ffmpeg_flags,
        ffmpeg_args,
    })
    .map_err(|e| format!("visual: {}", e))
}
