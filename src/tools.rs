// qoa binarioan sartutako tresnak: QOAn idatzitako programak. pakete orok darama, beraz `qoa compile
// --self-hosted`-ek QOA-ren mihiztatzailea QOAn exekuta dezake iturburu-zuhaitzik gabe, eta `qoa tools --write`-k
// tresnen iturburuak idazten ditu, irakurtzeko edo aldatzeko. tresna bakoitza bytecode gisa dago binarioan
// (src/qoa/*.qx), QOAn idatzitako mihiztatzaileak eraikia: exekutatzean rust-en mihiztatzaileak ez du parterik.
// probek egiaztatzen dute .qx-a iturburutik datorrela (bi mihiztatzaileekin, byte berberak)

use crate::program::Program;

// tresna bat: izena, fitxategia (src/qoa/ karpetan), iturburua, bytecode-a eta deskribapena bi hizkuntzetan
pub struct Tool {
    pub name: &'static str,
    pub file: &'static str,
    pub source: &'static str,
    pub binary: &'static [u8],
    pub about_en: &'static str,
    pub about_eu: &'static str,
}

pub const TOOLS: &[Tool] = &[
    Tool {
        name: "asm",
        file: "asm.qoa",
        source: include_str!("qoa/asm.qoa"),
        binary: include_bytes!("qoa/asm.qx"),
        about_en: "QOA's assembler written in QOA: .qoa source in, .qx binary out, byte for byte what qoa compile writes, with the same errors",
        about_eu: "QOAn idatzitako QOA-ren mihiztatzailea: .qoa iturburua sartu, .qx bitarra atera, qoa compile-k idazten dituen byte berberak, errore berberekin",
    },
    Tool {
        name: "dis",
        file: "dis.qoa",
        source: include_str!("qoa/dis.qoa"),
        binary: include_bytes!("qoa/dis.qx"),
        about_en: "QOA's disassembler written in QOA: .qx binary in, plain QOA out, byte for byte what qoa export --to qoa --long-names writes, with the same errors for damaged files",
        about_eu: "QOAn idatzitako QOA-ren desmihiztatzailea: .qx bitarra sartu, QOA arrunta atera, qoa export --to qoa --long-names-ek idazten dituen byte berberak, fitxategi hondatuentzat errore berberekin",
    },
];

pub fn find(name: &str) -> Option<&'static Tool> {
    TOOLS.iter().find(|t| t.name == name)
}

impl Tool {
    // fitxategiaren bidea iturburu-zuhaitzean (mezuetan eta programaren izenean)
    pub fn path(&self) -> String {
        format!("src/qoa/{}", self.file)
    }

    // bytecode-aren bidea (src/qoa/asm.qx)
    pub fn binary_path(&self) -> String {
        format!("src/qoa/{}", self.binary_file())
    }

    pub fn binary_file(&self) -> String {
        match self.file.strip_suffix(".qoa") {
            Some(stem) => format!("{}.qx", stem),
            None => format!("{}.qx", self.file),
        }
    }

    // exekutatzen den tresna: binarioan sartutako bytecode-a, laburpena egiaztatuta. haren lerro-zenbakiak
    // iturburuarenak dira, beraz mezuek src/qoa/asm.qoa-ren lerroak aipatzen dituzte
    pub fn program(&self) -> Result<Program, String> {
        Program::from_bytes(self.binary).map_err(|e| format!("{}: {}", self.binary_path(), e))
    }

    // tresna iturburutik, rust-en mihiztatzailearekin (probek bytecode-arekin konparatzeko)
    pub fn assemble(&self) -> Result<Program, String> {
        let name = self.path();
        crate::asm::assemble(self.source, &name)
            .map_err(|d| crate::asm::render(&d, self.source, &name))
    }

    // glifo-bertsioa (src/qoa/asm.qg), iturburutik sortua lerroz lerro (`--keep-source`): ez dago binarioan bi aldiz
    pub fn glyph_source(&self) -> Result<String, String> {
        crate::convert::qoa_to_glyph(self.source, &self.path())
    }

    pub fn glyph_file(&self) -> String {
        match self.file.strip_suffix(".qoa") {
            Some(stem) => format!("{}.qg", stem),
            None => format!("{}.qg", self.file),
        }
    }
}
