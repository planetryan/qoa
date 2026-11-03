// qoa binarioan sartutako tresnak: QOAn idatzitako programak. pakete orok darama, beraz `qoa compile
// --self-hosted`-ek QOA-ren mihiztatzailea QOAn exekuta dezake iturburu-zuhaitzik gabe, eta `qoa tools --write`-k
// tresnen iturburuak idazten ditu, irakurtzeko edo aldatzeko

use crate::program::Program;

// tresna bat: izena, fitxategia (src/qoa/ karpetan), iturburua eta deskribapena bi hizkuntzetan
pub struct Tool {
    pub name: &'static str,
    pub file: &'static str,
    pub source: &'static str,
    pub about_en: &'static str,
    pub about_eu: &'static str,
}

pub const TOOLS: &[Tool] = &[Tool {
    name: "asm",
    file: "asm.qoa",
    source: include_str!("qoa/asm.qoa"),
    about_en: "QOA's assembler written in QOA: .qoa source in, .qx binary out, byte for byte what qoa compile writes, with the same errors",
    about_eu: "QOAn idatzitako QOA-ren mihiztatzailea: .qoa iturburua sartu, .qx bitarra atera, qoa compile-k idazten dituen byte berberak, errore berberekin",
}];

pub fn find(name: &str) -> Option<&'static Tool> {
    TOOLS.iter().find(|t| t.name == name)
}

impl Tool {
    // fitxategiaren bidea iturburu-zuhaitzean (mezuetan eta programaren izenean)
    pub fn path(&self) -> String {
        format!("src/qoa/{}", self.file)
    }

    // tresna mihiztatuta, rust-en mihiztatzailearekin
    pub fn program(&self) -> Result<Program, String> {
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
