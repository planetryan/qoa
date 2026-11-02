// fitxategi honek iturburuen zigilua egiaztatzen eta idazten du. zigilua fitxategiaren azken lerroa da (hutsik
// ez dagoen azkena):
//
//   SEAL sha256:<64 digitu hamaseitar>          .qoa eta .qg
//   // SEAL sha256:<64 digitu hamaseitar>       .qasm (iruzkin bat, beste tresnek ere irakur dezaten)
//
// eta haren aurreko byte guztien SHA-256-a da. zigilatutako fitxategi batean byte bakar bat aldatzen bada,
// QOA-k ez du mihiztatzen, eta zer gertatu den eta nola konpondu esaten du. `qoa seal FITXATEGIA`-k idazten
// du. zigilua aukerakoa da: zigilurik gabeko fitxategiak lehen bezala doaz.
//
// lerroak byteetan bilatzen dira (ascii hutsuneak: zuriunea, tabulazioa, \r, \n), mihiztatzaile bietan
// berdin (tools/asm.qoa-k gauza bera egiten du).

use crate::t;

pub const WORD: &[u8] = b"SEAL";
pub const ALGO: &[u8] = b"sha256:";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
    // .qoa, .qg
    Plain,
    // .qasm: `// SEAL ...`
    Comment,
}

impl Style {
    pub fn for_name(name: &str) -> Style {
        if name.to_ascii_lowercase().ends_with(".qasm") {
            Style::Comment
        } else {
            Style::Plain
        }
    }
}

fn blank(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

// zigilu-lerro bat aurkitu bada
#[derive(Debug)]
pub struct Found<'a> {
    // lerroaren hasiera (byteetan): zigiluak hau aurreko guztia estaltzen du
    pub start: usize,
    // lerro-zenbakia, 1etik
    pub line: u32,
    // zutabea (1etik) eta luzera, mezuetarako
    pub col: u32,
    pub len: u32,
    // `sha256:` ondorengoa, zuzena bada
    pub claimed: Option<&'a str>,
}

// azken lerro ez-hutsa zigilu bat bada
pub fn find(bytes: &[u8], style: Style) -> Option<Found<'_>> {
    let end = bytes.iter().rposition(|&b| !blank(b))? + 1;
    let start = bytes[..end]
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |i| i + 1);
    let lead = bytes[start..end]
        .iter()
        .take_while(|&&b| b == b' ' || b == b'\t')
        .count();
    let mut text = &bytes[start + lead..end];
    if style == Style::Comment {
        text = text.strip_prefix(b"//")?;
        text = &text[text
            .iter()
            .take_while(|&&b| b == b' ' || b == b'\t')
            .count()..];
    }
    let rest = text.strip_prefix(WORD)?;
    if !rest.is_empty() && rest[0] != b' ' && rest[0] != b'\t' {
        return None;
    }
    let rest = &rest[rest
        .iter()
        .take_while(|&&b| b == b' ' || b == b'\t')
        .count()..];
    let claimed = rest
        .strip_prefix(ALGO)
        .filter(|h| h.len() == 64 && h.iter().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')))
        .map(|h| std::str::from_utf8(h).expect("ascii"));
    let line = bytes[..start].iter().filter(|&&b| b == b'\n').count() as u32 + 1;
    Some(Found {
        start,
        line,
        col: lead as u32 + 1,
        // karaktereetan (iturburua utf-8 da; utf-8 jarraipen-byteak ez dira kontatzen)
        len: bytes[start + lead..end]
            .iter()
            .filter(|&&b| !(0x80..0xc0).contains(&b))
            .count() as u32,
        claimed,
    })
}

// zigiluaren egiaztapena: (lerroa, zutabea, luzera, mezua) huts egiten badu
#[derive(Debug)]
pub struct Broken {
    pub line: u32,
    pub col: u32,
    pub len: u32,
    pub message: String,
}

// iturburu bat zigilatuta badago, egiaztatu: Ok(Some(hasiera)) zigilua zuzena bada (iturburua hasiera arte
// mihiztatzen da), Ok(None) zigilurik ez badago
pub fn check(bytes: &[u8], style: Style, name: &str) -> Result<Option<usize>, Broken> {
    let Some(f) = find(bytes, style) else {
        return Ok(None);
    };
    let Some(claimed) = f.claimed else {
        return Err(Broken {
            line: f.line,
            col: f.col,
            len: f.len,
            message: t!(
                "a seal is SEAL sha256: followed by 64 hexadecimal digits in lower case; `qoa seal {}` writes it",
                "zigilu bat SEAL sha256: da eta gero 64 digitu hamaseitar letra txikiz; `qoa seal {}`-k idazten du",
                name
            ),
        });
    };
    let now = crate::sha256::hex(&crate::sha256::digest(&bytes[..f.start]));
    if now != claimed {
        return Err(Broken {
            line: f.line,
            col: f.col,
            len: f.len,
            message: t!(
                "{} was changed after it was sealed: the SHA-256 of the lines above the seal is {}, the seal says {}. If the change is yours, seal it again with `qoa seal {}`",
                "{} aldatu egin da zigilatu ondoren: zigiluaren gaineko lerroen SHA-256-a {} da, zigiluak {} dio. aldaketa zurea bada, zigilatu berriz `qoa seal {}`-rekin",
                name,
                now,
                claimed,
                name
            ),
        });
    }
    Ok(Some(f.start))
}

// iturburua zigilatuta: aurreko zigilua (badago) kenduta, lerro-jauzi batekin amaituta, eta zigilu berria
pub fn seal(bytes: &[u8], style: Style) -> Vec<u8> {
    let body = match find(bytes, style) {
        Some(f) => &bytes[..f.start],
        None => bytes,
    };
    let mut out = body.to_vec();
    if !out.is_empty() && out.last() != Some(&b'\n') {
        out.push(b'\n');
    }
    let sum = crate::sha256::hex(&crate::sha256::digest(&out));
    if style == Style::Comment {
        out.extend_from_slice(b"// ");
    }
    out.extend_from_slice(WORD);
    out.push(b' ');
    out.extend_from_slice(ALGO);
    out.extend_from_slice(sum.as_bytes());
    out.push(b'\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_then_check_then_tamper() {
        for (style, text) in [
            (Style::Plain, "H q0\nMEASURE q0\n"),
            (Style::Plain, "H q0"),
            (Style::Plain, ""),
            (Style::Comment, "OPENQASM 2.0;\nqreg q[1];\nh q[0];\n"),
        ] {
            let s = seal(text.as_bytes(), style);
            let start = check(&s, style, "x").unwrap().unwrap();
            let mut body = text.to_string();
            if !body.is_empty() && !body.ends_with('\n') {
                body.push('\n');
            }
            assert_eq!(&s[..start], body.as_bytes());
            // berriz zigilatzea: berdina
            assert_eq!(seal(&s, style), s);
            // byte bakoitza aldatuta: inoiz ez zigilatu eta zuzen gisa (zigiluaren aurreko lerro-jauzia
            // hondatzen bada, fitxategia zigilurik gabea dirudi: QOA_REQUIRE_SEAL-ek hori ere geldiarazten du)
            for i in 0..s.len() {
                for bit in [0x01u8, 0x20, 0x80] {
                    let mut bad = s.clone();
                    bad[i] ^= bit;
                    assert!(
                        !matches!(check(&bad, style, "x"), Ok(Some(_))),
                        "{:?} byte {} bit {:x}",
                        text,
                        i,
                        bit
                    );
                }
            }
            // amaierako lerro hutsak ez dira kontatzen
            let mut more = s.clone();
            more.extend_from_slice(b"\n\n  \n");
            assert!(check(&more, style, "x").unwrap().is_some());
        }
    }

    #[test]
    fn unsealed_and_malformed() {
        assert!(check(b"H q0\n", Style::Plain, "x").unwrap().is_none());
        assert!(
            check(b"H q0\nSEALED\n", Style::Plain, "x")
                .unwrap()
                .is_none()
        );
        assert!(check(b"SEAL sha256:00\n", Style::Plain, "x").is_err());
        let e = check(b"H q0\n  SEAL md5:00\n", Style::Plain, "x")
            .err()
            .unwrap();
        assert_eq!((e.line, e.col, e.len), (2, 3, 11));
        // .qasm-n iruzkin gisa bakarrik
        let s = seal(b"OPENQASM 2.0;\n", Style::Plain);
        assert!(check(&s, Style::Comment, "x").unwrap().is_none());
    }
}
