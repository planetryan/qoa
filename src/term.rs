// fitxategi honek programek eta fitxategiek ekarritako testua terminal batean seguru erakusten du. terminalek
// kontrol-sekuentziak exekutatzen dituzte: izenburua aldatu, arbelean idatzi (OSC 52), kurtsorea mugitu eta
// pantaila gainidatzi, edo terminalak berak erantzun bat "idatz" dezan eskatu (DSR), erabiltzaileak idatzi
// balu bezala. programa edo .qx fitxategi gaizto batek horiek inprima litzake; hemen ikusgai bihurtzen dira
// (ESC -> ^[), eta terminalera bakarrik: hodi edo fitxategi batera byteak bere horretan doaz, programaren
// irteera aldatu gabe.
//
// gordetzen da: lerro-jauzia, tabulazioa, orga-itzulera eta koloreak (SGR: ESC [ zenbakiak m), kalterik
// gabeak. beste guztia ikusgai: C0 kontrolak eta DEL ^X notazioan, C1 kontrolak (U+0080..U+009F, CSI
// 8 bitekoa barne) \u{..} gisa, eta utf-8 baliogabeko byteak \x.. gisa.

use std::borrow::Cow;
use std::io::{IsTerminal, Write};

// SGR sekuentzia baten luzera handiena (hortik aurrera ez da kolore bat)
const MAX_SGR: usize = 64;

// bytes[i..]-ren hasiera: (zenbat byte kontsumitu, zer idatzi), edo None sekuentzia osatu gabe badago eta
// datu gehiago etor daitezke
fn step(bytes: &[u8], i: usize, done: bool, out: &mut Vec<u8>) -> Option<usize> {
    let b = bytes[i];
    match b {
        b'\n' | b'\t' | b'\r' => {
            out.push(b);
            Some(1)
        }
        0x1b => {
            // ESC [ (0-9 ; :)* m -> kolorea, bere horretan
            let rest = &bytes[i + 1..];
            match rest.first() {
                None if !done => return None,
                Some(b'[') => {
                    let params = rest[1..]
                        .iter()
                        .take_while(|c| c.is_ascii_digit() || **c == b';' || **c == b':')
                        .count();
                    match rest.get(1 + params) {
                        Some(b'm') if params <= MAX_SGR => {
                            out.extend_from_slice(&bytes[i..i + 2 + params + 1]);
                            return Some(2 + params + 1);
                        }
                        None if !done && params <= MAX_SGR => return None,
                        _ => {}
                    }
                }
                _ => {}
            }
            out.extend_from_slice(b"^[");
            Some(1)
        }
        0x00..=0x1f => {
            out.extend_from_slice(&[b'^', b + 0x40]);
            Some(1)
        }
        0x7f => {
            out.extend_from_slice(b"^?");
            Some(1)
        }
        0x20..=0x7e => {
            out.push(b);
            Some(1)
        }
        _ => {
            let need = match b {
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => 0,
            };
            if need > 0 && bytes.len() - i < need && !done {
                // utf-8 karaktere baten zatia: gainerakoa hurrengo idazketan dator
                if std::str::from_utf8(&bytes[i..]).is_err_and(|e| e.error_len().is_none()) {
                    return None;
                }
            }
            if need > 0
                && let Some(chunk) = bytes.get(i..i + need)
                && let Ok(s) = std::str::from_utf8(chunk)
            {
                let c = s.chars().next().expect("karaktere bat");
                if ('\u{80}'..='\u{9f}').contains(&c) {
                    out.extend_from_slice(format!("\\u{{{:x}}}", c as u32).as_bytes());
                } else {
                    out.extend_from_slice(chunk);
                }
                return Some(need);
            }
            out.extend_from_slice(format!("\\x{:02x}", b).as_bytes());
            Some(1)
        }
    }
}

// byte-segida bat ikusgai: (idazteko byteak, amaieran gelditu den zati osatugabea)
fn convert(bytes: &[u8], done: bool) -> (Vec<u8>, usize) {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match step(bytes, i, done, &mut out) {
            Some(n) => i += n,
            None => break,
        }
    }
    (out, i)
}

// testu bat terminalean erakusteko moduan; aldatu behar ez bada, kopiarik gabe
pub fn visible(s: &str) -> Cow<'_, str> {
    let plain = s
        .bytes()
        .all(|b| (0x20..0x7f).contains(&b) || matches!(b, b'\n' | b'\t' | b'\r'));
    if plain {
        return Cow::Borrowed(s);
    }
    let (out, _) = convert(s.as_bytes(), true);
    Cow::Owned(String::from_utf8(out).expect("irteera utf-8 baliozkoa da"))
}

// `visible`, irteera terminal bat bada bakarrik (bestela testua bere horretan)
pub fn for_stdout(s: &str) -> Cow<'_, str> {
    if std::io::stdout().is_terminal() {
        visible(s)
    } else {
        Cow::Borrowed(s)
    }
}

pub fn for_stderr(s: &str) -> Cow<'_, str> {
    if std::io::stderr().is_terminal() {
        visible(s)
    } else {
        Cow::Borrowed(s)
    }
}

// idazle bat, terminal baterako bada byteak ikusgai bihurtzen dituena (utf-8 karaktereak eta koloreak
// idazketa batetik bestera zatituta etor daitezke)
pub struct Screen<W: Write> {
    inner: W,
    terminal: bool,
    pending: Vec<u8>,
}

impl<W: Write> Screen<W> {
    pub fn new(inner: W, terminal: bool) -> Screen<W> {
        Screen {
            inner,
            terminal,
            pending: Vec::new(),
        }
    }

    fn drain(&mut self, done: bool) -> std::io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let (out, used) = convert(&self.pending, done);
        self.pending.drain(..used);
        self.inner.write_all(&out)
    }
}

impl<W: Write> Write for Screen<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if !self.terminal {
            return self.inner.write(buf);
        }
        self.pending.extend_from_slice(buf);
        self.drain(false)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // zati osatugabe bat (adib. ESC bakarrik) flush-ean ere zain geratzen da: hurrengo idazketak osa dezake
        self.inner.flush()
    }
}

impl<W: Write> Drop for Screen<W> {
    fn drop(&mut self) {
        let _ = self.drain(true);
        let _ = self.inner.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen(parts: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut s = Screen::new(&mut out, true);
            for p in parts {
                s.write_all(p).unwrap();
            }
        }
        out
    }

    #[test]
    fn dangerous_sequences_become_visible() {
        // izenburua, arbela (OSC 52), kurtsorea, egoera-txostena (DSR), 8 biteko CSI
        assert_eq!(visible("a\x1b]0;title\x07b"), "a^[]0;title^Gb");
        assert_eq!(visible("\x1b]52;c;cm0gLXJm\x07"), "^[]52;c;cm0gLXJm^G");
        assert_eq!(visible("\x1b[2J\x1b[6n"), "^[[2J^[[6n");
        assert_eq!(visible("x\u{9b}31m"), "x\\u{9b}31m");
        assert_eq!(visible("\0\x08\x7f"), "^@^H^?");
    }

    #[test]
    fn text_colours_and_line_ends_stay() {
        assert_eq!(visible("kaixo\tmundua\r\n"), "kaixo\tmundua\r\n");
        assert_eq!(visible("\x1b[1;31mred\x1b[0m"), "\x1b[1;31mred\x1b[0m");
        assert_eq!(visible("π é 你好 🎵"), "π é 你好 🎵");
    }

    #[test]
    fn split_writes_and_bad_bytes() {
        // utf-8 karakterea eta kolorea bi idazketatan
        assert_eq!(
            screen(&["é".as_bytes()[..1].as_ref(), "é".as_bytes()[1..].as_ref()]),
            "é".as_bytes()
        );
        assert_eq!(screen(&[b"\x1b[3", b"1mx"]), b"\x1b[31mx");
        // utf-8 baliogabea eta amaierako ESC bakarra
        assert_eq!(screen(&[b"\xff\xfe"]), b"\\xff\\xfe");
        assert_eq!(screen(&[b"a\x1b"]), b"a^[");
        // terminal ez bada, bere horretan
        let mut out = Vec::new();
        Screen::new(&mut out, false)
            .write_all(b"\x1b]0;x\x07")
            .unwrap();
        assert_eq!(out, b"\x1b]0;x\x07");
    }
}
