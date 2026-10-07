// fitxategi honek irteeraren hizkuntza aukeratzen du: euskara edo ingelesa.
// ordena: QOA_LANG=eu|en lehenik; bestela sistemaren LC_ALL, LC_MESSAGES edo LANG ("eu..." bada
// euskara); bestela ingelesa. mezu bakoitza bi hizkuntzetan dago bere erabilera-lekuan, `t!` eta
// `ts!` makroekin, bi bertsioak elkarrengandik ez urruntzeko.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    En,
    Eu,
}

static LANG: OnceLock<Lang> = OnceLock::new();

pub fn lang() -> Lang {
    *LANG.get_or_init(detect)
}

pub fn is_eu() -> bool {
    lang() == Lang::Eu
}

// hizkuntza behartu (lehen mezua inprimatu aurretik bakarrik du eragina)
pub fn set(l: Lang) {
    let _ = LANG.set(l);
}

fn detect() -> Lang {
    if let Ok(v) = std::env::var("QOA_LANG") {
        let v = v.to_ascii_lowercase();
        if v.starts_with("eu") {
            return Lang::Eu;
        }
        if v.starts_with("en") {
            return Lang::En;
        }
    }
    for key in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Ok(v) = std::env::var(key) {
            if v.is_empty() {
                continue;
            }
            return if v.to_ascii_lowercase().starts_with("eu") {
                Lang::Eu
            } else {
                Lang::En
            };
        }
    }
    Lang::En
}

// t!("ingelesa {}", "euskara {}", balioak...) -> String
#[macro_export]
macro_rules! t {
    ($en:literal, $eu:literal $(,)?) => {
        if $crate::lang::is_eu() { String::from($eu) } else { String::from($en) }
    };
    ($en:literal, $eu:literal, $($arg:tt)*) => {
        if $crate::lang::is_eu() { format!($eu, $($arg)*) } else { format!($en, $($arg)*) }
    };
}

// ts!("ingelesa", "euskara") -> &'static str (formaturik gabeko testuetarako)
#[macro_export]
macro_rules! ts {
    ($en:literal, $eu:literal $(,)?) => {
        if $crate::lang::is_eu() { $eu } else { $en }
    };
}
