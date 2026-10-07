// fitxategi honek eragigaien zenbakizko adierazpenak ebaluatzen ditu: `pi/2`, `-3*pi/4`, `(1+2)/sqrt(2)`.
//
// gramatika (ohiko lehentasuna; `^` lotuena da eta eskuinetik elkartzen da):
//   expr   = term (('+' | '-') term)*
//   term   = unary (('*' | '/') unary)*
//   unary  = ('-' | '+') unary | power
//   power  = atom ('^' unary)?
//   atom   = zenbakia | konstantea | aldagaia | funtzioa '(' expr ')' | '(' expr ')'
// konstanteak: pi, tau, e (eta π). funtzioak: sqrt, sin, cos, tan, exp, ln.

use crate::t;
use std::collections::HashMap;

pub fn eval(src: &str) -> Result<f64, String> {
    eval_with(src, &HashMap::new())
}

// aldagaiekin ebaluatu (openqasm `gate` definizioen parametroetarako)
pub fn eval_with(src: &str, vars: &HashMap<String, f64>) -> Result<f64, String> {
    let mut p = Parser {
        chars: src.chars().collect(),
        pos: 0,
        vars,
    };
    let v = p.expr()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(t!(
            "unexpected '{}' in expression '{}'",
            "ustekabeko '{}' '{}' adierazpenean",
            p.chars[p.pos],
            src
        ));
    }
    if !v.is_finite() {
        return Err(t!(
            "expression '{}' is not a finite number",
            "'{}' adierazpena ez da zenbaki finitu bat",
            src
        ));
    }
    Ok(v)
}

struct Parser<'v> {
    chars: Vec<char>,
    pos: usize,
    vars: &'v HashMap<String, f64>,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        while let Some(c) = self.peek() {
            match c {
                '+' => {
                    self.pos += 1;
                    v += self.term()?;
                }
                '-' => {
                    self.pos += 1;
                    v -= self.term()?;
                }
                _ => break,
            }
        }
        Ok(v)
    }

    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.unary()?;
        while let Some(c) = self.peek() {
            match c {
                '*' => {
                    self.pos += 1;
                    v *= self.unary()?;
                }
                '/' => {
                    self.pos += 1;
                    let d = self.unary()?;
                    if d == 0.0 {
                        return Err(t!(
                            "division by zero in expression",
                            "zeroz zatitzea adierazpenean"
                        ));
                    }
                    v /= d;
                }
                _ => break,
            }
        }
        Ok(v)
    }

    fn unary(&mut self) -> Result<f64, String> {
        match self.peek() {
            Some('-') => {
                self.pos += 1;
                Ok(-self.unary()?)
            }
            Some('+') => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<f64, String> {
        let base = self.atom()?;
        if self.peek() == Some('^') {
            self.pos += 1;
            let exp = self.unary()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<f64, String> {
        match self.peek() {
            None => Err(t!(
                "expression ended early",
                "adierazpena goizegi amaitu da"
            )),
            Some('(') => {
                self.pos += 1;
                let v = self.expr()?;
                if self.peek() != Some(')') {
                    return Err(t!(
                        "missing ')' in expression",
                        "')' falta da adierazpenean"
                    ));
                }
                self.pos += 1;
                Ok(v)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => self.number(),
            Some(c) if c.is_alphabetic() || c == '_' => {
                let start = self.pos;
                while self.pos < self.chars.len()
                    && (self.chars[self.pos].is_alphanumeric() || self.chars[self.pos] == '_')
                {
                    self.pos += 1;
                }
                let name: String = self.chars[start..self.pos].iter().collect();
                if let Some(v) = self.vars.get(&name) {
                    return Ok(*v);
                }
                match name.to_ascii_lowercase().as_str() {
                    "pi" | "π" => Ok(std::f64::consts::PI),
                    "tau" => Ok(std::f64::consts::TAU),
                    "e" => Ok(std::f64::consts::E),
                    f @ ("sqrt" | "sin" | "cos" | "tan" | "exp" | "ln") => {
                        if self.peek() != Some('(') {
                            return Err(t!(
                                "{0} needs parentheses, e.g. {0}(2)",
                                "{0}-k parentesiak behar ditu, adib. {0}(2)",
                                f
                            ));
                        }
                        let arg = self.atom()?;
                        Ok(match f {
                            "sqrt" => arg.sqrt(),
                            "sin" => arg.sin(),
                            "cos" => arg.cos(),
                            "tan" => arg.tan(),
                            "exp" => arg.exp(),
                            _ => arg.ln(),
                        })
                    }
                    _ => Err(t!(
                        "unknown name '{}' in expression",
                        "'{}' izen ezezaguna adierazpenean",
                        name
                    )),
                }
            }
            Some(c) => Err(t!(
                "unexpected '{}' in expression",
                "ustekabeko '{}' adierazpenean",
                c
            )),
        }
    }

    fn number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        while self.pos < self.chars.len()
            && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.')
        {
            self.pos += 1;
        }
        // idazkera zientifikoa: 1e-3, 2.5E+4
        if self.pos < self.chars.len()
            && (self.chars[self.pos] == 'e' || self.chars[self.pos] == 'E')
        {
            let save = self.pos;
            self.pos += 1;
            if self.pos < self.chars.len()
                && (self.chars[self.pos] == '+' || self.chars[self.pos] == '-')
            {
                self.pos += 1;
            }
            if self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            } else {
                // zenbakiaren ondorengo e konstantea zen ("2e"), eta hori ez dugu inplizituki onartzen
                self.pos = save;
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        text.parse::<f64>()
            .map_err(|_| t!("'{}' is not a number", "'{}' ez da zenbaki bat", text))
    }
}

#[cfg(test)]
mod tests {
    use super::eval;
    use std::f64::consts::PI;

    #[test]
    fn evaluates_common_angle_forms() {
        assert_eq!(eval("1.5").unwrap(), 1.5);
        assert_eq!(eval("pi/2").unwrap(), PI / 2.0);
        assert_eq!(eval("-pi/4").unwrap(), -PI / 4.0);
        assert_eq!(eval("2*pi/3").unwrap(), 2.0 * PI / 3.0);
        assert_eq!(eval("(1 + 2) * 3").unwrap(), 9.0);
        assert_eq!(eval("2^3^2").unwrap(), 512.0);
        assert_eq!(eval("1e-3").unwrap(), 0.001);
        assert!((eval("1/sqrt(2)").unwrap() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn rejects_garbage() {
        assert!(eval("pi pi").is_err());
        assert!(eval("2*").is_err());
        assert!(eval("foo").is_err());
        assert!(eval("1/0").is_err());
    }
}
