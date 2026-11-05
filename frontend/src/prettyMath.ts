// Plain-text maths as people write it on paper: x^2 -> x², 3*x -> 3x, sqrt( -> √(, <= -> ≤, -> -> →.
// Used for everything printed on the solution sheets (never for what the user types in the box).

const SUP: Record<string, string> = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
  '+': '⁺', '-': '⁻', '−': '⁻', '(': '⁽', ')': '⁾', n: 'ⁿ', i: 'ⁱ', x: 'ˣ', y: 'ʸ', k: 'ᵏ', t: 'ᵗ', a: 'ᵃ', b: 'ᵇ',
  m: 'ᵐ', '.': '·',
}

function superscript(s: string): string | null {
  const out = [...s].map((c) => SUP[c])
  return out.every(Boolean) ? out.join('') : null
}

export function prettyMath(text: string): string {
  if (!text) return text
  let s = text
  s = s.replace(/\*\*/g, '^')
  // exponents: x^2, x^-1, x^(2n), e^(x+1) when every character has a superscript form
  for (let i = 0; i < 3; i++) {
    s = s.replace(/\^\(([^()]{1,12})\)/g, (m, inner: string) => superscript(inner.replace(/\s+/g, '')) ?? m)
    s = s.replace(/\^(-?[0-9]+(?:\.[0-9]+)?|[nkixyt])(?![0-9A-Za-z(])/g, (m, e: string) => superscript(e) ?? m)
  }
  s = s
    .replace(/\bsqrt\((\d+)\)/g, '√$1')
    .replace(/\bsqrt\(/g, '√(')
    .replace(/\bpi\b/g, 'π')
    .replace(/(\d)\s*\*\s*(?=(?:sin|cos|tan|exp|log|ln|sec|csc|cot|asin|acos|atan)\b)/g, '$1 ') // 3*sin -> 3 sin
    .replace(/(\d)\s*\*\s*(?=[A-Za-z(√π])/g, '$1') // 3*x -> 3x
    .replace(/\s*\*\s*/g, '·') // x*cos(x) -> x·cos(x)
    .replace(/(^|[^\w])(-?)oo\b/g, (_m, pre: string, sign: string) => `${pre}${sign ? '−' : ''}∞`)
    .replace(/<=/g, '≤')
    .replace(/>=/g, '≥')
    .replace(/!=/g, '≠')
    .replace(/->/g, '→')
    .replace(/ - /g, ' − ')
    .replace(/\(-/g, '(−')
    .replace(/= -/g, '= −')
  return s
}
