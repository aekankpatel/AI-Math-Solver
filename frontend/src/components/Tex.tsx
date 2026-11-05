import katex from 'katex'
import 'katex/dist/katex.min.css'
import { useMemo } from 'react'
import { prettyMath } from '../prettyMath'

interface TexProps {
  math: string
  display?: boolean
}

/** Render LaTeX with KaTeX. Untrusted input is safe: KaTeX outputs escaped markup
 *  and `trust: false` disables \href, \url and similar commands. */
export function Tex({ math, display = false }: TexProps) {
  const html = useMemo(
    () =>
      katex.renderToString(math, {
        displayMode: display,
        throwOnError: false,
        trust: false,
        strict: 'ignore',
        maxSize: 20,
        maxExpand: 500,
      }),
    [math, display],
  )
  return <span className={display ? 'tex tex-display' : 'tex'} dangerouslySetInnerHTML={{ __html: html }} />
}

// $$…$$ and \[…\] (display), \(…\) and $…$ (inline), **bold**
const SEGMENT = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$|\*\*[^*\n]+?\*\*)/g

/** Plain text mixed with LaTeX (LLM explanations and follow-up answers). Any of the common
 *  delimiters is rendered; everything else stays plain, escaped text. */
export function RichText({ text }: { text: string }) {
  const parts = text.split(SEGMENT)
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('$$') && part.endsWith('$$') && part.length > 4)
          return <Tex key={i} math={part.slice(2, -2)} display />
        if (part.startsWith('\\[') && part.endsWith('\\]')) return <Tex key={i} math={part.slice(2, -2)} display />
        if (part.startsWith('\\(') && part.endsWith('\\)')) return <Tex key={i} math={part.slice(2, -2)} />
        if (part.startsWith('$') && part.endsWith('$') && part.length > 2) return <Tex key={i} math={part.slice(1, -1)} />
        if (part.startsWith('**') && part.endsWith('**') && part.length > 4)
          return <strong key={i}>{prettyMath(part.slice(2, -2))}</strong>
        return <span key={i}>{prettyMath(part)}</span>
      })}
    </>
  )
}
