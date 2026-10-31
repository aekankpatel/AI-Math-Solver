import katex from 'katex'
import 'katex/dist/katex.min.css'
import { useMemo } from 'react'

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

/** Plain text with inline `$...$` LaTeX segments (used for LLM explanations). */
export function RichText({ text }: { text: string }) {
  const parts = text.split(/(\$[^$\n]+\$)/g)
  return (
    <>
      {parts.map((part, i) =>
        part.startsWith('$') && part.endsWith('$') && part.length > 2 ? (
          <Tex key={i} math={part.slice(1, -1)} />
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  )
}
