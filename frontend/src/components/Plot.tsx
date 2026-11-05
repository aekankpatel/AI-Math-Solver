import { useEffect, useRef, useState } from 'react'
import { prettyMath } from '../prettyMath'
import type { PlotlyFigure } from '../types'

type PlotlyModule = typeof import('plotly.js')

const THEME = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#3b3a36', family: '"IBM Plex Mono", ui-monospace, monospace', size: 11 },
  xaxis: { gridcolor: '#e6dfcf', zerolinecolor: '#a69f8f', linecolor: '#a69f8f' },
  yaxis: { gridcolor: '#e6dfcf', zerolinecolor: '#a69f8f', linecolor: '#a69f8f' },
}

// The API draws in a neutral palette; recolour it to the page's inks.
const INKS: Record<string, string> = {
  '#6366f1': '#1d3557',
  '#f59e0b': '#b5332a',
  '#10b981': '#2e6b3f',
  'rgba(99,102,241': 'rgba(29,53,87',
  'rgba(245,158,11': 'rgba(181,51,42',
  'rgba(16,185,129': 'rgba(46,107,63',
}
function recolour<T>(value: T): T {
  if (typeof value === 'string') {
    const hit = Object.keys(INKS).find((k) => value.toLowerCase().startsWith(k))
    return (hit ? INKS[hit] + value.slice(hit.length) : value) as T
  }
  if (Array.isArray(value)) return value.map(recolour) as T
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, (k === 'name' || k === 'text') && typeof v === 'string' ? prettyMath(v) : recolour(v)]),
    ) as T
  return value
}

// plotly.js is large: load it lazily, once, and share the module.
let loaded: PlotlyModule | null = null
let loading: Promise<PlotlyModule> | null = null
function loadPlotly(): Promise<PlotlyModule> {
  loading ??= import('plotly.js-cartesian-dist-min').then((mod) => {
    loaded = ((mod as unknown as { default?: PlotlyModule }).default ?? mod) as PlotlyModule
    return loaded
  })
  return loading
}

/** Interactive Plotly chart rendered from a backend-generated figure spec. */
export function Plot({ figure }: { figure: PlotlyFigure }) {
  const ref = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const el = ref.current
    if (!el) return
    const layout = figure.layout as Record<string, Record<string, unknown>>
    const narrow = el.clientWidth < 480
    const title = layout.title as { text?: string } | undefined
    const merged = {
      ...layout,
      ...(title?.text ? { title: { ...title, text: prettyMath(title.text) } } : {}),
      ...THEME,
      // on phones: smaller title placed left of the toolbar, which only appears on tap
      ...(narrow
        ? {
            title: { ...(layout.title ?? {}), font: { size: 13 }, x: 0.02, xanchor: 'left' },
            margin: { l: 42, r: 10, t: 46, b: 40 },
          }
        : {}),
      xaxis: { ...(layout.xaxis ?? {}), ...THEME.xaxis },
      yaxis: { ...(layout.yaxis ?? {}), ...THEME.yaxis },
      autosize: true,
    }
    loadPlotly()
      .then((Plotly) => {
        if (!active) return
        return Plotly.react(el, recolour(figure.data) as Plotly.Data[], merged as Partial<Plotly.Layout>, {
          responsive: true,
          displaylogo: false,
          displayModeBar: false, // no toolbar: drag to zoom, double-click to reset
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        })
      })
      .catch(() => active && setError('Could not render the visualization.'))
    return () => {
      active = false
      loaded?.purge(el)
    }
  }, [figure])

  if (error) return <p className="muted">{error}</p>
  return <div ref={ref} className="plot" data-testid="plot" />
}
