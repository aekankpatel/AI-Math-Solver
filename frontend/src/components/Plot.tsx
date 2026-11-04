import { useEffect, useRef, useState } from 'react'
import type { PlotlyFigure } from '../types'

type PlotlyModule = typeof import('plotly.js')

const THEME = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#cbd5e1', family: 'Inter, system-ui, sans-serif' },
  xaxis: { gridcolor: '#1e293b', zerolinecolor: '#475569' },
  yaxis: { gridcolor: '#1e293b', zerolinecolor: '#475569' },
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
    const merged = {
      ...layout,
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
        return Plotly.react(el, figure.data as Plotly.Data[], merged as Partial<Plotly.Layout>, {
          responsive: true,
          displaylogo: false,
          displayModeBar: narrow ? false : 'hover',
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
