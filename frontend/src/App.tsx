import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import { api, ApiError, MAX_PROBLEM_CHARS } from './api'
import { StatusLegend } from './components/IntentPanel'
import { RichText } from './components/Tex'
import { EXAMPLE_LABELS } from './examples'
import { HealthPill } from './components/PlannerInfo'
import { SolutionCard } from './components/SolutionCard'
import type { Health, SolveResponse } from './types'

interface Turn {
  id: number
  result: SolveResponse
  followUp: boolean
}

const FALLBACK_EXAMPLES: Record<string, string[]> = {
  Algebra: ['Solve x^2 - 5x + 6 = 0'],
  Calculus: ['Find the derivative of x^3 sin(x)'],
  Integrals: ['Integrate x^2 from 0 to 3'],
  'Linear algebra': ['eigenvalues of [[2,1],[1,2]]'],
  'Probability & statistics': ['Binomial n=10 p=0.5, P(X <= 3)'],
}

const LINE = 32 // the notebook's line spacing (see --line in index.css)

/** Rounds its content's height up to whole notebook lines, so whatever follows stays on the rules. */
function OnTheLines({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const box = ref.current
    const inner = box?.firstElementChild as HTMLElement | null
    if (!box || !inner) return
    const fit = () => {
      const h = inner.getBoundingClientRect().height
      box.style.height = `${Math.ceil((h + 6) / LINE) * LINE}px` // + room for the sheet's shadow
    }
    fit()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(fit)
    observer.observe(inner)
    return () => observer.disconnect()
  }, [])
  return (
    <div ref={ref} className="on-lines">
      {children}
    </div>
  )
}

export default function App() {
  const [problem, setProblem] = useState('')
  const [followUp, setFollowUp] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [reachable, setReachable] = useState(true)
  const [examples, setExamples] = useState<Record<string, string[]>>(FALLBACK_EXAMPLES)
  const nextId = useRef(1)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // The free backend host sleeps when idle and needs up to ~2–3 minutes to wake: keep retrying.
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const check = (attempt: number) => {
      api
        .health()
        .then((h) => {
          if (cancelled) return
          setHealth(h)
          setReachable(true)
        })
        .catch(() => {
          if (cancelled) return
          setHealth(null)
          if (attempt < 36) {
            setReachable(true) // shown as "Connecting…" while retrying
            timer = setTimeout(() => check(attempt + 1), 5000)
          } else {
            setReachable(false)
          }
        })
    }
    check(0)
    api.examples().then(setExamples).catch(() => undefined)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [turns])

  async function run(action: () => Promise<SolveResponse>, isFollowUp: boolean) {
    setLoading(true)
    setError(null)
    try {
      const result = await action()
      setReachable(true)
      setSessionId(result.session_id)
      if (result.llm_status === 'budget' || result.llm_status === 'cooldown') {
        const state = result.llm_status
        setHealth((h) => (h ? { ...h, llm_available: false, llm_state: state } : h))
      } else if (result.llm_status === 'used') {
        setHealth((h) => (h ? { ...h, llm_available: true, llm_state: 'available' } : h))
      }
      setTurns((t) => [...t, { id: nextId.current++, result, followUp: isFollowUp }])
      return true
    } catch (e) {
      if (e instanceof ApiError && e.status === 0) setReachable(false)
      setError(e instanceof ApiError ? e.message : 'Unexpected error.')
      return false
    } finally {
      setLoading(false)
    }
  }

  async function submitProblem(e?: FormEvent, text = problem) {
    e?.preventDefault()
    const p = text.trim()
    if (!p || loading) return
    setTurns([])
    setSessionId(null)
    setProblem(p) // the box keeps showing the question that was asked
    await run(() => api.solve(p), false)
  }

  async function submitFollowUp(e: FormEvent) {
    e.preventDefault()
    const q = followUp.trim()
    if (!q || !sessionId || loading) return
    if (await run(() => api.followUp(sessionId, q), true)) setFollowUp('')
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void submitProblem()
    }
  }

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1 className="wordmark">Mathenza</h1>
          <p className="tagline">A calculator that shows its working, and then checks it.</p>
        </div>
        <HealthPill health={health} reachable={reachable} />
      </header>

      <main>
        <form className="ask" onSubmit={submitProblem}>
          <label htmlFor="problem" className="sr-only">
            Math problem
          </label>
          <span className="eyebrow" aria-hidden>
            Problem
          </span>
          <div className="ask-row">
            <textarea
              id="problem"
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="integrate x·sin(x) from 0 to pi"
              rows={2}
              maxLength={MAX_PROBLEM_CHARS}
              aria-describedby="problem-hint"
              spellCheck={false}
            />
            <button type="submit" disabled={loading || !problem.trim()}>
              {loading && !sessionId ? 'Solving…' : 'Solve'}
            </button>
          </div>
          <div className="ask-hint" id="problem-hint">
            <span>Plain text or LaTeX. Enter solves, Shift+Enter adds a line.</span>
            {problem.length > MAX_PROBLEM_CHARS * 0.8 && (
              <span>
                {problem.length}/{MAX_PROBLEM_CHARS}
              </span>
            )}
          </div>
        </form>

        {turns.length === 0 && !loading && (
          <section className="examples" aria-label="Examples">
            <h2 className="eyebrow">Or try one of these</h2>
            <dl>
              {Object.entries(examples).map(([cat, items]) => (
                <div key={cat} className="example-group">
                  <dt>{cat}</dt>
                  <dd>
                    {items.map((ex) => (
                      <button
                        key={ex}
                        type="button"
                        className="example"
                        aria-label={ex}
                        onClick={() => void submitProblem(undefined, ex)}
                      >
                        {EXAMPLE_LABELS[ex] ? <RichText text={EXAMPLE_LABELS[ex]} /> : ex}
                      </button>
                    ))}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        )}

        {error && (
          <div className="banner" role="alert">
            <span>{error}</span>
            <button type="button" className="banner-close" onClick={() => setError(null)} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}

        {turns.length === 0 && !loading && <StatusLegend />}

        <div className="turns">
          {turns.map((t) => (
            <OnTheLines key={t.id}>
              <SolutionCard result={t.result} isFollowUp={t.followUp} />
            </OnTheLines>
          ))}
          {loading && (
            <p className="working" aria-busy="true" aria-label="Working">
              Working it out<span className="dots" aria-hidden>…</span>
            </p>
          )}
          <div ref={bottom} />
        </div>

        {sessionId && turns.length > 0 && (
          <form className="ask followup" onSubmit={submitFollowUp}>
            <label htmlFor="followup" className="sr-only">
              Follow-up question
            </label>
            <span className="eyebrow" aria-hidden>
              Ask more about this question
            </span>
            <div className="ask-row">
              <input
                id="followup"
                value={followUp}
                onChange={(e) => setFollowUp(e.target.value)}
                placeholder="Type here: now integrate it, explain step 2, how was this checked?"
                maxLength={MAX_PROBLEM_CHARS}
              />
              <button type="submit" disabled={loading || !followUp.trim()}>
                Ask
              </button>
            </div>
          </form>
        )}
      </main>

      <footer className="colophon">
        <p>
          The arithmetic is done by SymPy, NumPy and SciPy. A language model may read your question, but it never
          computes the answer, and every result is checked a second, independent way. Still, double-check anything
          that matters.
        </p>
        <p className="colophon-sign">Built by Aekank Patel.</p>
      </footer>
    </div>
  )
}
