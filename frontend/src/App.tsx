import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { api, ApiError } from './api'
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

export default function App() {
  const [problem, setProblem] = useState('')
  const [followUp, setFollowUp] = useState('')
  const [turns, setTurns] = useState<Turn[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [examples, setExamples] = useState<Record<string, string[]>>(FALLBACK_EXAMPLES)
  const nextId = useRef(1)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
    api.examples().then(setExamples).catch(() => undefined)
  }, [])

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' })
  }, [turns])

  async function run(action: () => Promise<SolveResponse>, isFollowUp: boolean) {
    setLoading(true)
    setError(null)
    try {
      const result = await action()
      setSessionId(result.session_id)
      setTurns((t) => [...t, { id: nextId.current++, result, followUp: isFollowUp }])
      return true
    } catch (e) {
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
    if (await run(() => api.solve(p), false)) setProblem('')
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
      <header className="hero">
        <div className="brand">
          <span className="logo" aria-hidden>
            ∑
          </span>
          <div>
            <h1>MathForge AI</h1>
            <p className="tagline">Solve · Verify · Explain — every answer is independently checked.</p>
          </div>
        </div>
        <div className="status" data-testid="llm-status">
          <span className={`dot ${health ? (health.llm_configured ? 'on' : 'off') : 'down'}`} />
          {health
            ? health.llm_configured
              ? `LLM: ${health.model}`
              : 'Offline planner (no LLM key)'
            : 'API unreachable'}
        </div>
      </header>

      <main>
        <form className="composer" onSubmit={submitProblem}>
          <label htmlFor="problem" className="sr-only">
            Math problem
          </label>
          <textarea
            id="problem"
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a problem, e.g. “integrate x·sin(x) from 0 to pi” or “eigenvalues of [[2,1],[1,2]]”"
            rows={2}
            maxLength={2000}
          />
          <button type="submit" disabled={loading || !problem.trim()}>
            {loading && !sessionId ? 'Solving…' : 'Solve'}
          </button>
        </form>

        <section className="examples" aria-label="Examples">
          {Object.entries(examples).map(([cat, items]) => (
            <div key={cat} className="example-group">
              <span className="example-cat">{cat}</span>
              {items.map((ex) => (
                <button key={ex} type="button" className="example" onClick={() => void submitProblem(undefined, ex)}>
                  {ex}
                </button>
              ))}
            </div>
          ))}
        </section>

        {error && (
          <div className="banner" role="alert">
            {error}
          </div>
        )}

        <div className="turns">
          {turns.map((t) => (
            <SolutionCard key={t.id} result={t.result} isFollowUp={t.followUp} />
          ))}
          {loading && <div className="card skeleton" aria-busy="true" aria-label="Working" />}
          <div ref={bottom} />
        </div>

        {sessionId && turns.length > 0 && (
          <form className="composer followup" onSubmit={submitFollowUp}>
            <label htmlFor="followup" className="sr-only">
              Follow-up question
            </label>
            <input
              id="followup"
              value={followUp}
              onChange={(e) => setFollowUp(e.target.value)}
              placeholder="Ask a follow-up: “now integrate it”, “explain the steps”, “how was this verified?”"
              maxLength={2000}
            />
            <button type="submit" disabled={loading || !followUp.trim()}>
              Ask
            </button>
          </form>
        )}
      </main>

      <footer className="foot">
        Computation by SymPy, NumPy &amp; SciPy · orchestration by LangGraph · the LLM never does the arithmetic.
      </footer>
    </div>
  )
}
