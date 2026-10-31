import type { SolveResponse } from '../types'
import { Plot } from './Plot'
import { RichText, Tex } from './Tex'
import { VerificationBadge, VerificationPanel } from './VerificationPanel'

const CATEGORY_LABEL: Record<string, string> = {
  algebra: 'Algebra',
  calculus: 'Calculus',
  integrals: 'Integrals',
  linear_algebra: 'Linear algebra',
  probability_statistics: 'Probability & statistics',
  unsupported: 'Unsupported',
}

export function SolutionCard({ result, isFollowUp }: { result: SolveResponse; isFollowUp?: boolean }) {
  if (result.status === 'answered') {
    return (
      <article className="card answer-card" data-testid="followup-answer">
        <header className="card-head">
          <span className="chip">Tutor</span>
          <h3 className="problem">{result.problem}</h3>
        </header>
        <p className="explanation">
          <RichText text={result.explanation ?? ''} />
        </p>
      </article>
    )
  }

  if (result.status !== 'solved') {
    return (
      <article className="card error-card" role="alert" data-testid="error-card">
        <header className="card-head">
          <span className="chip chip-warn">{result.status === 'unsupported' ? 'Not supported' : 'Could not solve'}</span>
          <h3 className="problem">{result.problem}</h3>
        </header>
        <p>{result.error}</p>
      </article>
    )
  }

  return (
    <article className="card" data-testid="solution-card">
      <header className="card-head">
        <span className="chip">{isFollowUp ? 'Follow-up · ' : ''}{CATEGORY_LABEL[result.category]}</span>
        {result.verification && <VerificationBadge status={result.verification.status} />}
        <h3 className="problem">{result.problem}</h3>
      </header>

      <section className="answer" aria-label="Answer">
        <span className="answer-label">Answer</span>
        {result.answer_latex ? <Tex math={result.answer_latex} display /> : <code>{result.answer_text}</code>}
      </section>

      {result.verification && <VerificationPanel verification={result.verification} />}

      <div className="grid">
        <section className="panel steps" aria-label="Solution steps">
          <h4>Step-by-step</h4>
          <ol>
            {result.steps.map((s, i) => (
              <li key={i}>
                <div className="step-title">{s.title}</div>
                {s.detail && <div className="step-detail">{s.detail}</div>}
                {s.latex && (
                  <div className="step-math">
                    <Tex math={s.latex} display />
                  </div>
                )}
              </li>
            ))}
          </ol>
        </section>

        {result.visualization && (
          <section className="panel viz" aria-label="Visualization">
            <h4>Visualization</h4>
            <Plot figure={result.visualization} />
          </section>
        )}
      </div>

      {result.explanation && (
        <section className="panel">
          <h4>Explanation</h4>
          <p className="explanation">
            <RichText text={result.explanation} />
          </p>
        </section>
      )}

      <footer className="meta">
        <span>
          Tool: <code>{result.tool_call?.tool}</code>
        </span>
        <span>Planner: {result.planner === 'llm' ? 'LLM (Groq)' : 'deterministic rules'}</span>
        {result.tool_call?.rationale && <span className="muted">{result.tool_call.rationale}</span>}
      </footer>
    </article>
  )
}
