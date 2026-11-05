import type { ReactNode } from 'react'
import type { SolveResponse } from '../types'
import { IntentPanel, OverallBadge } from './IntentPanel'
import { PlannerBadge, PlannerNotice } from './PlannerInfo'
import { Plot } from './Plot'
import { prettyMath } from '../prettyMath'
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

function MultiPartCard({ result, isFollowUp }: { result: SolveResponse; isFollowUp?: boolean }) {
  const parts = result.parts ?? []
  const answered = parts.filter((p) => p.status === 'solved').length
  return (
    <article className="sheet multi-card" data-testid="multipart-card">
      <header className="sheet-head">
        <p className="sheet-meta">
          <span>
            {isFollowUp ? 'Follow-up · ' : ''}
            {parts.length} parts
          </span>
        </p>
        <h3 className="problem">
          <RichText text={result.problem} />
        </h3>
        {result.overall_status && <OverallBadge status={result.overall_status} />}
      </header>
      <p className="sheet-note">
        {result.split_by === 'llm' ? 'The AI planner split this into parts' : 'Split into parts'}; each one was read,
        checked and verified on its own, and the mark above is the weakest of them.
        {answered < parts.length ? ` ${parts.length - answered} of ${parts.length} could not be answered.` : ''}
      </p>
      <ol className="parts">
        {parts.map((part, i) => (
          <li key={i} className="part" data-testid="part">
            <span className="part-label">({String.fromCharCode(97 + i)})</span>
            <SolutionCard result={part} nested />
          </li>
        ))}
      </ol>
    </article>
  )
}

function Row({ label, children, wide }: { label: string; children: ReactNode; wide?: boolean }) {
  return (
    <section className={`row${wide ? ' row-wide' : ''}`} aria-label={label}>
      <h4 className="margin">{label}</h4>
      <div className="row-body">{children}</div>
    </section>
  )
}

function Head({ result, kind }: { result: SolveResponse; kind?: string }) {
  return (
    <header className="sheet-head">
      <p className="sheet-meta">
        <span>{kind ?? CATEGORY_LABEL[result.category]}</span>
      </p>
      <h3 className="problem">
        <RichText text={result.problem} />
      </h3>
    </header>
  )
}

export function SolutionCard({
  result,
  isFollowUp,
  nested,
}: {
  result: SolveResponse
  isFollowUp?: boolean
  nested?: boolean
}) {
  if (result.parts && result.parts.length > 0) return <MultiPartCard result={result} isFollowUp={isFollowUp} />
  const cls = nested ? ' nested' : ''
  if (result.status === 'answered') {
    return (
      <article className={`sheet answer-card${cls}`} data-testid="followup-answer">
        <Head result={result} kind="Tutor" />
        <p className="explanation">
          <RichText text={result.explanation ?? ''} />
        </p>
      </article>
    )
  }

  if (result.status === 'needs_clarification') {
    return (
      <article className={`sheet clarify-card${cls}`} role="alert" data-testid="clarify-card">
        <Head result={result} kind="Needs clarification" />
        <p className="lead">{prettyMath(result.error ?? '')}</p>
        <PlannerNotice result={result} />
        {result.intent?.interpretation && (
          <p className="muted">
            Closest reading I could form: {prettyMath(result.intent.interpretation)} I did not compute it, because it may not be what
            you asked.
          </p>
        )}
        <p className="muted small">Add the missing information and ask again.</p>
      </article>
    )
  }

  if (result.status !== 'solved') {
    const label =
      result.status === 'unsupported'
        ? 'Not supported'
        : result.error_code === 'timeout'
          ? 'Time limit reached'
          : result.error_code === 'memory' || result.error_code === 'too_large'
            ? 'Too large to compute'
            : 'Could not solve'
    return (
      <article className={`sheet error-card${cls}`} role="alert" data-testid="error-card">
        <Head result={result} kind={label} />
        <p className="lead">{prettyMath(result.error ?? '')}</p>
        {result.status === 'unsupported' && (
          <p className="muted small">
            It can do equations and systems, simplifying, factoring and expanding, derivatives, integrals, limits,
            matrices, statistics, probability distributions and counting.
          </p>
        )}
      </article>
    )
  }

  return (
    <article className={`sheet${cls}`} data-testid="solution-card">
      <header className="sheet-head">
        <p className="sheet-meta">
          <span>
            {isFollowUp ? 'Follow-up · ' : ''}
            {CATEGORY_LABEL[result.category]}
          </span>
          <PlannerBadge result={result} />
        </p>
        <h3 className="problem">
          <RichText text={result.problem} />
        </h3>
        {result.overall_status ? (
          <OverallBadge status={result.overall_status} />
        ) : (
          result.verification && <VerificationBadge status={result.verification.status} />
        )}
      </header>

      <PlannerNotice result={result} />

      <Row label="Answer">
        <div className="answer">
          {result.answer_latex ? <Tex math={result.answer_latex} display /> : <code>{prettyMath(result.answer_text ?? '')}</code>}
        </div>
      </Row>

      {result.intent && (
        <Row label="Reading">
          <IntentPanel intent={result.intent} />
        </Row>
      )}
      {result.verification && (
        <Row label="Check">
          <VerificationPanel verification={result.verification} />
        </Row>
      )}

      {result.steps.length > 0 && (
        <Row label="Working">
          <ol className="steps">
            {result.steps.map((s, i) => (
              <li key={i}>
                <div className="step-title">{prettyMath(s.title)}</div>
                {s.detail && <div className="step-detail">{prettyMath(s.detail)}</div>}
                {s.latex && (
                  <div className="step-math">
                    <Tex math={s.latex} display />
                  </div>
                )}
              </li>
            ))}
          </ol>
        </Row>
      )}

      {result.visualization && (
        <Row label="Graph">
          <Plot figure={result.visualization} />
        </Row>
      )}

      {result.explanation && (
        <Row label="In words">
          <p className="explanation">
            <RichText text={result.explanation} />
          </p>
        </Row>
      )}

      <footer className="meta">
        <span>
          tool <code>{result.tool_call?.tool}</code>
        </span>
        <span>
          planner{' '}
          {result.planner === 'llm'
            ? 'Groq LLM (it plans, it never computes)'
            : result.planner === 'nl'
              ? 'offline language analysis'
              : 'offline rules'}
        </span>
      </footer>
    </article>
  )
}
