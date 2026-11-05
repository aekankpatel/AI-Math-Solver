import { prettyMath } from '../prettyMath'
import type { IntentVerification, OverallStatus } from '../types'

const OVERALL: Record<OverallStatus, { label: string; icon: string; cls: string; hint: string }> = {
  verified: {
    label: 'Fully verified',
    icon: '✓',
    cls: 'verified',
    hint: 'The math was independently verified and the interpretation matches every part of the question',
  },
  verified_with_assumptions: {
    label: 'Verified with assumptions',
    icon: '✓',
    cls: 'assumed',
    hint: 'The math is verified, but the interpretation relies on the assumptions listed below',
  },
  approximate: {
    label: 'Approximate',
    icon: '≈',
    cls: 'approximate',
    hint: 'Confirmed numerically within tolerance, not symbolically',
  },
  unverified: { label: 'Unverified', icon: '?', cls: 'unverified', hint: 'No independent check was possible' },
  failed: { label: 'Verification failed', icon: '✕', cls: 'failed', hint: 'Do not trust this result' },
  needs_clarification: {
    label: 'Needs clarification',
    icon: '?',
    cls: 'clarify',
    hint: 'More information is needed; nothing was computed',
  },
}

export function OverallBadge({ status, testId = 'overall-badge' }: { status: OverallStatus; testId?: string }) {
  const meta = OVERALL[status]
  return (
    <span className={`stamp stamp-${meta.cls}`} title={meta.hint} data-testid={testId}>
      {meta.label}
    </span>
  )
}

const INTENT_LABEL: Record<string, string> = {
  confirmed: 'Interpretation confirmed',
  assumed: 'Interpretation relies on assumptions',
  ambiguous: 'Question is ambiguous',
  mismatch: 'Plan did not match the question',
  unchecked: 'Interpretation not checked',
}

export function IntentPanel({ intent }: { intent: IntentVerification }) {
  const failed = intent.checks.filter((c) => !c.passed)
  return (
    <details className="fold intent" open={intent.status !== 'confirmed'} data-testid="intent-panel">
      <summary>
        <span className="fold-title">{prettyMath(intent.interpretation)}</span>
        <span className={`tag tag-${intent.status === 'confirmed' ? 'ok' : 'note'}`}>{INTENT_LABEL[intent.status]}</span>
      </summary>
      {intent.assumptions.length > 0 && (
        <ul className="assumptions">
          {intent.assumptions.map((a, i) => (
            <li key={i}>{prettyMath(a)}</li>
          ))}
        </ul>
      )}
      <ul className="checks">
        {intent.checks
          .filter((c) => c.severity !== 'info' || c.passed)
          .map((c, i) => (
            <li key={i} className={c.passed ? 'pass' : 'fail'}>
              <span className="check-icon">{c.passed ? '✓' : c.severity === 'warning' ? '!' : '✕'}</span>
              <span className="check-method">{c.name}</span>
              {c.detail && <span className="check-detail">{prettyMath(c.detail)}</span>}
            </li>
          ))}
      </ul>
      {failed.length === 0 && intent.assumptions.length === 0 && (
        <p className="muted small">Every number and condition in the question was matched to the computation.</p>
      )}
    </details>
  )
}

const LEGEND: OverallStatus[] = ['verified', 'verified_with_assumptions', 'approximate', 'needs_clarification']

/** Explains the trust levels shown on every answer. */
export function StatusLegend() {
  return (
    <section className="legend" data-testid="status-legend" aria-label="What the marks mean">
      <h2 className="eyebrow">What the marks mean</h2>
      <ul>
        {LEGEND.map((s) => (
          <li key={s}>
            <OverallBadge status={s} testId="legend-badge" />
            <span className="muted">{OVERALL[s].hint}.</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
