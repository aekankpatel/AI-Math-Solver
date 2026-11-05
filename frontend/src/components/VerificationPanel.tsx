import { prettyMath } from '../prettyMath'
import type { Verification, VerificationStatus } from '../types'

const LABELS: Record<VerificationStatus, { label: string; icon: string; hint: string }> = {
  verified: { label: 'Verified', icon: '✓', hint: 'Independently confirmed by an exact check' },
  approximate: { label: 'Approximate', icon: '≈', hint: 'Confirmed numerically within tolerance' },
  unverified: { label: 'Unverified', icon: '?', hint: 'No independent check was possible' },
  failed: { label: 'Verification failed', icon: '✕', hint: 'An independent check disagreed — do not trust this result' },
}

export function VerificationBadge({ status }: { status: VerificationStatus }) {
  const meta = LABELS[status]
  return (
    <span className={`tag tag-${status}`} title={meta.hint} data-testid="verification-badge">
      {meta.label}
    </span>
  )
}

export function VerificationPanel({ verification }: { verification: Verification }) {
  return (
    <details className="fold verification" open={verification.status === 'failed'}>
      <summary>
        <span className="fold-title">{prettyMath(verification.summary)}</span>
        <VerificationBadge status={verification.status} />
      </summary>
      <ul className="checks">
        {verification.checks.map((c, i) => (
          <li key={i} className={c.passed ? 'pass' : 'fail'}>
            <span className="check-icon" aria-label={c.passed ? 'passed' : 'failed'}>
              {c.passed ? '✓' : '✕'}
            </span>
            <span className="check-method">{prettyMath(c.method)}</span>
            {c.detail && <span className="check-detail">{prettyMath(c.detail)}</span>}
          </li>
        ))}
      </ul>
    </details>
  )
}
