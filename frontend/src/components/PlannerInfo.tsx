import type { Health, SolveResponse } from '../types'

const OFFLINE_LABEL: Record<SolveResponse['planner'], string> = {
  llm: 'Groq LLM',
  rules: 'offline planner · rules',
  nl: 'offline planner · language analysis',
}

/** Who turned the question into a computation: the Groq LLM or the offline planner. */
export function PlannerBadge({ result }: { result: SolveResponse }) {
  const llm = result.planner === 'llm'
  return (
    <span
      className={`planner-badge ${llm ? 'planner-llm' : 'planner-offline'}`}
      data-testid="planner-badge"
      title={
        llm
          ? 'The Groq LLM chose the tool and arguments; SymPy/NumPy/SciPy did the math and it was checked independently.'
          : 'A deterministic offline planner chose the tool and arguments; the math was checked independently.'
      }
    >
      Planned by {OFFLINE_LABEL[result.planner]}
    </span>
  )
}

export function PlannerNotice({ result }: { result: SolveResponse }) {
  if (!result.planner_notice) return null
  return (
    <p className="notice" role="status" data-testid="planner-notice">
      {result.planner_notice}
    </p>
  )
}

export function HealthPill({ health, reachable }: { health: Health | null; reachable: boolean }) {
  let cls = 'down'
  let text = 'API unreachable'
  if (health) {
    const state = health.llm_state ?? (health.llm_configured ? 'available' : 'disabled')
    if (state === 'available') {
      cls = 'on'
      text = `Groq planner · ${health.model}`
    } else if (state === 'budget') {
      cls = 'off'
      text = 'Offline planner · daily AI budget used'
    } else if (state === 'cooldown') {
      cls = 'off'
      text = 'Offline planner · AI planner paused'
    } else {
      cls = 'off'
      text = 'Offline planner'
    }
  } else if (reachable) {
    cls = 'off'
    text = 'Connecting to the server…'
  }
  return (
    <div className="status" data-testid="llm-status" role="status">
      <span className={`dot ${cls}`} />
      {text}
    </div>
  )
}
