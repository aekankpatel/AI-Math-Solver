export type VerificationStatus = 'verified' | 'approximate' | 'unverified' | 'failed'

export type Category =
  | 'algebra'
  | 'calculus'
  | 'integrals'
  | 'linear_algebra'
  | 'probability_statistics'
  | 'unsupported'

export interface Step {
  title: string
  detail: string
  latex: string | null
}

export interface VerificationCheck {
  method: string
  passed: boolean
  detail: string
}

export interface Verification {
  status: VerificationStatus
  checks: VerificationCheck[]
  summary: string
}

export type IntentStatus = 'confirmed' | 'assumed' | 'ambiguous' | 'mismatch' | 'unchecked'

export type OverallStatus =
  | 'verified'
  | 'verified_with_assumptions'
  | 'approximate'
  | 'unverified'
  | 'failed'
  | 'needs_clarification'

export interface IntentCheck {
  name: string
  passed: boolean
  detail: string
  severity: 'error' | 'missing' | 'warning' | 'info'
}

export interface IntentVerification {
  status: IntentStatus
  interpretation: string
  checks: IntentCheck[]
  assumptions: string[]
  clarification: string | null
  repaired: boolean
}

export interface ToolCall {
  tool: string
  category: Category
  args: Record<string, unknown>
  rationale: string
}

export interface PlotlyFigure {
  data: Record<string, unknown>[]
  layout: Record<string, unknown>
}

export interface SolveResponse {
  session_id: string
  problem: string
  status: 'solved' | 'unsupported' | 'error' | 'answered' | 'needs_clarification'
  category: Category
  tool_call: ToolCall | null
  answer_text: string | null
  answer_latex: string | null
  steps: Step[]
  explanation: string | null
  verification: Verification | null
  intent?: IntentVerification | null
  overall_status?: OverallStatus | null
  visualization: PlotlyFigure | null
  error: string | null
  llm_used: boolean
  planner: 'llm' | 'rules' | 'nl'
  llm_status?: LlmStatus
  planner_notice?: string | null
  error_code?: 'timeout' | 'memory' | 'too_large' | 'busy' | null
  parts?: SolveResponse[] | null
  split_by?: 'rules' | 'llm' | null
}

export type LlmStatus =
  | 'used'
  | 'rejected'
  | 'disabled'
  | 'budget'
  | 'cooldown'
  | 'client_quota'
  | 'error'
  | 'unsupported'
  | 'not_needed'

export type LlmState = 'available' | 'disabled' | 'budget' | 'cooldown'

export interface Health {
  status: 'ok'
  llm_configured: boolean
  model: string
  llm_available?: boolean
  llm_state?: LlmState
}
