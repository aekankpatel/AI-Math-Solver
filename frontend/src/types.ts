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
  status: 'solved' | 'unsupported' | 'error' | 'answered'
  category: Category
  tool_call: ToolCall | null
  answer_text: string | null
  answer_latex: string | null
  steps: Step[]
  explanation: string | null
  verification: Verification | null
  visualization: PlotlyFigure | null
  error: string | null
  llm_used: boolean
  planner: 'llm' | 'rules'
}

export interface Health {
  status: 'ok'
  llm_configured: boolean
  model: string
}
