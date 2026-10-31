import type { Health, SolveResponse } from './types'

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ''

export class ApiError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch {
    throw new ApiError('Cannot reach the MathForge API. Is the backend running?', 0)
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : 'Invalid request.'
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/api/health'),
  examples: () => request<Record<string, string[]>>('/api/examples'),
  solve: (problem: string, sessionId?: string) =>
    request<SolveResponse>('/api/solve', {
      method: 'POST',
      body: JSON.stringify({ problem, session_id: sessionId ?? null }),
    }),
  followUp: (sessionId: string, question: string) =>
    request<SolveResponse>('/api/followup', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, question }),
    }),
}
