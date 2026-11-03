import type { Health, SolveResponse } from './types'

// Absolute backend URL in production (e.g. https://<user>-<space>.hf.space); empty = same origin (dev proxy / nginx).
const BASE = ((import.meta.env.VITE_API_URL as string | undefined) ?? '').trim().replace(/\/+$/, '')
const TIMEOUT_MS = 45_000

export const MAX_PROBLEM_CHARS = 1000

export class ApiError extends Error {
  readonly status: number
  readonly retryAfter: number | null
  constructor(message: string, status: number, retryAfter: number | null = null) {
    super(message)
    this.status = status
    this.retryAfter = retryAfter
  }
}

function messageFor(status: number, detail: string | null, retryAfter: number | null): string {
  switch (status) {
    case 429:
      return `You're sending requests quickly. Please wait ${retryAfter ? `${retryAfter} s` : 'a moment'} and try again.`
    case 503:
      return detail?.includes('busy')
        ? 'The server is busy with other computations. Please try again in a few seconds.'
        : 'The server is starting up (the free host sleeps when idle). Please try again in a minute.'
    case 502:
    case 504:
      return 'The server is starting up (the free host sleeps when idle). Please try again in a minute.'
    case 413:
      return 'That request is too large. Please shorten the problem.'
    case 422:
      return 'Please enter a problem of at most 1,000 characters (plain text, no control characters).'
    case 404:
      return detail ?? 'Not found.'
    default:
      return status >= 500 ? 'The server ran into a problem. Please try again.' : (detail ?? 'Request failed.')
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch (e) {
    const aborted = e instanceof DOMException && e.name === 'AbortError'
    throw new ApiError(
      aborted
        ? 'The server took too long to answer. Please try again.'
        : 'Cannot reach the Mathora API. Check your connection or try again shortly.',
      0,
    )
  } finally {
    clearTimeout(timer)
  }
  if (!res.ok) {
    let detail: string | null = null
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : null
    } catch {
      /* non-JSON error */
    }
    const retry = Number(res.headers.get('Retry-After'))
    const retryAfter = Number.isFinite(retry) && retry > 0 ? retry : null
    throw new ApiError(messageFor(res.status, detail, retryAfter), res.status, retryAfter)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/api/health'),
  examples: () => request<Record<string, string[]>>('/api/examples'),
  solve: (problem: string, sessionId?: string) =>
    request<SolveResponse>('/api/solve', {
      method: 'POST',
      body: JSON.stringify(sessionId ? { problem, session_id: sessionId } : { problem }),
    }),
  followUp: (sessionId: string, question: string) =>
    request<SolveResponse>('/api/followup', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, question }),
    }),
}
