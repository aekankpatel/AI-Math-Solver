import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { VerificationBadge } from './components/VerificationPanel'
import type { SolveResponse } from './types'

vi.mock('./components/Plot', () => ({ Plot: () => <div data-testid="plot" /> }))

const solved: SolveResponse = {
  session_id: 'abc',
  problem: 'Solve x^2 - 5x + 6 = 0',
  status: 'solved',
  category: 'algebra',
  tool_call: { tool: 'solve_equation', category: 'algebra', args: {}, rationale: 'Single equation.' },
  answer_text: 'x = 2, x = 3',
  answer_latex: 'x = 2, x = 3',
  steps: [{ title: 'Factor', detail: '', latex: '(x-2)(x-3)=0' }],
  explanation: 'We factored the quadratic.',
  verification: {
    status: 'verified',
    summary: 'Confirmed.',
    checks: [{ method: 'substitution (symbolic)', passed: true, detail: '' }],
  },
  visualization: { data: [], layout: {} },
  error: null,
  llm_used: false,
  planner: 'rules',
}

function mockFetch(routes: Record<string, unknown>) {
  return vi.fn(async (url: string) => {
    const key = Object.keys(routes).find((k) => url.endsWith(k))
    if (!key) return new Response('{}', { status: 404 })
    return new Response(JSON.stringify(routes[key]), { status: 200, headers: { 'Content-Type': 'application/json' } })
  })
}

describe('App', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      mockFetch({
        '/api/health': { status: 'ok', llm_configured: false, model: 'llama' },
        '/api/examples': { Algebra: ['Solve x^2 - 5x + 6 = 0'] },
        '/api/solve': solved,
        '/api/followup': { ...solved, status: 'answered', problem: 'why?', explanation: 'Because factoring.' },
      }),
    )
  })
  afterEach(() => vi.unstubAllGlobals())

  it('solves a problem and shows answer, verification, steps and plot', async () => {
    render(<App />)
    fireEvent.change(screen.getByLabelText('Math problem'), { target: { value: 'Solve x^2 - 5x + 6 = 0' } })
    fireEvent.click(screen.getByRole('button', { name: 'Solve' }))
    expect(await screen.findByTestId('solution-card')).toBeInTheDocument()
    expect(screen.getAllByTestId('verification-badge')[0]).toHaveTextContent('Verified')
    expect(screen.getByText('Factor')).toBeInTheDocument()
    expect(screen.getByTestId('plot')).toBeInTheDocument()
    expect(screen.getByText('Offline planner (no LLM key)')).toBeInTheDocument()
  })

  it('supports follow-up questions in the same session', async () => {
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: 'Solve x^2 - 5x + 6 = 0' }))
    await screen.findByTestId('solution-card')
    fireEvent.change(screen.getByLabelText('Follow-up question'), { target: { value: 'why?' } })
    fireEvent.click(screen.getByRole('button', { name: 'Ask' }))
    expect(await screen.findByTestId('followup-answer')).toHaveTextContent('Because factoring.')
  })

  it('shows unsupported problems as errors, not answers', async () => {
    vi.stubGlobal(
      'fetch',
      mockFetch({
        '/api/solve': { ...solved, status: 'unsupported', answer_text: null, answer_latex: null, error: 'Not math.' },
      }),
    )
    render(<App />)
    fireEvent.change(screen.getByLabelText('Math problem'), { target: { value: 'write a poem' } })
    fireEvent.click(screen.getByRole('button', { name: 'Solve' }))
    expect(await screen.findByTestId('error-card')).toHaveTextContent('Not math.')
    expect(screen.queryByTestId('solution-card')).not.toBeInTheDocument()
  })

  it('reports an unreachable backend', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('network'))))
    render(<App />)
    fireEvent.change(screen.getByLabelText('Math problem'), { target: { value: 'x=1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Solve' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Cannot reach the MathForge API'))
  })
})

describe('VerificationBadge', () => {
  it.each([
    ['verified', 'Verified'],
    ['approximate', 'Approximate'],
    ['unverified', 'Unverified'],
    ['failed', 'Verification failed'],
  ] as const)('renders %s', (status, label) => {
    render(<VerificationBadge status={status} />)
    expect(screen.getByTestId('verification-badge')).toHaveTextContent(label)
  })
})
