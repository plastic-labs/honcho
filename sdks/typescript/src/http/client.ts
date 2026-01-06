import {
  APIError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ServerError,
} from './exceptions'

export interface HttpClientConfig {
  baseURL: string
  apiKey?: string
  timeout?: number
  maxRetries?: number
  defaultHeaders?: Record<string, string>
}

export class HttpClient {
  private _baseURL: string
  private _apiKey?: string
  private _timeout: number
  private _maxRetries: number
  private _defaultHeaders: Record<string, string>

  constructor(config: HttpClientConfig) {
    this._baseURL = config.baseURL.replace(/\/$/, '')
    this._apiKey = config.apiKey
    this._timeout = config.timeout ?? 60000
    this._maxRetries = config.maxRetries ?? 2
    this._defaultHeaders = config.defaultHeaders ?? {}
  }

  get baseURL(): string {
    return this._baseURL
  }

  get apiKey(): string | undefined {
    return this._apiKey
  }

  private buildHeaders(extra?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this._defaultHeaders,
    }
    if (this._apiKey) {
      headers['Authorization'] = `Bearer ${this._apiKey}`
    }
    if (extra) {
      Object.assign(headers, extra)
    }
    return headers
  }

  private async mapError(response: Response): Promise<APIError> {
    const status = response.status
    let message: string
    try {
      const body = await response.json()
      message = body.detail ?? JSON.stringify(body)
    } catch {
      message = await response.text().catch(() => `HTTP error ${status}`)
    }

    if (status === 401) return new AuthenticationError(message)
    if (status === 404) return new NotFoundError(message)
    if (status === 429) return new RateLimitError(message)
    if (status >= 500) return new ServerError(message, status)
    return new APIError(message, status)
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  /**
   * Make an HTTP request with retry logic.
   */
  async request<T>(
    method: string,
    path: string,
    options?: {
      json?: unknown
      params?: Record<string, string | number | boolean | undefined>
      headers?: Record<string, string>
      formData?: FormData
    }
  ): Promise<T> {
    const url = new URL(path, this._baseURL)
    if (options?.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        if (value !== undefined) {
          url.searchParams.set(key, String(value))
        }
      })
    }

    let lastError: Error | undefined

    for (let attempt = 0; attempt <= this._maxRetries; attempt++) {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), this._timeout)

      try {
        const headers = this.buildHeaders(options?.headers)

        // Remove Content-Type for FormData (browser sets it with boundary)
        if (options?.formData) {
          delete headers['Content-Type']
        }

        const response = await fetch(url.toString(), {
          method,
          headers,
          body:
            options?.formData ??
            (options?.json !== undefined
              ? JSON.stringify(options.json)
              : undefined),
          signal: controller.signal,
        })

        clearTimeout(timeoutId)

        if (response.ok) {
          // Handle 204 No Content
          if (response.status === 204) {
            return undefined as T
          }
          const text = await response.text()
          if (!text) {
            return undefined as T
          }
          return JSON.parse(text) as T
        }

        // Retry on 5xx errors
        if (response.status >= 500 && attempt < this._maxRetries) {
          lastError = await this.mapError(response)
          await this.sleep(2 ** attempt * 1000)
          continue
        }

        throw await this.mapError(response)
      } catch (error) {
        clearTimeout(timeoutId)

        if (error instanceof APIError) throw error

        // Handle abort/timeout
        if (error instanceof Error && error.name === 'AbortError') {
          lastError = new APIError('Request timed out')
          if (attempt < this._maxRetries) {
            await this.sleep(2 ** attempt * 1000)
            continue
          }
          throw lastError
        }

        lastError = error instanceof Error ? error : new Error(String(error))
        if (attempt < this._maxRetries) {
          await this.sleep(2 ** attempt * 1000)
          continue
        }
        throw new APIError(lastError.message)
      }
    }

    throw lastError ?? new APIError('Max retries exceeded')
  }

  /**
   * Stream HTTP response for SSE.
   */
  async *stream(
    method: string,
    path: string,
    options?: { json?: unknown }
  ): AsyncGenerator<string, void, undefined> {
    const url = `${this._baseURL}${path}`
    const response = await fetch(url, {
      method,
      headers: this.buildHeaders({ Accept: 'text/event-stream' }),
      body: options?.json ? JSON.stringify(options.json) : undefined,
    })

    if (!response.ok || !response.body) {
      throw await this.mapError(response)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line) yield line
        }
      }
      // Yield any remaining content
      if (buffer) yield buffer
    } finally {
      reader.releaseLock()
    }
  }
}
