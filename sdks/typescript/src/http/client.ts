import {
  ConnectionError,
  createErrorFromResponse,
  RateLimitError,
  ServerError,
  TimeoutError,
} from './errors'

export interface HonchoHTTPClientConfig {
  baseURL: string
  apiKey?: string
  timeout?: number
  maxRetries?: number
  defaultHeaders?: Record<string, string>
}

export interface RequestOptions {
  body?: unknown
  query?: Record<string, string | number | boolean | undefined>
  headers?: Record<string, string>
  timeout?: number
  signal?: AbortSignal
}

const DEFAULT_TIMEOUT = 60000 // 60 seconds
const DEFAULT_MAX_RETRIES = 2
const RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
const INITIAL_RETRY_DELAY = 500 // 500ms

/**
 * Minimal HTTP client for the Honcho API with retry logic and timeout support.
 */
export class HonchoHTTPClient {
  readonly baseURL: string
  readonly apiKey?: string
  readonly timeout: number
  readonly maxRetries: number
  readonly defaultHeaders: Record<string, string>

  constructor(config: HonchoHTTPClientConfig) {
    // Remove trailing slash from baseURL
    this.baseURL = config.baseURL.replace(/\/$/, '')
    this.apiKey = config.apiKey
    this.timeout = config.timeout ?? DEFAULT_TIMEOUT
    this.maxRetries = config.maxRetries ?? DEFAULT_MAX_RETRIES
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...config.defaultHeaders,
    }
  }

  /**
   * Make an HTTP request with automatic retries and timeout handling.
   */
  async request<T>(
    method: string,
    path: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const url = this.buildURL(path, options.query)
    const headers = this.buildHeaders(options.headers)
    const timeout = options.timeout ?? this.timeout

    let lastError: Error | undefined
    let attempt = 0

    while (attempt <= this.maxRetries) {
      try {
        const response = await this.fetchWithTimeout(
          url,
          {
            method,
            headers,
            body: options.body ? JSON.stringify(options.body) : undefined,
            signal: options.signal,
          },
          timeout
        )

        if (response.ok) {
          // Handle empty responses
          const text = await response.text()
          if (!text) {
            return undefined as T
          }
          return JSON.parse(text) as T
        }

        // Handle error responses
        const errorBody = await this.parseErrorBody(response)
        const retryAfter = this.parseRetryAfter(response)
        const error = createErrorFromResponse(
          response.status,
          errorBody.message || `HTTP ${response.status}`,
          errorBody,
          retryAfter
        )

        // Only retry on specific status codes
        if (
          RETRY_STATUS_CODES.includes(response.status) &&
          attempt < this.maxRetries
        ) {
          lastError = error
          await this.sleep(this.getRetryDelay(attempt, retryAfter))
          attempt++
          continue
        }

        throw error
      } catch (error) {
        if (error instanceof TimeoutError || error instanceof ConnectionError) {
          // Retry on network errors
          if (attempt < this.maxRetries) {
            lastError = error
            await this.sleep(this.getRetryDelay(attempt))
            attempt++
            continue
          }
        }

        // Don't retry on other errors, just throw
        if (
          error instanceof RateLimitError ||
          error instanceof ServerError ||
          error instanceof TimeoutError ||
          error instanceof ConnectionError
        ) {
          throw error
        }

        // Handle fetch errors (network issues)
        if (error instanceof TypeError && error.message.includes('fetch')) {
          const connError = new ConnectionError(error.message)
          if (attempt < this.maxRetries) {
            lastError = connError
            await this.sleep(this.getRetryDelay(attempt))
            attempt++
            continue
          }
          throw connError
        }

        throw error
      }
    }

    // If we exhausted retries, throw the last error
    throw lastError || new Error('Request failed after retries')
  }

  /**
   * Make a GET request.
   */
  async get<T>(
    path: string,
    options?: Omit<RequestOptions, 'body'>
  ): Promise<T> {
    return this.request<T>('GET', path, options)
  }

  /**
   * Make a POST request.
   */
  async post<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>('POST', path, options)
  }

  /**
   * Make a PUT request.
   */
  async put<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>('PUT', path, options)
  }

  /**
   * Make a PATCH request.
   */
  async patch<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>('PATCH', path, options)
  }

  /**
   * Make a DELETE request.
   */
  async delete<T>(
    path: string,
    options?: Omit<RequestOptions, 'body'>
  ): Promise<T> {
    return this.request<T>('DELETE', path, options)
  }

  /**
   * Make a streaming request that returns a Response object for SSE parsing.
   */
  async stream(
    method: string,
    path: string,
    options: RequestOptions = {}
  ): Promise<Response> {
    const url = this.buildURL(path, options.query)
    const headers = {
      ...this.buildHeaders(options.headers),
      Accept: 'text/event-stream',
    }
    const timeout = options.timeout ?? this.timeout

    const response = await this.fetchWithTimeout(
      url,
      {
        method,
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: options.signal,
      },
      timeout
    )

    if (!response.ok) {
      const errorBody = await this.parseErrorBody(response)
      throw createErrorFromResponse(
        response.status,
        errorBody.message || `HTTP ${response.status}`,
        errorBody
      )
    }

    return response
  }

  /**
   * Make a multipart form data request (for file uploads).
   */
  async upload<T>(
    path: string,
    formData: FormData,
    options: Omit<RequestOptions, 'body'> = {}
  ): Promise<T> {
    const url = this.buildURL(path, options.query)
    // Don't set Content-Type for FormData - browser will set it with boundary
    const headers: Record<string, string> = {}
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`
    }
    if (options.headers) {
      Object.assign(headers, options.headers)
    }

    const timeout = options.timeout ?? this.timeout

    const response = await this.fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers,
        body: formData,
        signal: options.signal,
      },
      timeout
    )

    if (!response.ok) {
      const errorBody = await this.parseErrorBody(response)
      throw createErrorFromResponse(
        response.status,
        errorBody.message || `HTTP ${response.status}`,
        errorBody
      )
    }

    const text = await response.text()
    if (!text) {
      return undefined as T
    }
    return JSON.parse(text) as T
  }

  private buildURL(
    path: string,
    query?: Record<string, string | number | boolean | undefined>
  ): string {
    const url = new URL(path, this.baseURL)

    if (query) {
      for (const [key, value] of Object.entries(query)) {
        if (value !== undefined) {
          url.searchParams.set(key, String(value))
        }
      }
    }

    return url.toString()
  }

  private buildHeaders(extra?: Record<string, string>): Record<string, string> {
    const headers: Record<string, string> = { ...this.defaultHeaders }

    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`
    }

    if (extra) {
      Object.assign(headers, extra)
    }

    return headers
  }

  private async fetchWithTimeout(
    url: string,
    init: RequestInit,
    timeout: number
  ): Promise<Response> {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    // Combine with any existing signal
    if (init.signal) {
      init.signal.addEventListener('abort', () => controller.abort())
    }

    try {
      const response = await fetch(url, {
        ...init,
        signal: controller.signal,
      })
      return response
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new TimeoutError(`Request timed out after ${timeout}ms`)
      }
      throw error
    } finally {
      clearTimeout(timeoutId)
    }
  }

  private async parseErrorBody(
    response: Response
  ): Promise<{ message?: string; detail?: string }> {
    try {
      const body = await response.json()
      return {
        message: body.detail || body.message || body.error,
        ...body,
      }
    } catch {
      return { message: `HTTP ${response.status}` }
    }
  }

  private parseRetryAfter(response: Response): number | undefined {
    const header = response.headers.get('Retry-After')
    if (!header) return undefined

    const seconds = Number.parseInt(header, 10)
    if (!Number.isNaN(seconds)) {
      return seconds * 1000 // Convert to milliseconds
    }

    // Try parsing as date
    const date = Date.parse(header)
    if (!Number.isNaN(date)) {
      return Math.max(0, date - Date.now())
    }

    return undefined
  }

  private getRetryDelay(attempt: number, retryAfter?: number): number {
    if (retryAfter) {
      return retryAfter
    }
    // Exponential backoff: 500ms, 1000ms, 2000ms, etc.
    return INITIAL_RETRY_DELAY * 2 ** attempt
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }
}
