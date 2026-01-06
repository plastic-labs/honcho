/**
 * Base exception for API errors.
 */
export class APIError extends Error {
  readonly statusCode: number | undefined

  constructor(message: string, statusCode?: number) {
    super(message)
    this.name = 'APIError'
    this.statusCode = statusCode
  }
}

/**
 * Raised for 401 authentication errors.
 */
export class AuthenticationError extends APIError {
  constructor(message: string) {
    super(message, 401)
    this.name = 'AuthenticationError'
  }
}

/**
 * Raised for 404 not found errors.
 */
export class NotFoundError extends APIError {
  constructor(message: string) {
    super(message, 404)
    this.name = 'NotFoundError'
  }
}

/**
 * Raised for 429 rate limit errors.
 */
export class RateLimitError extends APIError {
  constructor(message: string) {
    super(message, 429)
    this.name = 'RateLimitError'
  }
}

/**
 * Raised for 5xx server errors.
 */
export class ServerError extends APIError {
  constructor(message: string, statusCode: number = 500) {
    super(message, statusCode)
    this.name = 'ServerError'
  }
}
