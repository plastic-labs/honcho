/**
 * Base error class for all Honcho SDK errors.
 */
export class HonchoError extends Error {
  readonly status: number
  readonly code?: string
  readonly body?: unknown

  constructor(
    message: string,
    status: number,
    options?: { code?: string; body?: unknown }
  ) {
    super(message)
    this.name = 'HonchoError'
    this.status = status
    this.code = options?.code
    this.body = options?.body
  }
}

/**
 * Error thrown when request validation fails (400).
 */
export class BadRequestError extends HonchoError {
  constructor(message: string, body?: unknown) {
    super(message, 400, { code: 'bad_request', body })
    this.name = 'BadRequestError'
  }
}

/**
 * Error thrown when authentication fails (401).
 */
export class AuthenticationError extends HonchoError {
  constructor(message = 'Authentication failed') {
    super(message, 401, { code: 'authentication_error' })
    this.name = 'AuthenticationError'
  }
}

/**
 * Error thrown when the user lacks permission (403).
 */
export class PermissionDeniedError extends HonchoError {
  constructor(message = 'Permission denied') {
    super(message, 403, { code: 'permission_denied' })
    this.name = 'PermissionDeniedError'
  }
}

/**
 * Error thrown on resource conflict (409).
 */
export class ConflictError extends HonchoError {
  constructor(message = 'Resource conflict', body?: unknown) {
    super(message, 409, { code: 'conflict', body })
    this.name = 'ConflictError'
  }
}

/**
 * Error thrown when entity cannot be processed (422).
 */
export class UnprocessableEntityError extends HonchoError {
  constructor(message = 'Unprocessable entity', body?: unknown) {
    super(message, 422, { code: 'unprocessable_entity', body })
    this.name = 'UnprocessableEntityError'
  }
}

/**
 * Error thrown when a resource is not found (404).
 */
export class NotFoundError extends HonchoError {
  constructor(message = 'Resource not found') {
    super(message, 404, { code: 'not_found' })
    this.name = 'NotFoundError'
  }
}

/**
 * Error thrown when rate limited (429).
 */
export class RateLimitError extends HonchoError {
  readonly retryAfter?: number

  constructor(message = 'Rate limit exceeded', retryAfter?: number) {
    super(message, 429, { code: 'rate_limit_exceeded' })
    this.name = 'RateLimitError'
    this.retryAfter = retryAfter
  }
}

/**
 * Error thrown on server errors (5xx).
 */
export class ServerError extends HonchoError {
  constructor(message = 'Server error', status = 500) {
    super(message, status, { code: 'server_error' })
    this.name = 'ServerError'
  }
}

/**
 * Error thrown when a request times out.
 */
export class TimeoutError extends HonchoError {
  constructor(message = 'Request timed out') {
    super(message, 0, { code: 'timeout' })
    this.name = 'TimeoutError'
  }
}

/**
 * Error thrown when a connection fails.
 */
export class ConnectionError extends HonchoError {
  constructor(message = 'Connection failed') {
    super(message, 0, { code: 'connection_error' })
    this.name = 'ConnectionError'
  }
}

/**
 * Create the appropriate error type based on HTTP status code.
 */
export function createErrorFromResponse(
  status: number,
  message: string,
  body?: unknown,
  retryAfter?: number
): HonchoError {
  switch (status) {
    case 400:
      return new BadRequestError(message, body)
    case 401:
      return new AuthenticationError(message)
    case 403:
      return new PermissionDeniedError(message)
    case 404:
      return new NotFoundError(message)
    case 409:
      return new ConflictError(message, body)
    case 422:
      return new UnprocessableEntityError(message, body)
    case 429:
      return new RateLimitError(message, retryAfter)
    default:
      if (status >= 500) {
        return new ServerError(message, status)
      }
      return new HonchoError(message, status, { body })
  }
}
