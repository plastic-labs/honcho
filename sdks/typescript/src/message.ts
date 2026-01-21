import type { MessageResponse } from './types/api'
import type { MessageConfiguration } from './validation'

/**
 * Input for creating a message.
 *
 * This is the type returned by `Peer.message()` and accepted by
 * `Session.addMessages()`.
 */
export interface MessageInput {
  /** The peer ID who authored this message */
  peerId: string
  /** The message content */
  content: string
  /** Optional metadata to associate with the message */
  metadata?: Record<string, unknown>
  /** Optional configuration for the message (reasoning settings) */
  configuration?: MessageConfiguration
  /** Optional ISO 8601 timestamp for when the message was created */
  createdAt?: string
}

/**
 * A message in a Honcho session.
 */
export class Message {
  /** Unique identifier for this message */
  readonly id: string
  /** The message content */
  readonly content: string
  /** The peer ID who authored this message */
  readonly peerId: string
  /** The session ID this message belongs to */
  readonly sessionId: string
  /** The workspace ID this message belongs to */
  readonly workspaceId: string
  /** Metadata associated with this message */
  readonly metadata: Record<string, unknown>
  /** ISO 8601 timestamp for when the message was created */
  readonly createdAt: string
  /** Number of tokens in this message */
  readonly tokenCount: number

  constructor(
    id: string,
    content: string,
    peerId: string,
    sessionId: string,
    workspaceId: string,
    metadata: Record<string, unknown>,
    createdAt: string,
    tokenCount: number
  ) {
    this.id = id
    this.content = content
    this.peerId = peerId
    this.sessionId = sessionId
    this.workspaceId = workspaceId
    this.metadata = metadata
    this.createdAt = createdAt
    this.tokenCount = tokenCount
  }

  /**
   * Create a Message from an API response.
   */
  static fromApiResponse(data: MessageResponse): Message {
    return new Message(
      data.id,
      data.content,
      data.peer_id,
      data.session_id,
      data.workspace_id,
      data.metadata,
      data.created_at,
      data.token_count
    )
  }

  toString(): string {
    const truncatedContent =
      this.content.length > 50
        ? `${this.content.slice(0, 50)}...`
        : this.content
    return `Message(id='${this.id}', peerId='${this.peerId}', content='${truncatedContent}')`
  }
}
