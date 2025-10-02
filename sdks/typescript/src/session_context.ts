import type { Message } from '@honcho-ai/core/resources/workspaces/sessions/messages'
import type { Peer } from './peer'

export interface SummaryData {
  content: string
  message_id: number
  summary_type: string
  created_at: string
  token_count: number
}

/**
 * Represents a summary of a session's conversation.
 */
export class Summary {
  /**
   * The summary text.
   */
  readonly content: string

  /**
   * The ID of the message that this summary covers up to.
   */
  readonly messageId: number

  /**
   * The type of summary (short or long).
   */
  readonly summaryType: string

  /**
   * The timestamp of when the summary was created (ISO format).
   */
  readonly createdAt: string

  /**
   * The number of tokens in the summary text.
   */
  readonly tokenCount: number

  constructor(data: SummaryData) {
    this.content = data.content
    this.messageId = data.message_id
    this.summaryType = data.summary_type
    this.createdAt = data.created_at
    this.tokenCount = data.token_count
  }
}

/**
 * Contains both short and long summaries for a session.
 */
export class SessionSummaries {
  /**
   * The session ID.
   */
  readonly id: string

  /**
   * The short summary if available.
   */
  readonly shortSummary: Summary | null

  /**
   * The long summary if available.
   */
  readonly longSummary: Summary | null

  constructor(data: {
    id: string
    short_summary?: SummaryData | null
    long_summary?: SummaryData | null
  }) {
    this.id = data.id
    this.shortSummary = data.short_summary
      ? new Summary(data.short_summary)
      : null
    this.longSummary = data.long_summary ? new Summary(data.long_summary) : null
  }
}

/**
 * Represents the context of a session containing a curated list of messages.
 *
 * The SessionContext provides methods to convert message history into formats
 * compatible with different LLM providers while staying within token limits
 * and providing optimal conversation context.
 */
export class SessionContext {
  /**
   * ID of the session this context belongs to.
   */
  readonly sessionId: string

  /**
   * List of Message objects representing the conversation context.
   */
  readonly messages: Message[]

  /**
   * Summary of the session history prior to the message cutoff.
   */
  readonly summary: Summary | null

  /**
   * The peer representation, if context is requested from a specific perspective.
   */
  readonly peerRepresentation: string | null

  /**
   * The peer card, if context is requested from a specific perspective.
   */
  readonly peerCard: string[] | null

  /**
   * Initialize a new SessionContext.
   *
   * @param sessionId ID of the session this context belongs to
   * @param messages List of Message objects to include in the context
   * @param summary Summary of the session history prior to the message cutoff
   * @param peerRepresentation The peer representation, if context is requested from a specific perspective
   * @param peerCard The peer card, if context is requested from a specific perspective
   */
  constructor(
    sessionId: string,
    messages: Message[],
    summary: Summary | null = null,
    peerRepresentation: string | null = null,
    peerCard: string[] | null = null
  ) {
    this.sessionId = sessionId
    this.messages = messages
    this.summary = summary
    this.peerRepresentation = peerRepresentation
    this.peerCard = peerCard
  }

  /**
   * Convert the context to OpenAI-compatible message format.
   *
   * Transforms the message history and summary into the format expected by
   * OpenAI's Chat Completions API, with proper role assignments based on the
   * assistant's identity.
   *
   * @param assistant The assistant peer (Peer object or peer ID string) to use
   *                  for determining message roles. Messages from this peer will
   *                  be marked as "assistant", others as "user"
   * @returns A list of dictionaries in OpenAI format, where each dictionary contains
   *          "role" and "content" keys suitable for the OpenAI API
   */
  toOpenAI(
    assistant: string | Peer
  ): Array<{ role: string; content: string; name?: string }> {
    const assistantId = typeof assistant === 'string' ? assistant : assistant.id
    const messages = this.messages.map((message) => ({
      role: message.peer_id === assistantId ? 'assistant' : 'user',
      name: message.peer_id,
      content: message.content,
    }))

    const systemMessages: Array<{ role: string; content: string }> = []

    if (this.peerRepresentation) {
      systemMessages.push({
        role: 'system',
        content: `<peer_representation>${this.peerRepresentation}</peer_representation>`,
      })
    }

    if (this.peerCard) {
      systemMessages.push({
        role: 'system',
        content: `<peer_card>${this.peerCard}</peer_card>`,
      })
    }

    if (this.summary) {
      systemMessages.push({
        role: 'system',
        content: `<summary>${this.summary.content}</summary>`,
      })
    }

    return [...systemMessages, ...messages]
  }

  /**
   * Convert the context to Anthropic-compatible message format.
   *
   * Transforms the message history into the format expected by Anthropic's
   * Claude API, with proper role assignments based on the assistant's identity.
   *
   * @param assistant The assistant peer (Peer object or peer ID string) to use
   *                  for determining message roles. Messages from this peer will
   *                  be marked as "assistant", others as "user"
   * @returns A list of dictionaries in Anthropic format, where each dictionary contains
   *          "role" and "content" keys suitable for the Anthropic API
   *
   * Note:
   *   Future versions may implement role alternation requirements for
   *   Anthropic's API compatibility
   */
  toAnthropic(
    assistant: string | Peer
  ): Array<{ role: string; content: string }> {
    const assistantId = typeof assistant === 'string' ? assistant : assistant.id
    const messages = this.messages.map((message) =>
      message.peer_id === assistantId
        ? {
            role: 'assistant',
            content: message.content,
          }
        : {
            role: 'user',
            content: `${message.peer_id}: ${message.content}`,
          }
    )

    const systemMessages: Array<{ role: string; content: string }> = []

    if (this.peerRepresentation) {
      systemMessages.push({
        role: 'user',
        content: `<peer_representation>${this.peerRepresentation}</peer_representation>`,
      })
    }

    if (this.peerCard) {
      systemMessages.push({
        role: 'user',
        content: `<peer_card>${this.peerCard}</peer_card>`,
      })
    }

    if (this.summary) {
      systemMessages.push({
        role: 'user',
        content: `<summary>${this.summary.content}</summary>`,
      })
    }

    return [...systemMessages, ...messages]
  }

  /**
   * Return the number of messages in the context.
   */
  get length(): number {
    return this.messages.length + (this.summary ? 1 : 0)
  }

  /**
   * Return a string representation of the SessionContext.
   */
  toString(): string {
    return `SessionContext(messages=${this.messages.length}, summary=${this.summary ? 'present' : 'none'})`
  }
}
