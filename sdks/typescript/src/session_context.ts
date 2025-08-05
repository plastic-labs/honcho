import type { Peer } from './peer'

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
  readonly messages: any[]

  /**
   * Summary of the session history prior to the message cutoff.
   */
  readonly summary: string

  /**
   * Initialize a new SessionContext.
   *
   * @param sessionId ID of the session this context belongs to
   * @param messages List of Message objects to include in the context
   * @param summary Summary of the session history prior to the message cutoff
   */
  constructor(sessionId: string, messages: any[], summary: string = '') {
    this.sessionId = sessionId
    this.messages = messages
    this.summary = summary || ''
  }

  /**
   * Convert the context to OpenAI-compatible message format.
   *
   * Transforms the message history into the format expected by OpenAI's
   * Chat Completions API, with proper role assignments based on the
   * assistant's identity.
   *
   * @param assistant The assistant peer (Peer object or peer ID string) to use
   *                  for determining message roles. Messages from this peer will
   *                  be marked as "assistant", others as "user"
   * @returns A list of dictionaries in OpenAI format, where each dictionary contains
   *          "role" and "content" keys suitable for the OpenAI API
   */
  toOpenAI(assistant: string | Peer): Array<{ role: string; content: string }> {
    const assistantId = typeof assistant === 'string' ? assistant : assistant.id
    return this.messages.map((message) => ({
      role: message.peer_name === assistantId ? 'assistant' : 'user',
      content: message.content,
    }))
  }

  /**
   * Convert the context to Anthropic-compatible message format.
   *
   * Transforms the message history into the format expected by Anthropic's
   * Claude API.
   *
   * @param assistant The assistant peer (Peer object or peer ID string) to use
   *                  for determining message roles. Messages from this peer will
   *                  be marked as "assistant", others as "user"
   * @returns A list of dictionaries in Anthropic format, where each dictionary contains
   *          "role" and "content" keys suitable for the Anthropic API
   */
  toAnthropic(
    assistant: string | Peer
  ): Array<{ role: string; content: string }> {
    const assistantId = typeof assistant === 'string' ? assistant : assistant.id
    return this.messages.map((message) => ({
      role: message.peer_name === assistantId ? 'assistant' : 'user',
      content: message.content,
    }))
  }

  /**
   * Return the number of messages in the context.
   */
  get length(): number {
    return this.messages.length
  }

  /**
   * Return a string representation of the SessionContext.
   */
  toString(): string {
    return `SessionContext(messages=${this.messages.length})`
  }
}
