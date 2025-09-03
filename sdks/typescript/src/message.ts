import type HonchoCore from '@honcho-ai/core'
import type { Message as CoreMessage } from '@honcho-ai/core/src/resources/workspaces/sessions/messages'
import { MessageMetadataUpdateSchema } from './validation'

/**
 * Represents a message in the Honcho system.
 *
 * Messages are scoped to sessions and provide full message data plus SDK methods.
 * Messages contain content, metadata, and connection information, and support
 * operations like metadata updates through direct method calls.
 *
 * @example
 * ```typescript
 * // Messages are typically created from API responses via fromCore
 * const message = Message.fromCore(coreMessage, client)
 *
 * // Update message metadata
 * await message.update({ status: 'processed', priority: 'high' })
 * ```
 */
export class Message {
  /**
   * Unique identifier for this message.
   */
  readonly id: string
  /**
   * The message content.
   */
  readonly content: string
  /**
   * When the message was created.
   */
  readonly createdAt: string | Date
  /**
   * ID of the peer who created the message.
   */
  readonly peer_id: string
  /**
   * ID of the session this message belongs to.
   */
  readonly session_id: string
  /**
   * Number of tokens in the message.
   */
  readonly token_count: number
  /**
   * Workspace ID for scoping operations.
   */
  readonly workspace_id: string
  /**
   * Optional metadata dictionary.
   */
  metadata: Record<string, unknown>

  /**
   * Reference to the parent Honcho client instance.
   */
  private _client: HonchoCore

  /**
   * Initialize a new Message. **Use fromCore() instead for creating from API responses.**
   *
   * @param id - Unique identifier for this message
   * @param content - The message content
   * @param createdAt - When the message was created
   * @param peer_id - ID of the peer who created the message
   * @param session_id - ID of the session this message belongs to
   * @param token_count - Number of tokens in the message
   * @param workspace_id - Workspace ID for scoping operations
   * @param client - Reference to the parent Honcho client instance
   * @param metadata - Optional metadata dictionary
   */
  constructor(
    id: string,
    content: string,
    created_at: string | Date,
    peer_id: string,
    session_id: string,
    token_count: number,
    workspace_id: string,
    client: HonchoCore,
    metadata: Record<string, unknown> = {}
  ) {
    let createdAt: string
    if (created_at instanceof Date) {
      // Check if the date is valid before calling toISOString
      if (isNaN(created_at.getTime())) {
        createdAt = new Date().toISOString() // fallback to current time
      } else {
        createdAt = created_at.toISOString()
      }
    } else {
      createdAt = created_at
    }

    this.id = id
    this.content = content
    this.createdAt = createdAt
    this.peer_id = peer_id
    this.session_id = session_id
    this.token_count = token_count
    this.workspace_id = workspace_id
    this.metadata = metadata
    this._client = client
  }

  /**
   * Create a Message from a core Message object.
   *
   * This is the primary way Message objects are created from API responses.
   * Automatically handles type conversion and client injection.
   *
   * @param coreMessage - Core message object from @honcho-ai/core
   * @param client - Reference to the parent Honcho client instance
   * @returns SDK Message object with all data and methods available
   */
  static fromCore(coreMessage: CoreMessage, client: HonchoCore): Message {
    return new Message(
      coreMessage.id,
      coreMessage.content,
      new Date(coreMessage.created_at),
      coreMessage.peer_id,
      coreMessage.session_id,
      coreMessage.token_count,
      coreMessage.workspace_id,
      client,
      coreMessage.metadata || {}
    )
  }

  /**
   * Update metadata for this message.
   *
   * Makes an API call to update the metadata associated with this message.
   * This will overwrite any existing metadata with the provided values.
   * The local metadata is also updated to reflect the changes.
   *
   * @param metadata - A dictionary of metadata to associate with the message.
   *                   Keys must be strings, values can be any JSON-serializable type
   * @returns Promise resolving to the updated Message object (this instance)
   *
   * @example
   * ```typescript
   * // Update message metadata
   * await message.update({
   *   status: 'processed',
   *   priority: 'high',
   *   tags: ['important', 'urgent']
   * })
   * ```
   */
  async update(metadata: Record<string, unknown>): Promise<Message> {
    const validatedMetadata = MessageMetadataUpdateSchema.parse(metadata)

    await this._client.workspaces.sessions.messages.update(
      this.workspace_id,
      this.session_id,
      this.id,
      {
        metadata: validatedMetadata,
      }
    )

    // Update local copy
    this.metadata = validatedMetadata
    return this
  }

  /**
   * Return a string representation of the Message.
   *
   * @returns A string representation suitable for debugging
   */
  toString(): string {
    return `Message(id='${this.id}', session_id='${this.session_id}', workspace_id='${this.workspace_id}')`
  }
}
