import type { HonchoHTTPClient } from './http/client'
import { Page } from './pagination'
import { Peer } from './peer'
import type { RepresentationOptions } from './representation'
import { SessionContext, SessionSummaries, Summary } from './session_context'
import {
  API_VERSION,
  type MessageResponse,
  type PageResponse,
  type PeerResponse,
  type QueueStatusParams,
  type QueueStatusResponse,
  type RepresentationResponse,
  type SessionContextResponse,
  type SessionResponse,
  type SessionSummariesResponse,
} from './types'
import {
  ContextParamsSchema,
  FileUploadSchema,
  FilterSchema,
  type Filters,
  GetRepresentationParamsSchema,
  LimitSchema,
  type MessageAddition,
  MessageAdditionSchema,
  type PeerAddition,
  PeerAdditionSchema,
  type PeerRemoval,
  PeerRemovalSchema,
  type QueueStatusOptions,
  SearchQuerySchema,
  SessionPeerConfigSchema,
} from './validation'

/**
 * Configuration options for a peer within a specific session.
 */
export class SessionPeerConfig {
  observe_me?: boolean | null
  observe_others?: boolean | null

  constructor(observe_me?: boolean | null, observe_others?: boolean | null) {
    const validatedConfig = SessionPeerConfigSchema.parse({
      observe_me,
      observe_others,
    })
    this.observe_me = validatedConfig.observe_me
    this.observe_others = validatedConfig.observe_others
  }
}

/**
 * Represents a session in the Honcho system.
 */
export class Session {
  readonly id: string
  readonly workspaceId: string
  private _http: HonchoHTTPClient
  private _metadata?: Record<string, unknown>
  private _configuration?: Record<string, unknown>

  get metadata(): Record<string, unknown> | undefined {
    return this._metadata
  }

  get configuration(): Record<string, unknown> | undefined {
    return this._configuration
  }

  constructor(
    id: string,
    workspaceId: string,
    http: HonchoHTTPClient,
    metadata?: Record<string, unknown>,
    configuration?: Record<string, unknown>
  ) {
    this.id = id
    this.workspaceId = workspaceId
    this._http = http
    this._metadata = metadata
    this._configuration = configuration
  }

  // ===========================================================================
  // Private API Methods
  // ===========================================================================

  private async _getOrCreate(params: {
    id: string
    metadata?: Record<string, unknown>
    configuration?: Record<string, unknown>
  }): Promise<SessionResponse> {
    return this._http.post<SessionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions`,
      { body: params }
    )
  }

  private async _update(params: {
    metadata?: Record<string, unknown>
    configuration?: Record<string, unknown>
  }): Promise<SessionResponse> {
    return this._http.put<SessionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}`,
      { body: params }
    )
  }

  private async _delete(): Promise<SessionResponse> {
    return this._http.delete<SessionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}`
    )
  }

  private async _clone(params?: {
    message_id?: string
  }): Promise<SessionResponse> {
    return this._http.post<SessionResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/clone`,
      { body: params || {} }
    )
  }

  private async _getContext(params: {
    tokens?: number
    summary?: boolean
    last_message?: string
    peer_target?: string
    peer_perspective?: string
    limit_to_session?: boolean
    search_top_k?: number
    search_max_distance?: number
    include_most_frequent?: boolean
    max_conclusions?: number
  }): Promise<SessionContextResponse> {
    return this._http.post<SessionContextResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/context`,
      { body: params }
    )
  }

  private async _getSummaries(): Promise<SessionSummariesResponse> {
    return this._http.get<SessionSummariesResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/summaries`
    )
  }

  private async _search(params: {
    query: string
    filters?: Record<string, unknown>
    limit?: number
  }): Promise<MessageResponse[]> {
    return this._http.post<MessageResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/search`,
      { body: params }
    )
  }

  private async _addPeers(
    peers: Record<
      string,
      { observe_me?: boolean | null; observe_others?: boolean | null }
    >
  ): Promise<void> {
    await this._http.post(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers/add`,
      { body: { peers } }
    )
  }

  private async _setPeers(
    peers: Record<
      string,
      { observe_me?: boolean | null; observe_others?: boolean | null }
    >
  ): Promise<void> {
    await this._http.post(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers/set`,
      { body: { peers } }
    )
  }

  private async _removePeers(peerIds: string[]): Promise<void> {
    await this._http.post(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers/remove`,
      { body: { peers: peerIds } }
    )
  }

  private async _listPeers(): Promise<PageResponse<PeerResponse>> {
    return this._http.get<PageResponse<PeerResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers`
    )
  }

  private async _getPeerConfig(peerId: string): Promise<SessionPeerConfig> {
    return this._http.get<SessionPeerConfig>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers/${peerId}/config`
    )
  }

  private async _setPeerConfig(
    peerId: string,
    config: { observe_me?: boolean | null; observe_others?: boolean | null }
  ): Promise<void> {
    await this._http.put(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/peers/${peerId}/config`,
      { body: config }
    )
  }

  private async _createMessages(params: {
    messages: Array<{
      peer_id: string
      content: string
      metadata?: Record<string, unknown>
      configuration?: Record<string, unknown>
      created_at?: string
    }>
  }): Promise<MessageResponse[]> {
    return this._http.post<MessageResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/messages`,
      { body: params }
    )
  }

  private async _listMessages(params?: {
    filters?: Record<string, unknown>
    page?: number
    size?: number
  }): Promise<PageResponse<MessageResponse>> {
    return this._http.post<PageResponse<MessageResponse>>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/messages/list`,
      {
        body: { filters: params?.filters },
        query: { page: params?.page, size: params?.size },
      }
    )
  }

  private async _uploadFile(formData: FormData): Promise<MessageResponse[]> {
    return this._http.upload<MessageResponse[]>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/sessions/${this.id}/messages/upload`,
      formData
    )
  }

  private async _getQueueStatus(
    params?: QueueStatusParams
  ): Promise<QueueStatusResponse> {
    const query: Record<string, string | number | boolean | undefined> = {}
    if (params?.observer_id) query.observer_id = params.observer_id
    if (params?.sender_id) query.sender_id = params.sender_id
    if (params?.session_id) query.session_id = params.session_id

    return this._http.get<QueueStatusResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/queue/status`,
      { query }
    )
  }

  private async _getRepresentation(
    peerId: string,
    params: {
      session_id?: string
      target?: string
      search_query?: string
      search_top_k?: number
      search_max_distance?: number
      include_most_frequent?: boolean
      max_conclusions?: number
    }
  ): Promise<RepresentationResponse> {
    return this._http.post<RepresentationResponse>(
      `/${API_VERSION}/workspaces/${this.workspaceId}/peers/${peerId}/representation`,
      { body: params }
    )
  }

  // ===========================================================================
  // Public Methods
  // ===========================================================================

  async addPeers(peers: PeerAddition): Promise<void> {
    const validatedPeers = PeerAdditionSchema.parse(peers)
    const peerDict: Record<string, SessionPeerConfig> = {}
    const peersArray = Array.isArray(validatedPeers)
      ? validatedPeers
      : [validatedPeers]

    for (const peer of peersArray) {
      if (typeof peer === 'string') {
        peerDict[peer] = {}
      } else if (Array.isArray(peer)) {
        const peerId = typeof peer[0] === 'string' ? peer[0] : peer[0].id
        peerDict[peerId] = peer[1]
      } else if (typeof peer === 'object' && 'id' in peer) {
        peerDict[peer.id] = {}
      } else {
        throw new Error(`Invalid peer type: ${typeof peer}`)
      }
    }

    await this._addPeers(peerDict)
  }

  async setPeers(peers: PeerAddition): Promise<void> {
    const validatedPeers = PeerAdditionSchema.parse(peers)
    const peerDict: Record<string, SessionPeerConfig> = {}
    const peersArray = Array.isArray(validatedPeers)
      ? validatedPeers
      : [validatedPeers]

    for (const peer of peersArray) {
      if (typeof peer === 'string') {
        peerDict[peer] = {}
      } else if (Array.isArray(peer)) {
        const peerId = typeof peer[0] === 'string' ? peer[0] : peer[0].id
        peerDict[peerId] = peer[1]
      } else if (typeof peer === 'object' && 'id' in peer) {
        peerDict[peer.id] = {}
      } else {
        throw new Error(`Invalid peer type: ${typeof peer}`)
      }
    }

    await this._setPeers(peerDict)
  }

  async removePeers(peers: PeerRemoval): Promise<void> {
    const validatedPeers = PeerRemovalSchema.parse(peers)
    const peerIds = Array.isArray(validatedPeers)
      ? validatedPeers.map((p) => (typeof p === 'string' ? p : p.id))
      : [
          typeof validatedPeers === 'string'
            ? validatedPeers
            : validatedPeers.id,
        ]
    await this._removePeers(peerIds)
  }

  async getPeers(): Promise<Peer[]> {
    const peersPage = await this._listPeers()
    return peersPage.items.map(
      (peer) => new Peer(peer.id, this.workspaceId, this._http)
    )
  }

  async getPeerConfig(peer: string | Peer): Promise<SessionPeerConfig> {
    const peerId = typeof peer === 'string' ? peer : peer.id
    return await this._getPeerConfig(peerId)
  }

  async setPeerConfig(
    peer: string | Peer,
    config: SessionPeerConfig
  ): Promise<void> {
    const peerId = typeof peer === 'string' ? peer : peer.id
    const validatedConfig = SessionPeerConfigSchema.parse(config)
    await this._setPeerConfig(peerId, {
      observe_others: validatedConfig.observe_others,
      observe_me: validatedConfig.observe_me,
    })
  }

  async addMessages(messages: MessageAddition): Promise<MessageResponse[]> {
    const validatedMessages = MessageAdditionSchema.parse(messages)
    const messagesList = Array.isArray(validatedMessages)
      ? validatedMessages
      : [validatedMessages]
    // Transform null values to undefined for API compatibility
    const transformedMessages = messagesList.map((msg) => ({
      peer_id: msg.peer_id,
      content: msg.content,
      metadata: msg.metadata,
      configuration: msg.configuration ?? undefined,
      created_at: msg.created_at ?? undefined,
    }))
    return await this._createMessages({ messages: transformedMessages })
  }

  async getMessages(filters?: Filters): Promise<Page<MessageResponse>> {
    const validatedFilter = filters ? FilterSchema.parse(filters) : undefined
    const messagesPage = await this._listMessages({ filters: validatedFilter })

    const fetchNextPage = async (
      page: number,
      size: number
    ): Promise<PageResponse<MessageResponse>> => {
      return this._listMessages({ filters: validatedFilter, page, size })
    }

    return new Page(messagesPage, undefined, fetchNextPage)
  }

  async getMetadata(): Promise<Record<string, unknown>> {
    const session = await this._getOrCreate({ id: this.id })
    this._metadata = session.metadata || {}
    return this._metadata
  }

  async setMetadata(metadata: Record<string, unknown>): Promise<void> {
    await this._update({ metadata })
    this._metadata = metadata
  }

  async getConfig(): Promise<Record<string, unknown>> {
    const session = await this._getOrCreate({ id: this.id })
    this._configuration = session.configuration || {}
    return this._configuration
  }

  async setConfig(configuration: Record<string, unknown>): Promise<void> {
    await this._update({ configuration })
    this._configuration = configuration
  }

  async refresh(): Promise<void> {
    const session = await this._getOrCreate({ id: this.id })
    this._metadata = session.metadata || {}
    this._configuration = session.configuration || {}
  }

  async delete(): Promise<void> {
    await this._delete()
  }

  async clone(messageId?: string): Promise<Session> {
    const clonedSessionData = await this._clone(
      messageId ? { message_id: messageId } : undefined
    )

    return new Session(
      clonedSessionData.id,
      this.workspaceId,
      this._http,
      clonedSessionData.metadata ?? undefined,
      clonedSessionData.configuration ?? undefined
    )
  }

  async getContext(
    summary?: boolean,
    tokens?: number,
    peerTarget?: string | Peer,
    lastUserMessage?: string | MessageResponse,
    peerPerspective?: string | Peer,
    representationOptions?: RepresentationOptions
  ): Promise<SessionContext>
  async getContext(options?: {
    summary?: boolean
    tokens?: number
    peerTarget?: string | Peer
    lastUserMessage?: string | MessageResponse
    peerPerspective?: string | Peer
    limitToSession?: boolean
    representationOptions?: RepresentationOptions
  }): Promise<SessionContext>
  async getContext(
    summaryOrOptions?:
      | boolean
      | {
          summary?: boolean
          tokens?: number
          peerTarget?: string | Peer
          lastUserMessage?: string | MessageResponse
          peerPerspective?: string | Peer
          limitToSession?: boolean
          representationOptions?: RepresentationOptions
        },
    tokens?: number,
    peerTarget?: string | Peer,
    lastUserMessage?: string | MessageResponse,
    peerPerspective?: string | Peer,
    representationOptions?: RepresentationOptions
  ): Promise<SessionContext> {
    let options: {
      summary?: boolean
      tokens?: number
      peerTarget?: string
      lastUserMessage?: string
      peerPerspective?: string
      limitToSession?: boolean
      representationOptions?: RepresentationOptions
    }

    if (
      typeof summaryOrOptions === 'boolean' ||
      // biome-ignore lint/complexity/noArguments: Need to detect which overload pattern is being used
      (summaryOrOptions === undefined && arguments.length > 1)
    ) {
      options = {
        summary: summaryOrOptions as boolean | undefined,
        tokens,
        peerTarget: typeof peerTarget === 'object' ? peerTarget.id : peerTarget,
        lastUserMessage:
          typeof lastUserMessage === 'string'
            ? lastUserMessage
            : lastUserMessage?.id,
        peerPerspective:
          typeof peerPerspective === 'object'
            ? peerPerspective.id
            : peerPerspective,
        representationOptions,
      }
    } else {
      options = (summaryOrOptions as typeof options) || {}
    }

    const contextParams = ContextParamsSchema.parse({
      summary: options.summary,
      tokens: options.tokens,
      peerTarget: options.peerTarget,
      lastUserMessage: options.lastUserMessage,
      peerPerspective: options.peerPerspective,
      limitToSession: options.limitToSession,
      representationOptions: options.representationOptions,
    })

    const lastMessageId =
      typeof contextParams.lastUserMessage === 'string'
        ? contextParams.lastUserMessage
        : contextParams.lastUserMessage?.id

    const context = await this._getContext({
      tokens: contextParams.tokens,
      summary: contextParams.summary,
      last_message: lastMessageId,
      peer_target: contextParams.peerTarget,
      peer_perspective: contextParams.peerPerspective,
      limit_to_session: contextParams.limitToSession,
      search_top_k: contextParams.representationOptions?.searchTopK,
      search_max_distance:
        contextParams.representationOptions?.searchMaxDistance,
      include_most_frequent:
        contextParams.representationOptions?.includeMostFrequent,
      max_conclusions: contextParams.representationOptions?.maxConclusions,
    })

    const summary = context.summary ? new Summary(context.summary) : null
    return new SessionContext(
      this.id,
      context.messages,
      summary,
      context.peer_representation
        ? JSON.stringify(context.peer_representation)
        : null,
      context.peer_card ?? null
    )
  }

  async getSummaries(): Promise<SessionSummaries> {
    const data = await this._getSummaries()
    return new SessionSummaries(data)
  }

  async search(
    query: string,
    options?: { filters?: Filters; limit?: number }
  ): Promise<MessageResponse[]> {
    const validatedQuery = SearchQuerySchema.parse(query)
    const validatedFilters = options?.filters
      ? FilterSchema.parse(options.filters)
      : undefined
    const validatedLimit = options?.limit
      ? LimitSchema.parse(options.limit)
      : undefined
    return await this._search({
      query: validatedQuery,
      filters: validatedFilters,
      limit: validatedLimit,
    })
  }

  async getQueueStatus(
    options?: Omit<
      QueueStatusOptions,
      'sessionId' | 'observerId' | 'senderId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
    }
  ): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<
      string,
      {
        total_work_units: number
        completed_work_units: number
        in_progress_work_units: number
        pending_work_units: number
      }
    >
  }> {
    const resolvedObserverId = options?.observer
      ? typeof options.observer === 'string'
        ? options.observer
        : options.observer.id
      : undefined
    const resolvedSenderId = options?.sender
      ? typeof options.sender === 'string'
        ? options.sender
        : options.sender.id
      : undefined

    const queryParams: QueueStatusParams = { session_id: this.id }
    if (resolvedObserverId) queryParams.observer_id = resolvedObserverId
    if (resolvedSenderId) queryParams.sender_id = resolvedSenderId

    const status = await this._getQueueStatus(queryParams)

    return {
      totalWorkUnits: status.total_work_units,
      completedWorkUnits: status.completed_work_units,
      inProgressWorkUnits: status.in_progress_work_units,
      pendingWorkUnits: status.pending_work_units,
      sessions: status.sessions || undefined,
    }
  }

  async pollQueueStatus(
    options?: Omit<
      QueueStatusOptions,
      'sessionId' | 'observerId' | 'senderId'
    > & {
      observer?: string | Peer
      sender?: string | Peer
    }
  ): Promise<{
    totalWorkUnits: number
    completedWorkUnits: number
    inProgressWorkUnits: number
    pendingWorkUnits: number
    sessions?: Record<
      string,
      {
        total_work_units: number
        completed_work_units: number
        in_progress_work_units: number
        pending_work_units: number
      }
    >
  }> {
    const timeoutMs = options?.timeoutMs ?? 300000
    const startTime = Date.now()

    while (true) {
      const status = await this.getQueueStatus(options)
      if (status.pendingWorkUnits === 0 && status.inProgressWorkUnits === 0) {
        return status
      }

      const elapsedTime = Date.now() - startTime
      if (elapsedTime >= timeoutMs) {
        throw new Error(
          `Polling timeout exceeded after ${timeoutMs}ms. ` +
            `Current status: ${status.pendingWorkUnits} pending, ${status.inProgressWorkUnits} in progress work units.`
        )
      }

      const totalWorkUnits =
        status.pendingWorkUnits + status.inProgressWorkUnits
      const sleepMs = Math.max(1000, totalWorkUnits * 1000)
      const remainingTime = timeoutMs - elapsedTime
      const actualSleepMs = Math.min(sleepMs, remainingTime)

      if (actualSleepMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, actualSleepMs))
      }
    }
  }

  async uploadFile(
    file:
      | File
      | Blob
      | {
          filename: string
          content: Buffer | Uint8Array
          content_type: string
        },
    peer: string | Peer,
    options?: {
      metadata?: Record<string, unknown>
      configuration?: Record<string, unknown>
      created_at?: string | Date
    }
  ): Promise<MessageResponse[]> {
    const createdAt =
      options?.created_at instanceof Date
        ? options.created_at.toISOString()
        : options?.created_at

    const resolvedPeerId = typeof peer === 'string' ? peer : peer.id

    const uploadParams = FileUploadSchema.parse({
      file,
      peer: resolvedPeerId,
      metadata: options?.metadata,
      configuration: options?.configuration,
      created_at: createdAt,
    })

    const formData = new FormData()

    if (file instanceof File || file instanceof Blob) {
      formData.append('file', file)
    } else {
      // Convert to Uint8Array for Blob compatibility
      const content = new Uint8Array(file.content)
      const blob = new Blob([content], { type: file.content_type })
      formData.append('file', blob, file.filename)
    }

    formData.append('peer_id', resolvedPeerId)
    if (uploadParams.metadata !== undefined && uploadParams.metadata !== null) {
      formData.append('metadata', JSON.stringify(uploadParams.metadata))
    }
    if (
      uploadParams.configuration !== undefined &&
      uploadParams.configuration !== null
    ) {
      formData.append(
        'configuration',
        JSON.stringify(uploadParams.configuration)
      )
    }
    if (
      uploadParams.created_at !== undefined &&
      uploadParams.created_at !== null
    ) {
      formData.append('created_at', uploadParams.created_at)
    }

    return await this._uploadFile(formData)
  }

  async getRepresentation(
    peer: string | Peer,
    target?: string | Peer,
    options?: {
      searchQuery?: string
      searchTopK?: number
      searchMaxDistance?: number
      includeMostFrequent?: boolean
      maxConclusions?: number
    }
  ): Promise<string> {
    const getRepresentationParams = GetRepresentationParamsSchema.parse({
      peer,
      target,
      options,
    })
    const peerId =
      typeof getRepresentationParams.peer === 'string'
        ? getRepresentationParams.peer
        : getRepresentationParams.peer.id
    const targetId = getRepresentationParams.target
      ? typeof getRepresentationParams.target === 'string'
        ? getRepresentationParams.target
        : getRepresentationParams.target.id
      : undefined

    const response = await this._getRepresentation(peerId, {
      session_id: this.id,
      target: targetId,
      search_query: getRepresentationParams.options?.searchQuery,
      search_top_k: getRepresentationParams.options?.searchTopK,
      search_max_distance: getRepresentationParams.options?.searchMaxDistance,
      include_most_frequent:
        getRepresentationParams.options?.includeMostFrequent,
      max_conclusions: getRepresentationParams.options?.maxConclusions,
    })
    return response.representation
  }

  toString(): string {
    return `Session(id='${this.id}')`
  }
}
