import { z } from 'zod'

/**
 * Validation schemas for the Honcho TypeScript SDK.
 *
 * These schemas ensure type safety and runtime validation for all inputs
 * to the SDK, providing clear error messages when validation fails.
 */

/**
 * Schema for workspace ID validation.
 */
export const WorkspaceIdSchema = z
  .string()
  .min(1, 'Workspace ID must be a non-empty string')
  .regex(
    /^[a-zA-Z0-9_-]+$/,
    'Workspace ID may only contain letters, numbers, underscores, and hyphens'
  )
  .max(512, 'Workspace ID can be at most 512 characters')

/**
 * Schema for Honcho client configuration options.
 */
export const HonchoConfigSchema = z
  .object({
    apiKey: z.string().optional(),
    environment: z.enum(['local', 'production']).optional(),
    baseURL: z.url('Base URL must be a valid URL').optional(),
    workspaceId: WorkspaceIdSchema.optional(),
    timeout: z
      .number()
      .positive('Timeout must be a positive number')
      .optional(),
    maxRetries: z
      .number()
      .int()
      .min(0, 'Max retries must be a non-negative integer')
      .max(3, 'Max retries must be at most 3')
      .optional(),
    defaultHeaders: z.record(z.string(), z.string()).optional(),
    defaultQuery: z
      .record(z.string(), z.union([z.string(), z.number(), z.boolean()]))
      .optional(),
  })
  .strict()

/**
 * Schema for peer metadata.
 */
export const PeerMetadataSchema = z.record(z.string(), z.unknown())

/**
 * Schema for peer configuration.
 */
export const PeerConfigSchema = z
  .object({
    observeMe: z.boolean().nullable().optional(),
  })
  .strict()

/**
 * Schema for peer ID validation.
 */
export const PeerIdSchema = z
  .string()
  .min(1, 'Peer ID must be a non-empty string')
  .regex(
    /^[a-zA-Z0-9_-]+$/,
    'Peer ID may only contain letters, numbers, underscores, and hyphens'
  )
  .max(512, 'Peer ID can be at most 512 characters')

/**
 * Strict helper: peer ID as object.
 */
const PeerIdObjectSchema = z.object({ id: PeerIdSchema })

/**
 * Schema for session metadata.
 */
export const SessionMetadataSchema = z.record(z.string(), z.unknown())

// =============================================================================
// Configuration Schemas (typed)
// =============================================================================

/**
 * Schema for reasoning configuration.
 * Used in workspace, session, and message configuration.
 */
export const ReasoningConfigSchema = z
  .object({
    enabled: z.boolean().nullable().optional(),
    customInstructions: z.string().nullable().optional(),
  })
  .strict()

/**
 * Schema for peer card configuration.
 * Used in workspace and session configuration.
 */
export const PeerCardConfigSchema = z
  .object({
    use: z.boolean().nullable().optional(),
    create: z.boolean().nullable().optional(),
  })
  .strict()

/**
 * Schema for summary configuration.
 * Used in workspace and session configuration.
 */
export const SummaryConfigSchema = z
  .object({
    enabled: z.boolean().nullable().optional(),
    messagesPerShortSummary: z.number().int().min(10).nullable().optional(),
    messagesPerLongSummary: z.number().int().min(20).nullable().optional(),
  })
  .strict()

/**
 * Schema for dream configuration.
 * Used in workspace and session configuration.
 */
export const DreamConfigSchema = z
  .object({
    enabled: z.boolean().nullable().optional(),
  })
  .strict()

/**
 * Schema for session configuration.
 * Includes reasoning, peer card, summary, and dream settings.
 */
export const SessionConfigSchema = z
  .object({
    reasoning: ReasoningConfigSchema.nullable().optional(),
    peerCard: PeerCardConfigSchema.nullable().optional(),
    summary: SummaryConfigSchema.nullable().optional(),
    dream: DreamConfigSchema.nullable().optional(),
  })
  .strict()

/**
 * Schema for session ID validation.
 */
export const SessionIdSchema = z
  .string()
  .min(1, 'Session ID must be a non-empty string')
  .regex(
    /^[a-zA-Z0-9_-]+$/,
    'Session ID may only contain letters, numbers, underscores, and hyphens'
  )
  .max(512, 'Session ID can be at most 512 characters')

/**
 * Strict helper: session ID as object.
 */
const SessionIdObjectSchema = z.object({ id: SessionIdSchema })

/**
 * Reserved peer-name prefix the server uses to store a scope.
 */
const SCOPE_PEER_PREFIX = 'scope.'

/**
 * Scope IDs are stored as peer names with the reserved prefix prepended, so
 * they must leave room for it within the 512-character peer name limit.
 */
const SCOPE_ID_MAX_LENGTH = 512 - SCOPE_PEER_PREFIX.length

/**
 * The scope ID rules, as a plain function rather than only a schema.
 *
 * Zod reports a failing union as a single `invalid_union` / "Invalid input"
 * issue and buries the branch errors, so a schema alone cannot carry these
 * messages out of `ScopeOptionSchema`. Keeping the rules callable lets both the
 * bare schema and the union surface the same specific message.
 *
 * @returns The problems found, or an empty array when the ID is valid.
 */
function scopeIdIssues(value: string): string[] {
  if (value.length < 1) {
    return ['Scope ID must be a non-empty string']
  }
  if (value.length > SCOPE_ID_MAX_LENGTH) {
    return [`Scope ID can be at most ${SCOPE_ID_MAX_LENGTH} characters`]
  }
  // Checked before the charset: the reserved prefix contains '.', which is
  // itself outside the charset, so a charset-first check would report the
  // charset instead of the real mistake for a double-prefixed name.
  if (value.startsWith(SCOPE_PEER_PREFIX)) {
    return [
      `Scope ID must not start with the reserved prefix '${SCOPE_PEER_PREFIX}' (scope IDs are unprefixed)`,
    ]
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(value)) {
    return [
      'Scope ID may only contain letters, numbers, underscores, and hyphens',
    ]
  }
  return []
}

/**
 * Add every scope ID problem in `values` as a top-level issue.
 */
function addScopeIdIssues(values: string[], ctx: z.RefinementCtx): void {
  for (const value of values) {
    for (const message of scopeIdIssues(value)) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message })
    }
  }
}

/**
 * Schema for scope ID validation.
 *
 * Scope IDs are unprefixed — the `scope.` prefix is a server-side storage
 * detail and never appears on the wire.
 */
export const ScopeIdSchema = z.string().superRefine((val, ctx) => {
  addScopeIdIssues([val], ctx)
})

/**
 * Shape-only branch for the `scope` option: an ID string, or an object carrying
 * one (so a `Scope` instance is accepted). The ID itself is validated after the
 * union resolves — see `ScopeOptionSchema`.
 */
const ScopeIdLikeSchema = z.union([z.string(), z.object({ id: z.string() })])

/**
 * Schema for the `scope` read option: one scope, or a bounded list of them.
 *
 * A single scope reads that scope's own view. A list restricts recall to the
 * union of the scopes' member sessions. An empty list is rejected rather than
 * resolved to an empty allowlist, which would silently recall nothing.
 *
 * The union discriminates shape only; IDs and list bounds are checked after the
 * transform so their messages are not swallowed as `invalid_union`.
 */
export const ScopeOptionSchema = z
  .union([ScopeIdLikeSchema, z.array(ScopeIdLikeSchema)])
  .transform((val) =>
    Array.isArray(val)
      ? val.map((entry) => (typeof entry === 'string' ? entry : entry.id))
      : typeof val === 'string'
        ? val
        : val.id
  )
  .superRefine((resolved, ctx) => {
    if (Array.isArray(resolved)) {
      if (resolved.length === 0) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'scope must name at least one scope',
        })
      }
      if (resolved.length > 100) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'scope can name at most 100 scopes',
        })
      }
    }
    addScopeIdIssues(Array.isArray(resolved) ? resolved : [resolved], ctx)
  })

/**
 * Schema for a scope membership change: the sessions to add to a scope.
 *
 * Capped at 100 to match the server rather than silently chunking, so a
 * rejected batch is the batch the caller passed.
 */
export const ScopeSessionsSchema = z
  .array(SessionIdSchema)
  .min(1, 'At least one session must be given')
  .max(100, 'At most 100 sessions can be added per call')

/**
 * Schema for the `scopes` option on session creation: the scopes a new session
 * should join.
 */
export const SessionScopesSchema = z
  .array(ScopeIdSchema)
  .min(1, 'scopes must name at least one scope')
  .max(100, 'scopes can name at most 100 scopes')

/**
 * Schema for the `sessions` allowlist option — sugar for the wire-level
 * `filters: { session_id: [...] }`.
 *
 * Capped at 1,000 entries to match the server. An empty list is rejected: the
 * server treats an empty allowlist as fail-closed (recalls nothing), which is
 * never what a caller passing `sessions: []` intends.
 */
export const SessionAllowlistSchema = z
  .array(z.union([SessionIdSchema, SessionIdObjectSchema]))
  .min(1, 'sessions must name at least one session')
  .max(1000, 'sessions can name at most 1000 sessions')
  .transform((vals) => vals.map((v) => (typeof v === 'string' ? v : v.id)))

/**
 * Schema for session peer configuration.
 */
export const SessionPeerConfigSchema = z
  .object({
    observeMe: z.boolean().nullable().optional(),
    observeOthers: z.boolean().nullable().optional(),
  })
  .strict()

/**
 * Schema for message content.
 */
export const MessageContentSchema = z
  .string()
  .refine(
    (content: string) => content === '' || content.trim().length > 0,
    'Message content cannot be only whitespace'
  )

/**
 * Schema for message metadata.
 */
export const MessageMetadataSchema = z
  .record(z.string(), z.unknown())
  .optional()

/**
 * Schema for message configuration.
 * Only includes reasoning settings.
 */
export const MessageConfigurationSchema = z
  .object({
    reasoning: ReasoningConfigSchema.nullable().optional(),
  })
  .strict()
  .nullable()
  .optional()

/**
 * Schema for message input.
 */
export const MessageInputSchema = z
  .object({
    peerId: PeerIdSchema,
    content: MessageContentSchema,
    metadata: MessageMetadataSchema,
    configuration: MessageConfigurationSchema,
    createdAt: z.string().nullable().optional(),
  })
  .strict()

/**
 * Schema for search query validation.
 */
export const SearchQuerySchema = z
  .string()
  .min(1, 'Search query must be a non-empty string')
  .refine(
    (query: string) => query.trim().length > 0,
    'Search query cannot be only whitespace'
  )

/**
 * Schema for content-like search query objects.
 * Accepts SDK Message instances and other objects with a valid content field.
 */
export const SearchQueryObjectSchema = z
  .object({
    content: SearchQuerySchema,
  })
  .passthrough()

/**
 * Schema for search query inputs that can be normalized to a string.
 */
export const SearchQueryLikeSchema = z.union([
  SearchQuerySchema,
  SearchQueryObjectSchema,
])

/**
 * Normalize a supported search query input to plain text.
 */
export function normalizeSearchQuery(searchQuery: unknown): string | undefined {
  if (searchQuery === undefined) {
    return undefined
  }

  const validatedSearchQuery = SearchQueryLikeSchema.parse(searchQuery)
  return typeof validatedSearchQuery === 'string'
    ? validatedSearchQuery
    : validatedSearchQuery.content
}

/**
 * Schema for filter objects.
 */
export const FilterSchema = z.record(z.string(), z.unknown()).optional()

/**
 * Normalize list-method input so legacy raw filters and the new options object
 * shape are both accepted.
 *
 * Discriminates on the `filters` key: if the input has a `filters` property or
 * any of the pagination-only keys (`page`, `size`, `reverse`) it is treated as
 * the new options object. Otherwise it is treated as a legacy raw filter.
 */
export function normalizeListOptions<T extends { filters?: Filters }>(
  input: Filters | T | undefined,
  optionKeys: string[]
): T {
  if (input === undefined) {
    return {} as T
  }

  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    return { filters: input as Filters } as T
  }

  // Pagination-only keys can never appear in a raw filter object
  const paginationKeys = optionKeys.filter((k) => k !== 'filters')
  const hasFiltersKey = 'filters' in input
  const hasPaginationKey = paginationKeys.some((key) => key in input)

  if (hasFiltersKey || hasPaginationKey) {
    return input as T
  }

  return { filters: input as Filters } as T
}

/**
 * Translate validated `scope` / `sessions` options into their wire fields.
 *
 * `sessions` is sugar: it goes out as the constrained
 * `filters: { session_id: [...] }` body the recall endpoints accept, never as a
 * field of its own — the server rejects unknown keys with a 422. Shared by chat,
 * chatStream, and representation so the three cannot drift apart.
 *
 * Purely a translation; the schemas have already rejected the invalid
 * combinations by the time this runs.
 */
export function scopeRecallFields(options: {
  scope?: string | string[]
  sessions?: string[]
}): { scope?: string | string[]; filters?: Record<string, unknown> } {
  return {
    scope: options.scope,
    filters: options.sessions ? { session_id: options.sessions } : undefined,
  }
}

/**
 * Add issues for the `scope` exclusions the server enforces with a 422.
 *
 * A scope already determines what a query can see, so combining it with a
 * session allowlist or a single session is a contradiction rather than a
 * narrowing. Shared by the chat, representation, and context schemas so the
 * three surfaces cannot drift apart.
 */
function scopeExclusivityIssues(
  data: { scope?: unknown; sessions?: unknown; session?: unknown },
  ctx: z.RefinementCtx
): void {
  if (data.scope === undefined) {
    return
  }
  if (data.sessions !== undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'scope and sessions are mutually exclusive',
      path: ['sessions'],
    })
  }
  if (data.session !== undefined) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'scope and session are mutually exclusive',
      path: ['session'],
    })
  }
}

/**
 * Schema for chat query parameters.
 */
export const ChatQuerySchema = z
  .object({
    query: SearchQuerySchema,
    target: z
      .union([PeerIdSchema, PeerIdObjectSchema])
      .optional()
      .transform((val) =>
        val ? (typeof val === 'string' ? val : val.id) : undefined
      ),
    session: z
      .union([SessionIdSchema, SessionIdObjectSchema])
      .optional()
      .transform((val) =>
        val ? (typeof val === 'string' ? val : val.id) : undefined
      ),
    scope: ScopeOptionSchema.optional(),
    sessions: SessionAllowlistSchema.optional(),
    reasoningLevel: z
      .enum(['minimal', 'low', 'medium', 'high', 'max'])
      .optional(),
    // A Zod schema (checked first — it is itself an object) or a raw JSON
    // Schema object describing the desired response structure.
    responseFormat: z
      .union([z.instanceof(z.ZodType), z.record(z.string(), z.unknown())])
      .optional(),
  })
  .strict()
  .superRefine(scopeExclusivityIssues)

/**
 * Schema for representation options.
 */
export const RepresentationOptionsSchema = z
  .object({
    searchQuery: SearchQueryLikeSchema.optional(),
    searchTopK: z
      .number()
      .int()
      .min(1, 'searchTopK must be at least 1')
      .max(100, 'searchTopK must be at most 100')
      .optional(),
    searchMaxDistance: z
      .number()
      .min(0.0, 'searchMaxDistance must be at least 0.0')
      .max(1.0, 'searchMaxDistance must be at most 1.0')
      .optional(),
    includeMostFrequent: z.boolean().optional(),
    maxConclusions: z
      .number()
      .int()
      .min(1, 'maxConclusions must be at least 1')
      .max(100, 'maxConclusions must be at most 100')
      .optional(),
  })
  .strict()

/**
 * Schema for context retrieval parameters.
 */
export const ContextParamsSchema = z
  .object({
    summary: z.boolean().optional(),
    tokens: z.int('Token limit must be an integer').optional(),
    peerTarget: PeerIdSchema.optional(),
    peerPerspective: PeerIdSchema.optional(),
    // Only a single scope is accepted here: the context route uses a scope as
    // the *perspective source* for the target's representation and card, which
    // is one observer. A list of scopes has no meaning for that.
    scope: ScopeIdSchema.optional(),
    sessions: SessionAllowlistSchema.optional(),
    limitToSession: z.boolean().optional(),
    representationOptions: RepresentationOptionsSchema.optional(),
  })
  .strict()
  .superRefine((data, ctx) => {
    if (data.sessions && !data.peerTarget) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'peerTarget is required when sessions is provided',
        path: ['sessions'],
      })
    }

    if (data.sessions && data.scope) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'sessions and scope are mutually exclusive',
        path: ['sessions'],
      })
    }

    if (data.sessions && data.limitToSession) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'sessions and limitToSession are mutually exclusive',
        path: ['sessions'],
      })
    }

    if (data.representationOptions?.searchQuery && !data.peerTarget) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'peerTarget is required when searchQuery is provided',
        path: ['representationOptions', 'searchQuery'],
      })
    }

    if (data.peerPerspective && !data.peerTarget) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'peerTarget is required when peerPerspective is provided',
        path: ['peerPerspective'],
      })
    }

    if (data.scope && !data.peerTarget) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'peerTarget is required when scope is provided',
        path: ['scope'],
      })
    }

    if (data.scope && data.peerPerspective) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'scope and peerPerspective are mutually exclusive',
        path: ['scope'],
      })
    }
  })

/**
 * Schema for deriver status options.
 */
export const QueueStatusOptionsSchema = z
  .object({
    observer: z.union([PeerIdSchema, PeerIdObjectSchema]).optional(),
    sender: z.union([PeerIdSchema, PeerIdObjectSchema]).optional(),
    session: z.union([SessionIdSchema, SessionIdObjectSchema]).optional(),
    timeout: z
      .number()
      .positive('Timeout must be a positive number')
      .optional(),
  })
  .strict()

/**
 * Schema for file upload parameters.
 * Supports Blob/File objects and custom uploadable objects with binary content.
 */
export const FileUploadSchema = z
  .object({
    file: z.union([
      // Browser/File API objects
      z.instanceof(Blob),
      // Custom uploadable object with filename, content, and content_type
      z
        .object({
          filename: z.string().min(1, 'Filename must be a non-empty string'),
          content: z.instanceof(Uint8Array),
          content_type: z
            .string()
            .min(1, 'Content type must be a non-empty string'),
        })
        .strict(),
    ]),
    peer: z.union([PeerIdSchema, PeerIdObjectSchema]),
    metadata: MessageMetadataSchema,
    configuration: MessageConfigurationSchema,
    createdAt: z.string().nullable().optional(),
  })
  .strict()

/**
 * Schema for get representation parameters.
 */
export const GetRepresentationParamsSchema = z
  .object({
    peer: z.union([PeerIdSchema, PeerIdObjectSchema]),
    target: z.union([PeerIdSchema, PeerIdObjectSchema]).optional(),
    options: RepresentationOptionsSchema.optional(),
  })
  .strict()

/**
 * Schema for peer get representation parameters.
 */
export const PeerGetRepresentationParamsSchema = z
  .object({
    session: z.union([SessionIdSchema, SessionIdObjectSchema]).optional(),
    scope: ScopeOptionSchema.optional(),
    sessions: SessionAllowlistSchema.optional(),
    target: z.union([PeerIdSchema, PeerIdObjectSchema]).optional(),
    options: RepresentationOptionsSchema.optional(),
  })
  .strict()
  .superRefine(scopeExclusivityIssues)

/**
 * Schema for peer card target parameter.
 */
export const CardTargetSchema = z
  .union([PeerIdSchema, PeerIdObjectSchema])
  .optional()
  .transform((val) =>
    val ? (typeof val === 'string' ? val : val.id) : undefined
  )

/**
 * Schema for peer card content (array of strings).
 */
export const PeerCardContentSchema = z.array(z.string())

/**
 * Schema for peer addition to session.
 */
export const PeerAdditionSchema = z.union([
  PeerIdSchema,
  PeerIdObjectSchema,
  z.array(
    z.union([
      PeerIdSchema,
      PeerIdObjectSchema,
      z.tuple([
        z.union([PeerIdSchema, PeerIdObjectSchema]),
        SessionPeerConfigSchema,
      ]),
    ])
  ),
  z.tuple([
    z.union([PeerIdSchema, PeerIdObjectSchema]),
    SessionPeerConfigSchema,
  ]),
])

/**
 * API format for session peer config.
 */
export type SessionPeerConfigApi = {
  observe_me?: boolean | null
  observe_others?: boolean | null
}

/**
 * API format for peer config.
 */
export type PeerConfigApi = {
  observe_me?: boolean | null
}

/**
 * Transform peer config to API format.
 */
export function peerConfigToApi(
  config: { observeMe?: boolean | null } | undefined
): PeerConfigApi | undefined {
  if (!config) return undefined
  return {
    observe_me: config.observeMe,
  }
}

/**
 * Transform peer config from snake_case (API) to camelCase (SDK).
 */
export function peerConfigFromApi(
  config: PeerConfigApi | Record<string, unknown> | undefined
): { observeMe?: boolean | null } | undefined {
  if (!config) return undefined
  const apiConfig = config as PeerConfigApi
  return {
    observeMe: apiConfig.observe_me,
  }
}

// =============================================================================
// Configuration API Types
// =============================================================================

/**
 * API format for reasoning config (snake_case).
 */
export type ReasoningConfigApi = {
  enabled?: boolean | null
  custom_instructions?: string | null
}

/**
 * API format for peer card config (snake_case).
 */
export type PeerCardConfigApi = {
  use?: boolean | null
  create?: boolean | null
}

/**
 * API format for summary config (snake_case).
 */
export type SummaryConfigApi = {
  enabled?: boolean | null
  messages_per_short_summary?: number | null
  messages_per_long_summary?: number | null
}

/**
 * API format for dream config (snake_case).
 */
export type DreamConfigApi = {
  enabled?: boolean | null
}

/**
 * API format for workspace configuration (snake_case).
 */
export type WorkspaceConfigApi = {
  reasoning?: ReasoningConfigApi | null
  peer_card?: PeerCardConfigApi | null
  summary?: SummaryConfigApi | null
  dream?: DreamConfigApi | null
}

/**
 * API format for session configuration (same as workspace).
 */
export type SessionConfigApi = WorkspaceConfigApi

/**
 * API format for message configuration (snake_case).
 */
export type MessageConfigApi = {
  reasoning?: ReasoningConfigApi | null
}

// =============================================================================
// Configuration Conversion Functions
// =============================================================================

/**
 * Transform reasoning config to API format.
 */
function reasoningConfigToApi(
  config:
    | { enabled?: boolean | null; customInstructions?: string | null }
    | null
    | undefined
): ReasoningConfigApi | null | undefined {
  if (config === null) return null
  if (config === undefined) return undefined
  return {
    enabled: config.enabled,
    custom_instructions: config.customInstructions,
  }
}

/**
 * Transform reasoning config from API format.
 */
function reasoningConfigFromApi(
  config: ReasoningConfigApi | null | undefined
):
  | { enabled?: boolean | null; customInstructions?: string | null }
  | null
  | undefined {
  if (config === null) return null
  if (config === undefined) return undefined
  return {
    enabled: config.enabled,
    customInstructions: config.custom_instructions,
  }
}

/**
 * Transform peer card config to API format.
 */
function peerCardConfigToApi(
  config: { use?: boolean | null; create?: boolean | null } | null | undefined
): PeerCardConfigApi | null | undefined {
  if (config === null) return null
  if (config === undefined) return undefined
  return {
    use: config.use,
    create: config.create,
  }
}

/**
 * Transform peer card config from API format.
 */
function peerCardConfigFromApi(
  config: PeerCardConfigApi | null | undefined
): { use?: boolean | null; create?: boolean | null } | null | undefined {
  if (config === null) return null
  if (config === undefined) return undefined
  return {
    use: config.use,
    create: config.create,
  }
}

/**
 * Transform summary config to API format.
 */
function summaryConfigToApi(
  config:
    | {
        enabled?: boolean | null
        messagesPerShortSummary?: number | null
        messagesPerLongSummary?: number | null
      }
    | null
    | undefined
): SummaryConfigApi | null | undefined {
  if (config === null) return null
  if (config === undefined) return undefined
  return {
    enabled: config.enabled,
    messages_per_short_summary: config.messagesPerShortSummary,
    messages_per_long_summary: config.messagesPerLongSummary,
  }
}

/**
 * Transform summary config from API format.
 */
function summaryConfigFromApi(config: SummaryConfigApi | null | undefined):
  | {
      enabled?: boolean | null
      messagesPerShortSummary?: number | null
      messagesPerLongSummary?: number | null
    }
  | null
  | undefined {
  if (config === null) return null
  if (config === undefined) return undefined
  return {
    enabled: config.enabled,
    messagesPerShortSummary: config.messages_per_short_summary,
    messagesPerLongSummary: config.messages_per_long_summary,
  }
}

/**
 * Transform dream config to API format.
 */
function dreamConfigToApi(
  config: { enabled?: boolean | null } | null | undefined
): DreamConfigApi | null | undefined {
  if (config === null) return null
  if (config === undefined) return undefined
  return {
    enabled: config.enabled,
  }
}

/**
 * Transform dream config from API format.
 */
function dreamConfigFromApi(
  config: DreamConfigApi | null | undefined
): { enabled?: boolean | null } | null | undefined {
  if (config === null) return null
  if (config === undefined) return undefined
  return {
    enabled: config.enabled,
  }
}

/**
 * Transform workspace config to API format (camelCase to snake_case).
 */
export function workspaceConfigToApi(
  config: WorkspaceConfig | undefined
): WorkspaceConfigApi | undefined {
  if (!config) return undefined
  return {
    reasoning: reasoningConfigToApi(config.reasoning),
    peer_card: peerCardConfigToApi(config.peerCard),
    summary: summaryConfigToApi(config.summary),
    dream: dreamConfigToApi(config.dream),
  }
}

/**
 * Transform workspace config from API format (snake_case to camelCase).
 */
export function workspaceConfigFromApi(
  config: WorkspaceConfigApi | Record<string, unknown> | undefined
): WorkspaceConfig | undefined {
  if (!config) return undefined
  const apiConfig = config as WorkspaceConfigApi
  return {
    reasoning: reasoningConfigFromApi(apiConfig.reasoning),
    peerCard: peerCardConfigFromApi(apiConfig.peer_card),
    summary: summaryConfigFromApi(apiConfig.summary),
    dream: dreamConfigFromApi(apiConfig.dream),
  }
}

/**
 * Transform session config to API format (camelCase to snake_case).
 */
export function sessionConfigToApi(
  config: SessionConfig | undefined
): SessionConfigApi | undefined {
  if (!config) return undefined
  return {
    reasoning: reasoningConfigToApi(config.reasoning),
    peer_card: peerCardConfigToApi(config.peerCard),
    summary: summaryConfigToApi(config.summary),
    dream: dreamConfigToApi(config.dream),
  }
}

/**
 * Transform session config from API format (snake_case to camelCase).
 */
export function sessionConfigFromApi(
  config: SessionConfigApi | Record<string, unknown> | undefined
): SessionConfig | undefined {
  if (!config) return undefined
  const apiConfig = config as SessionConfigApi
  return {
    reasoning: reasoningConfigFromApi(apiConfig.reasoning),
    peerCard: peerCardConfigFromApi(apiConfig.peer_card),
    summary: summaryConfigFromApi(apiConfig.summary),
    dream: dreamConfigFromApi(apiConfig.dream),
  }
}

/**
 * Transform message config to API format (camelCase to snake_case).
 */
export function messageConfigToApi(
  config: MessageConfiguration | undefined
): MessageConfigApi | undefined {
  if (!config) return undefined
  return {
    reasoning: reasoningConfigToApi(config.reasoning),
  }
}

/**
 * Transform message config from API format (snake_case to camelCase).
 */
export function messageConfigFromApi(
  config: MessageConfigApi | Record<string, unknown> | undefined
): MessageConfiguration | undefined {
  if (!config) return undefined
  const apiConfig = config as MessageConfigApi
  return {
    reasoning: reasoningConfigFromApi(apiConfig.reasoning),
  }
}

/**
 * Check if a value is a config object (has observeMe or observeOthers).
 */
function isSessionPeerConfig(
  val: unknown
): val is { observeMe?: boolean | null; observeOthers?: boolean | null } {
  return (
    typeof val === 'object' &&
    val !== null &&
    !('id' in val) &&
    ('observeMe' in val || 'observeOthers' in val)
  )
}

/**
 * Check if input is a tuple [peer, config].
 */
function isTuple(
  input: unknown
): input is
  | [string, { observeMe?: boolean | null; observeOthers?: boolean | null }]
  | [
      { id: string },
      { observeMe?: boolean | null; observeOthers?: boolean | null },
    ] {
  return (
    Array.isArray(input) && input.length === 2 && isSessionPeerConfig(input[1])
  )
}

/**
 * Schema that validates and transforms peer addition input to API format.
 * Handles all input variations and outputs a dictionary ready for the API.
 */
export const PeerAdditionToApiSchema = PeerAdditionSchema.transform(
  (input): Record<string, SessionPeerConfigApi> => {
    const result: Record<string, SessionPeerConfigApi> = {}

    // Helper to process a single peer entry
    const processEntry = (entry: unknown): void => {
      if (typeof entry === 'string') {
        result[entry] = {}
      } else if (isTuple(entry)) {
        const [peer, config] = entry
        const id = typeof peer === 'string' ? peer : peer.id
        result[id] = {
          observe_me: config.observeMe,
          observe_others: config.observeOthers,
        }
      } else if (typeof entry === 'object' && entry !== null && 'id' in entry) {
        result[(entry as { id: string }).id] = {}
      }
    }

    // Handle single tuple specially (it's an array but represents one entry)
    if (isTuple(input)) {
      processEntry(input)
    } else if (Array.isArray(input)) {
      // Array of entries
      for (const item of input) {
        processEntry(item)
      }
    } else {
      // Single string or object
      processEntry(input)
    }

    return result
  }
)

/**
 * Schema for peer removal from session.
 */
export const PeerRemovalSchema = z.union([
  PeerIdSchema,
  PeerIdObjectSchema,
  z.array(z.union([PeerIdSchema, PeerIdObjectSchema])),
])

/**
 * Schema for message addition to session.
 */
export const MessageAdditionSchema = z.union([
  MessageInputSchema,
  z.array(MessageInputSchema),
])

/**
 * Schema that validates and transforms message addition to API format.
 */
export const MessageAdditionToApiSchema = MessageAdditionSchema.transform(
  (input) => {
    const messages = Array.isArray(input) ? input : [input]
    return messages.map((msg) => ({
      peer_id: msg.peerId,
      content: msg.content,
      metadata: msg.metadata,
      configuration: messageConfigToApi(msg.configuration ?? undefined),
      created_at: msg.createdAt,
    }))
  }
)

/**
 * Schema for workspace metadata.
 */
export const WorkspaceMetadataSchema = z.record(z.string(), z.unknown())

/**
 * Schema for workspace configuration.
 * Includes reasoning, peer card, summary, and dream settings.
 */
export const WorkspaceConfigSchema = z
  .object({
    reasoning: ReasoningConfigSchema.nullable().optional(),
    peerCard: PeerCardConfigSchema.nullable().optional(),
    summary: SummaryConfigSchema.nullable().optional(),
    dream: DreamConfigSchema.nullable().optional(),
  })
  .strict()

/**
 * Schema for limit.
 */
export const LimitSchema = z
  .number()
  .int()
  .min(1, 'Limit must be a positive integer')
  .max(100, 'Limit must be less than or equal to 100')

/**
 * Schema for conclusion query parameters.
 */
export const ConclusionQueryParamsSchema = z
  .object({
    query: SearchQuerySchema,
    top_k: z
      .number()
      .int()
      .min(1, 'top_k must be at least 1')
      .max(100, 'top_k must be at most 100')
      .optional(),
    distance: z
      .number()
      .min(0.0, 'distance must be at least 0.0')
      .max(1.0, 'distance must be at most 1.0')
      .optional(),
    filters: FilterSchema,
  })
  .strict()

/**
 * Type exports for use throughout the SDK.
 */
export type HonchoConfig = z.infer<typeof HonchoConfigSchema>
export type PeerMetadata = z.infer<typeof PeerMetadataSchema>
export type PeerConfig = z.infer<typeof PeerConfigSchema>
export type SessionMetadata = z.infer<typeof SessionMetadataSchema>
export type SessionConfig = z.infer<typeof SessionConfigSchema>
export type SessionPeerConfig = z.infer<typeof SessionPeerConfigSchema>
export type MessageInput = z.infer<typeof MessageInputSchema>
export type Filters = z.infer<typeof FilterSchema>
export type ChatQuery = z.infer<typeof ChatQuerySchema>
export type ContextParams = z.infer<typeof ContextParamsSchema>
export type SearchQueryLike = z.infer<typeof SearchQueryLikeSchema>
export type QueueStatusOptions = z.infer<typeof QueueStatusOptionsSchema>
export type FileUpload = z.infer<typeof FileUploadSchema>
export type GetRepresentationParams = z.infer<
  typeof GetRepresentationParamsSchema
>
export type PeerGetRepresentationParams = z.infer<
  typeof PeerGetRepresentationParamsSchema
>
export type PeerAddition = z.infer<typeof PeerAdditionSchema>
export type PeerAdditionApi = z.infer<typeof PeerAdditionToApiSchema>
export type PeerRemoval = z.infer<typeof PeerRemovalSchema>
export type MessageAddition = z.infer<typeof MessageAdditionSchema>
export type WorkspaceMetadata = z.infer<typeof WorkspaceMetadataSchema>
export type WorkspaceConfig = z.infer<typeof WorkspaceConfigSchema>
export type ReasoningConfig = z.infer<typeof ReasoningConfigSchema>
export type PeerCardConfig = z.infer<typeof PeerCardConfigSchema>
export type SummaryConfig = z.infer<typeof SummaryConfigSchema>
export type DreamConfig = z.infer<typeof DreamConfigSchema>
export type MessageConfiguration = z.infer<typeof MessageConfigurationSchema>
export type Limit = z.infer<typeof LimitSchema>
export type ConclusionQueryParams = z.infer<typeof ConclusionQueryParamsSchema>
