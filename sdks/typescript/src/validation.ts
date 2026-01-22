import { z } from 'zod'
import type { MessageResponse } from './types/api'

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
  .max(100, 'Workspace ID can be at most 100 characters')

/**
 * Schema for Honcho client configuration options.
 */
export const HonchoConfigSchema = z.object({
  apiKey: z.string().optional(),
  environment: z.enum(['local', 'production']).optional(),
  baseURL: z.url('Base URL must be a valid URL').optional(),
  workspaceId: WorkspaceIdSchema.optional(),
  timeout: z.number().positive('Timeout must be a positive number').optional(),
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

/**
 * Schema for peer metadata.
 */
export const PeerMetadataSchema = z.record(z.string(), z.unknown())

/**
 * Schema for peer configuration.
 */
export const PeerConfigSchema = z.object({
  observeMe: z.boolean().nullable().optional(),
})

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
  .max(100, 'Peer ID can be at most 100 characters')

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
export const ReasoningConfigSchema = z.object({
  enabled: z.boolean().nullable().optional(),
  customInstructions: z.string().nullable().optional(),
})

/**
 * Schema for peer card configuration.
 * Used in workspace and session configuration.
 */
export const PeerCardConfigSchema = z.object({
  use: z.boolean().nullable().optional(),
  create: z.boolean().nullable().optional(),
})

/**
 * Schema for summary configuration.
 * Used in workspace and session configuration.
 */
export const SummaryConfigSchema = z.object({
  enabled: z.boolean().nullable().optional(),
  messagesPerShortSummary: z.number().int().min(10).nullable().optional(),
  messagesPerLongSummary: z.number().int().min(20).nullable().optional(),
})

/**
 * Schema for dream configuration.
 * Used in workspace and session configuration.
 */
export const DreamConfigSchema = z.object({
  enabled: z.boolean().nullable().optional(),
})

/**
 * Schema for session configuration.
 * Includes reasoning, peer card, summary, and dream settings.
 */
export const SessionConfigSchema = z.object({
  reasoning: ReasoningConfigSchema.nullable().optional(),
  peerCard: PeerCardConfigSchema.nullable().optional(),
  summary: SummaryConfigSchema.nullable().optional(),
  dream: DreamConfigSchema.nullable().optional(),
})

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
  .max(100, 'Session ID can be at most 100 characters')

/**
 * Schema for session peer configuration.
 */
export const SessionPeerConfigSchema = z.object({
  observeMe: z.boolean().nullable().optional(),
  observeOthers: z.boolean().nullable().optional(),
})

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
  .nullable()
  .optional()

/**
 * Schema for message input.
 */
export const MessageInputSchema = z.object({
  peerId: PeerIdSchema,
  content: MessageContentSchema,
  metadata: MessageMetadataSchema,
  configuration: MessageConfigurationSchema,
  createdAt: z.string().nullable().optional(),
})

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
 * Schema for filter objects.
 */
export const FilterSchema = z.record(z.string(), z.unknown()).optional()

/**
 * Schema for chat query parameters.
 */
export const ChatQuerySchema = z.object({
  query: SearchQuerySchema,
  target: z
    .union([PeerIdSchema, z.object({ id: PeerIdSchema })])
    .optional()
    .transform((val) =>
      val ? (typeof val === 'string' ? val : val.id) : undefined
    ),
  session: z
    .union([SessionIdSchema, z.object({ id: SessionIdSchema })])
    .optional()
    .transform((val) =>
      val ? (typeof val === 'string' ? val : val.id) : undefined
    ),
  reasoningLevel: z
    .enum(['minimal', 'low', 'medium', 'high', 'max'])
    .optional(),
})

/**
 * Schema for validating Message API responses (snake_case).
 */
const MessageResponseSchema: z.ZodType<MessageResponse> = z.object({
  id: z.string(),
  content: z.string(),
  created_at: z.string(),
  peer_id: PeerIdSchema,
  session_id: SessionIdSchema,
  token_count: z.number(),
  workspace_id: WorkspaceIdSchema,
  metadata: z.record(z.string(), z.unknown()),
}) as z.ZodType<MessageResponse>

/**
 * Schema for representation options.
 */
export const RepresentationOptionsSchema = z.object({
  searchQuery: z
    .string()
    .min(1, 'searchQuery must be a non-empty string')
    .refine(
      (query: string) => query.trim().length > 0,
      'searchQuery cannot be only whitespace'
    )
    .optional(),
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

/**
 * Schema for context retrieval parameters.
 */
export const ContextParamsSchema = z
  .object({
    summary: z.boolean().optional(),
    tokens: z.int('Token limit must be an integer').optional(),
    lastUserMessage: z
      .union([
        z.string().min(1, 'Last user message must be a non-empty string'),
        MessageResponseSchema,
      ])
      .optional(),
    peerTarget: PeerIdSchema.optional(),
    peerPerspective: PeerIdSchema.optional(),
    limitToSession: z.boolean().optional(),
    representationOptions: RepresentationOptionsSchema.optional(),
  })
  .superRefine((data, ctx) => {
    if (data.lastUserMessage && !data.peerTarget) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'peerTarget is required when lastUserMessage is provided',
        path: ['lastUserMessage'],
      })
    }

    if (data.peerPerspective && !data.peerTarget) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'peerTarget is required when peerPerspective is provided',
        path: ['peerPerspective'],
      })
    }
  })

/**
 * Schema for deriver status options.
 */
export const QueueStatusOptionsSchema = z.object({
  observer: z.union([PeerIdSchema, z.object({ id: PeerIdSchema })]).optional(),
  sender: z.union([PeerIdSchema, z.object({ id: PeerIdSchema })]).optional(),
  session: z
    .union([SessionIdSchema, z.object({ id: SessionIdSchema })])
    .optional(),
  timeout: z.number().positive('Timeout must be a positive number').optional(),
})

/**
 * Schema for file upload parameters.
 * Supports File objects (browser), Buffer, Uint8Array, and custom uploadable objects.
 */
export const FileUploadSchema = z.object({
  file: z.union([
    // Browser File object
    z.instanceof(File),
    // Node.js Buffer
    z.instanceof(Buffer),
    // Uint8Array
    z.instanceof(Uint8Array),
    // Custom uploadable object with filename, content, and content_type
    z.object({
      filename: z.string().min(1, 'Filename must be a non-empty string'),
      content: z.union([z.instanceof(Buffer), z.instanceof(Uint8Array)]),
      content_type: z
        .string()
        .min(1, 'Content type must be a non-empty string'),
    }),
    // Fallback for any other uploadable type
    z
      .any()
      .refine(
        (val) => val !== null && val !== undefined,
        'File must not be null or undefined'
      ),
  ]),
  peer: z.union([PeerIdSchema, z.object({ id: PeerIdSchema })]),
  metadata: MessageMetadataSchema,
  configuration: z.record(z.string(), z.unknown()).optional(),
  createdAt: z.string().nullable().optional(),
})

/**
 * Schema for get representation parameters.
 */
export const GetRepresentationParamsSchema = z.object({
  peer: z.union([PeerIdSchema, z.object({ id: PeerIdSchema })]),
  target: z.union([PeerIdSchema, z.object({ id: PeerIdSchema })]).optional(),
  options: RepresentationOptionsSchema.optional(),
})

/**
 * Schema for peer get representation parameters.
 */
export const PeerGetRepresentationParamsSchema = z.object({
  session: z
    .union([SessionIdSchema, z.object({ id: SessionIdSchema })])
    .optional(),
  target: z.union([PeerIdSchema, z.object({ id: PeerIdSchema })]).optional(),
  options: RepresentationOptionsSchema.optional(),
})

/**
 * Schema for peer card target parameter.
 */
export const CardTargetSchema = z
  .union([PeerIdSchema, z.object({ id: PeerIdSchema })])
  .optional()
  .transform((val) =>
    val ? (typeof val === 'string' ? val : val.id) : undefined
  )

/**
 * Schema for peer addition to session.
 */
export const PeerAdditionSchema = z.union([
  PeerIdSchema,
  z.object({ id: PeerIdSchema }),
  z.array(
    z.union([
      PeerIdSchema,
      z.object({ id: PeerIdSchema }),
      z.tuple([
        z.union([PeerIdSchema, z.object({ id: PeerIdSchema })]),
        SessionPeerConfigSchema,
      ]),
    ])
  ),
  z.tuple([
    z.union([PeerIdSchema, z.object({ id: PeerIdSchema })]),
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
  z.object({ id: PeerIdSchema }),
  z.array(z.union([PeerIdSchema, z.object({ id: PeerIdSchema })])),
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
export const WorkspaceConfigSchema = z.object({
  reasoning: ReasoningConfigSchema.nullable().optional(),
  peerCard: PeerCardConfigSchema.nullable().optional(),
  summary: SummaryConfigSchema.nullable().optional(),
  dream: DreamConfigSchema.nullable().optional(),
})

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
export const ConclusionQueryParamsSchema = z.object({
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
