import type { Message } from '@honcho-ai/core/resources/workspaces/sessions/messages'
import { z } from 'zod'

/**
 * Validation schemas for the Honcho TypeScript SDK.
 *
 * These schemas ensure type safety and runtime validation for all inputs
 * to the SDK, providing clear error messages when validation fails.
 */

/**
 * Schema for Honcho client configuration options.
 */
export const HonchoConfigSchema = z.object({
  apiKey: z.string().optional(),
  environment: z.enum(['local', 'production', 'demo']).optional(),
  baseURL: z.string().url('Base URL must be a valid URL').optional(),
  workspaceId: z
    .string()
    .min(1, 'Workspace ID must be a non-empty string')
    .optional(),
  timeout: z.number().positive('Timeout must be a positive number').optional(),
  maxRetries: z
    .number()
    .int()
    .min(0, 'Max retries must be a non-negative integer')
    .optional(),
  defaultHeaders: z.record(z.string(), z.string()).optional(),
  defaultQuery: z.record(z.string(), z.unknown()).optional(),
})

/**
 * Schema for peer metadata.
 */
export const PeerMetadataSchema = z.record(z.string(), z.unknown())

/**
 * Schema for peer configuration.
 */
export const PeerConfigSchema = z.record(z.string(), z.unknown())

/**
 * Schema for peer ID validation.
 */
export const PeerIdSchema = z
  .string()
  .min(1, 'Peer ID must be a non-empty string')

/**
 * Schema for session metadata.
 */
export const SessionMetadataSchema = z.record(z.string(), z.unknown())

/**
 * Schema for session configuration.
 */
export const SessionConfigSchema = z.record(z.string(), z.unknown())

/**
 * Schema for session ID validation.
 */
export const SessionIdSchema = z
  .string()
  .min(1, 'Session ID must be a non-empty string')

/**
 * Schema for session peer configuration.
 */
export const SessionPeerConfigSchema = z.object({
  observe_me: z.boolean().nullable().optional(),
  observe_others: z.boolean().optional(),
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
 * Schema for message creation.
 */
export const MessageCreateSchema = z.object({
  peer_id: PeerIdSchema,
  content: MessageContentSchema,
  metadata: MessageMetadataSchema,
  created_at: z.string().nullable().optional(),
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
  stream: z.boolean().optional().default(false),
  target: z.union([z.string(), z.object({ id: z.string() })]).optional(),
  sessionId: z.string().optional(),
})

/**
 * Schema for validating Message objects from the core SDK.
 */
const MessageSchema: z.ZodType<Message> = z.object({
  id: z.string(),
  content: z.string(),
  created_at: z.string(),
  peer_id: z.string(),
  session_id: z.string(),
  token_count: z.number(),
  workspace_id: z.string(),
  metadata: z.record(z.string(), z.unknown()).optional(),
}) as z.ZodType<Message>

/**
 * Schema for representation options.
 */
export const RepresentationOptionsSchema = z.object({
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
  includeMostDerived: z.boolean().optional(),
  maxObservations: z
    .number()
    .int()
    .min(1, 'maxObservations must be at least 1')
    .max(100, 'maxObservations must be at most 100')
    .optional(),
})

/**
 * Schema for context retrieval parameters.
 */
export const ContextParamsSchema = z
  .object({
    summary: z.boolean().optional(),
    tokens: z
      .number()
      .positive('Token limit must be a positive number')
      .optional(),
    lastUserMessage: z
      .union([
        z.string().min(1, 'Last user message must be a non-empty string'),
        MessageSchema,
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
export const DeriverStatusOptionsSchema = z.object({
  observerId: z.string().optional(),
  senderId: z.string().optional(),
  sessionId: z.string().optional(),
  timeoutMs: z
    .number()
    .positive('Timeout must be a positive number')
    .optional(),
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
  peerId: PeerIdSchema,
})

/**
 * Schema for working representation parameters.
 */
export const WorkingRepParamsSchema = z.object({
  peer: z.union([z.string(), z.object({ id: z.string() })]),
  target: z.union([z.string(), z.object({ id: z.string() })]).optional(),
  options: RepresentationOptionsSchema.extend({
    searchQuery: SearchQuerySchema.optional(),
  }).optional(),
})

/**
 * Schema for peer working representation parameters.
 */
export const PeerWorkingRepParamsSchema = z.object({
  session: z.union([z.string(), z.object({ id: z.string() })]).optional(),
  target: z.union([z.string(), z.object({ id: z.string() })]).optional(),
  options: RepresentationOptionsSchema.extend({
    searchQuery: SearchQuerySchema.optional(),
  }).optional(),
})

/**
 * Schema for peer addition to session.
 */
export const PeerAdditionSchema = z.union([
  z.string(),
  z.object({ id: z.string() }),
  z.array(
    z.union([
      z.string(),
      z.object({ id: z.string() }),
      z.tuple([
        z.union([z.string(), z.object({ id: z.string() })]),
        SessionPeerConfigSchema,
      ]),
    ])
  ),
  z.tuple([
    z.union([z.string(), z.object({ id: z.string() })]),
    SessionPeerConfigSchema,
  ]),
])

/**
 * Schema for peer removal from session.
 */
export const PeerRemovalSchema = z.union([
  z.string(),
  z.object({ id: z.string() }),
  z.array(z.union([z.string(), z.object({ id: z.string() })])),
])

/**
 * Schema for message addition to session.
 */
export const MessageAdditionSchema = z.union([
  MessageCreateSchema,
  z.array(MessageCreateSchema),
])

/**
 * Schema for workspace metadata.
 */
export const WorkspaceMetadataSchema = z.record(z.string(), z.unknown())

/**
 * Schema for workspace configuration.
 */
export const WorkspaceConfigSchema = z.record(z.string(), z.unknown())

/**
 * Schema for limit.
 */
export const LimitSchema = z
  .number()
  .int()
  .min(1, 'Limit must be a positive integer')
  .max(100, 'Limit must be less than or equal to 100')

/**
 * Type exports for use throughout the SDK.
 */
export type HonchoConfig = z.infer<typeof HonchoConfigSchema>
export type PeerMetadata = z.infer<typeof PeerMetadataSchema>
export type PeerConfig = z.infer<typeof PeerConfigSchema>
export type SessionMetadata = z.infer<typeof SessionMetadataSchema>
export type SessionConfig = z.infer<typeof SessionConfigSchema>
export type SessionPeerConfig = z.infer<typeof SessionPeerConfigSchema>
export type MessageCreate = z.infer<typeof MessageCreateSchema>
export type Filters = z.infer<typeof FilterSchema>
export type ChatQuery = z.infer<typeof ChatQuerySchema>
export type ContextParams = z.infer<typeof ContextParamsSchema>
export type DeriverStatusOptions = z.infer<typeof DeriverStatusOptionsSchema>
export type FileUpload = z.infer<typeof FileUploadSchema>
export type WorkingRepParams = z.infer<typeof WorkingRepParamsSchema>
export type PeerWorkingRepParams = z.infer<typeof PeerWorkingRepParamsSchema>
export type PeerAddition = z.infer<typeof PeerAdditionSchema>
export type PeerRemoval = z.infer<typeof PeerRemovalSchema>
export type MessageAddition = z.infer<typeof MessageAdditionSchema>
export type WorkspaceMetadata = z.infer<typeof WorkspaceMetadataSchema>
export type WorkspaceConfig = z.infer<typeof WorkspaceConfigSchema>
export type Limit = z.infer<typeof LimitSchema>
