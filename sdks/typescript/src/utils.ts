import type {
  QueueStatus,
  QueueStatusResponse,
  SessionQueueStatus,
  SessionQueueStatusResponse,
} from './types/api'

/**
 * Resolve an ID from either a string or an object with an `id` property.
 */
export function resolveId(obj: string | { id: string }): string {
  return typeof obj === 'string' ? obj : obj.id
}

/**
 * Interface for queue status objects that can be polled
 */
export interface PollableQueueStatus {
  pendingWorkUnits: number
  inProgressWorkUnits: number
}

/**
 * Transform a SessionQueueStatusResponse to SessionQueueStatus (snake_case to camelCase).
 */
function transformSessionQueueStatus(
  status: SessionQueueStatusResponse
): SessionQueueStatus {
  return {
    sessionId: status.session_id,
    totalWorkUnits: status.total_work_units,
    completedWorkUnits: status.completed_work_units,
    inProgressWorkUnits: status.in_progress_work_units,
    pendingWorkUnits: status.pending_work_units,
  }
}

/**
 * Transform a QueueStatusResponse to QueueStatus (snake_case to camelCase).
 */
export function transformQueueStatus(status: QueueStatusResponse): QueueStatus {
  const sessions = status.sessions
    ? Object.fromEntries(
        Object.entries(status.sessions).map(([key, value]) => [
          key,
          transformSessionQueueStatus(value),
        ])
      )
    : undefined

  return {
    totalWorkUnits: status.total_work_units,
    completedWorkUnits: status.completed_work_units,
    inProgressWorkUnits: status.in_progress_work_units,
    pendingWorkUnits: status.pending_work_units,
    sessions,
  }
}
