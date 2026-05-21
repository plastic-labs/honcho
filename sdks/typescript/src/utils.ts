import type {
  QueueStatus,
  QueueStatusResponse,
  QueueWorkUnit,
  QueueWorkUnitResponse,
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
    pendingStalledWorkUnits: status.pending_stalled_work_units ?? 0,
    pendingReadyWorkUnits: status.pending_ready_work_units ?? 0,
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
    pendingStalledWorkUnits: status.pending_stalled_work_units ?? 0,
    pendingReadyWorkUnits: status.pending_ready_work_units ?? 0,
    sessions,
  }
}

/**
 * Transform a QueueWorkUnitResponse to QueueWorkUnit (snake_case to camelCase).
 */
export function transformQueueWorkUnit(
  row: QueueWorkUnitResponse
): QueueWorkUnit {
  return {
    workUnitKey: row.work_unit_key,
    taskType: row.task_type,
    sessionId: row.session_id,
    sessionName: row.session_name,
    observer: row.observer,
    observed: row.observed,
    pendingItems: row.pending_items,
    pendingTokens: row.pending_tokens,
    tokensUntilThreshold: row.tokens_until_threshold,
    hitThreshold: row.hit_threshold,
    inProgress: row.in_progress,
    oldestItemAt: row.oldest_item_at,
    newestItemAt: row.newest_item_at,
  }
}
