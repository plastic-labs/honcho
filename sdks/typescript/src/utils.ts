import type {
  QueueStatus,
  QueueStatusResponse,
  SessionQueueStatus,
  SessionQueueStatusResponse,
} from './types/api'

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

/**
 * Poll a status getter function until pendingWorkUnits and inProgressWorkUnits are both 0.
 *
 * This utility allows you to guarantee that all work has been processed by the queue.
 * The polling estimates sleep time by assuming each work unit takes 1 second.
 *
 * @param getStatus - A function that returns a Promise resolving to the current status
 * @param timeout - Optional timeout in seconds (default: 300 - 5 minutes)
 * @returns Promise resolving to the final status when processing is complete
 * @throws Error if timeout is exceeded before processing completes
 */
export async function pollUntilComplete<T extends PollableQueueStatus>(
  getStatus: () => Promise<T>,
  timeout: number = 300
): Promise<T> {
  const startTime = Date.now()
  const timeoutMs = timeout * 1000

  while (true) {
    const status = await getStatus()
    if (status.pendingWorkUnits === 0 && status.inProgressWorkUnits === 0) {
      return status
    }

    // Check if timeout has been exceeded
    const elapsedTime = Date.now() - startTime
    if (elapsedTime >= timeoutMs) {
      throw new Error(
        `Polling timeout exceeded after ${timeout}s. ` +
          `Current status: ${status.pendingWorkUnits} pending, ${status.inProgressWorkUnits} in progress work units.`
      )
    }

    // Sleep for the expected time to complete all current work units
    // Assuming each pending and in-progress work unit takes 1 second
    const totalWorkUnits = status.pendingWorkUnits + status.inProgressWorkUnits
    const sleepMs = Math.max(1000, totalWorkUnits * 1000) // Sleep at least 1 second

    // Ensure we don't sleep past the timeout
    const remainingTime = timeoutMs - elapsedTime
    const actualSleepMs = Math.min(sleepMs, remainingTime)

    if (actualSleepMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, actualSleepMs))
    }
  }
}
