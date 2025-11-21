export interface RepresentationOptions {
  searchQuery?: string
  searchTopK?: number
  searchMaxDistance?: number
  includeMostDerived?: boolean
  maxObservations?: number
}

/**
 * Metadata associated with an observation.
 */
export interface ObservationMetadata {
  created_at: string
  message_ids: Array<[number, number]>
  session_name: string
}

/**
 * An explicit observation with full metadata.
 * Represents facts LITERALLY stated - direct quotes or clear paraphrases only.
 */
export interface ExplicitObservationBase {
  content: string
}

/**
 * Base interface for deductive observations - logical conclusions.
 */
export interface DeductiveObservationBase {
  premises: string[]
  conclusion: string
}

export interface ExplicitObservation
  extends ExplicitObservationBase,
    ObservationMetadata {}

/**
 * A deductive observation with full metadata.
 * Represents conclusions that MUST be true given explicit facts and premises.
 */
export interface DeductiveObservation
  extends DeductiveObservationBase,
    ObservationMetadata {}

/**
 * Raw representation data structure returned from the API.
 */
export interface RepresentationData {
  explicit: ExplicitObservation[]
  deductive: DeductiveObservation[]
}

/**
 * A Representation is a traversable and diffable map of observations.
 *
 * At the base, we have a list of explicit observations, derived from a peer's messages.
 * From there, deductive observations can be made by establishing logical relationships
 * between explicit observations.
 *
 * All of a peer's observations are stored as documents in a collection. These documents
 * can be queried in various ways to produce this Representation object.
 *
 * A "working representation" is a version of this data structure representing the most
 * recent observations within a single session.
 */
export class Representation {
  /**
   * Facts LITERALLY stated - direct quotes or clear paraphrases only, no interpretation or inference.
   */
  explicit: ExplicitObservation[]

  /**
   * Conclusions that MUST be true given explicit facts and premises - strict logical necessities.
   */
  deductive: DeductiveObservation[]

  /**
   * Create a new Representation from observation lists.
   *
   * @param explicit - List of explicit observations
   * @param deductive - List of deductive observations
   */
  constructor(
    explicit: ExplicitObservation[] = [],
    deductive: DeductiveObservation[] = []
  ) {
    this.explicit = explicit
    this.deductive = deductive
  }

  /**
   * Check if the representation is empty.
   *
   * @returns True if both explicit and deductive observation lists are empty
   */
  isEmpty(): boolean {
    return this.explicit.length === 0 && this.deductive.length === 0
  }

  /**
   * Given this and another representation, return a new representation with only
   * observations that are unique to the other.
   *
   * Note: This only removes literal duplicates based on stringified comparison,
   * not semantically equivalent ones.
   *
   * @param other - The representation to compare against
   * @returns A new Representation containing only observations unique to other
   */
  diff(other: Representation): Representation {
    const thisExplicitSet = new Set(
      this.explicit.map((obs) => this._hashExplicit(obs))
    )
    const thisDeductiveSet = new Set(
      this.deductive.map((obs) => this._hashDeductive(obs))
    )

    const uniqueExplicit = other.explicit.filter(
      (obs) => !thisExplicitSet.has(this._hashExplicit(obs))
    )
    const uniqueDeductive = other.deductive.filter(
      (obs) => !thisDeductiveSet.has(this._hashDeductive(obs))
    )

    return new Representation(uniqueExplicit, uniqueDeductive)
  }

  /**
   * Merge another representation into this one.
   *
   * This automatically deduplicates explicit and deductive observations.
   * Preserves order of observations to retain FIFO order.
   *
   * Note: Observations with the same timestamp may not have order preserved,
   * but that's acceptable since they're from the same timestamp.
   *
   * @param other - The representation to merge into this one
   * @param maxObservations - Optional maximum number of observations to keep per type
   */
  merge(other: Representation, maxObservations?: number): void {
    // Deduplicate by converting to Set using hash, then back to array
    const explicitMap = new Map<string, ExplicitObservation>()
    const deductiveMap = new Map<string, DeductiveObservation>()

    // Add existing observations
    for (const obs of this.explicit) {
      explicitMap.set(this._hashExplicit(obs), obs)
    }
    for (const obs of this.deductive) {
      deductiveMap.set(this._hashDeductive(obs), obs)
    }

    // Add new observations (overwrites duplicates)
    for (const obs of other.explicit) {
      explicitMap.set(this._hashExplicit(obs), obs)
    }
    for (const obs of other.deductive) {
      deductiveMap.set(this._hashDeductive(obs), obs)
    }

    // Convert back to arrays and sort by created_at
    this.explicit = Array.from(explicitMap.values()).sort(
      (a, b) =>
        this._parseTimestampForSort(a.created_at) -
        this._parseTimestampForSort(b.created_at)
    )
    this.deductive = Array.from(deductiveMap.values()).sort(
      (a, b) =>
        this._parseTimestampForSort(a.created_at) -
        this._parseTimestampForSort(b.created_at)
    )

    // Apply max observations limit if specified
    if (maxObservations !== undefined) {
      this.explicit = this.explicit.slice(-maxObservations)
      this.deductive = this.deductive.slice(-maxObservations)
    }
  }

  /**
   * Format representation into a clean, readable string for LLM prompts.
   *
   * Timestamps are stripped of subsecond precision for cleaner display.
   *
   * @returns Formatted string with clear sections and numbered items including timestamps
   *
   * @example
   * ```
   * EXPLICIT:
   * 1. [2025-01-01T12:00:00Z] The user has a dog named Rover
   * 2. [2025-01-01T12:01:00Z] The user's dog is 5 years old
   *
   * DEDUCTIVE:
   * 1. [2025-01-01T12:01:00Z] Rover is 5 years old
   *     - The user has a dog named Rover
   *     - The user's dog is 5 years old
   * ```
   */
  toString(): string {
    const parts: string[] = []

    parts.push('EXPLICIT:\n')
    for (let i = 0; i < this.explicit.length; i++) {
      const obs = this.explicit[i]
      const timestamp = this._stripMicroseconds(obs.created_at)
      parts.push(`${i + 1}. [${timestamp}] ${obs.content}`)
    }
    parts.push('')

    parts.push('DEDUCTIVE:\n')
    for (let i = 0; i < this.deductive.length; i++) {
      const obs = this.deductive[i]
      const timestamp = this._stripMicroseconds(obs.created_at)
      parts.push(`${i + 1}. [${timestamp}] ${obs.conclusion}`)
      for (const premise of obs.premises) {
        parts.push(`    - ${premise}`)
      }
    }
    parts.push('')

    return parts.join('\n')
  }

  /**
   * Format representation into a clean, readable string without timestamps.
   *
   * @returns Formatted string with clear sections and numbered items without temporal metadata
   *
   * @example
   * ```
   * EXPLICIT:
   * 1. The user has a dog named Rover
   * 2. The user's dog is 5 years old
   *
   * DEDUCTIVE:
   * 1. Rover is 5 years old
   *     - The user has a dog named Rover
   *     - The user's dog is 5 years old
   * ```
   */
  toStringNoTimestamps(): string {
    const parts: string[] = []

    parts.push('EXPLICIT:\n')
    for (let i = 0; i < this.explicit.length; i++) {
      parts.push(`${i + 1}. ${this.explicit[i].content}`)
    }
    parts.push('')

    parts.push('DEDUCTIVE:\n')
    for (let i = 0; i < this.deductive.length; i++) {
      const obs = this.deductive[i]
      parts.push(`${i + 1}. ${obs.conclusion}`)
      for (const premise of obs.premises) {
        parts.push(`    - ${premise}`)
      }
    }
    parts.push('')

    return parts.join('\n')
  }

  /**
   * Format a Representation object as markdown.
   *
   * Timestamps are stripped of subsecond precision for cleaner display.
   *
   * @returns Formatted markdown string with headers and lists
   */
  toMarkdown(): string {
    const parts: string[] = []

    parts.push('## Explicit Observations\n')
    for (let i = 0; i < this.explicit.length; i++) {
      const obs = this.explicit[i]
      const timestamp = this._stripMicroseconds(obs.created_at)
      parts.push(`${i + 1}. [${timestamp}] ${obs.content}`)
    }
    parts.push('')

    parts.push('## Deductive Observations\n')
    for (let i = 0; i < this.deductive.length; i++) {
      const obs = this.deductive[i]
      const timestamp = this._stripMicroseconds(obs.created_at)
      parts.push(`${i + 1}. **Conclusion**: ${obs.conclusion}`)
      parts.push(`   **Created**: ${timestamp}`)
      if (obs.premises.length > 0) {
        parts.push('   **Premises**:')
        for (const premise of obs.premises) {
          parts.push(`   - ${premise}`)
        }
      }
      parts.push('')
    }

    return parts.join('\n')
  }

  /**
   * Create a Representation from raw API response data.
   *
   * @param data - Raw representation data from the API
   * @returns A new Representation instance
   */
  static fromData(data: RepresentationData): Representation {
    return new Representation(data.explicit, data.deductive)
  }

  /**
   * Create a hash string for an explicit observation for deduplication.
   * Based on content, created_at, and session_name.
   */
  private _hashExplicit(obs: ExplicitObservation): string {
    return JSON.stringify({
      content: obs.content,
      created_at: obs.created_at,
      session_name: obs.session_name,
    })
  }

  /**
   * Create a hash string for a deductive observation for deduplication.
   * Based on conclusion, created_at, and session_name (premises not included).
   */
  private _hashDeductive(obs: DeductiveObservation): string {
    return JSON.stringify({
      conclusion: obs.conclusion,
      created_at: obs.created_at,
      session_name: obs.session_name,
    })
  }

  /**
   * Strip microseconds from ISO timestamp for cleaner display.
   */
  private _stripMicroseconds(timestamp: string): string {
    try {
      const date = new Date(timestamp)
      return date.toISOString().replace(/\.\d{3}Z$/, 'Z')
    } catch {
      return timestamp
    }
  }

  /**
   * Safely parse a timestamp and return milliseconds since epoch for sorting.
   * Handles microsecond precision by truncating to milliseconds before parsing.
   *
   * @param timestamp - ISO 8601 timestamp string (may include microseconds)
   * @returns Milliseconds since epoch, or 0 if parsing fails
   */
  private _parseTimestampForSort(timestamp: string): number {
    try {
      // Normalize fractional seconds to 3 digits (milliseconds)
      // Match pattern: YYYY-MM-DDTHH:mm:ss.SSSSSS(Z or timezone)
      const normalized = timestamp.replace(
        /(\.\d{3})\d+(Z|[+-]\d{2}:\d{2})$/,
        '$1$2'
      )
      const time = new Date(normalized).getTime()
      // Return 0 if parsing failed (NaN)
      return isNaN(time) ? 0 : time
    } catch {
      return 0
    }
  }
}
