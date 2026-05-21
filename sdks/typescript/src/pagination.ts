import type {
  CursorPageResponse,
  PageResponse,
  QueueWorkUnitsResponse,
} from './types/api'

/**
 * Function type for fetching the next page of results.
 */
export type NextPageFetcher<T> = (
  page: number,
  size: number
) => Promise<PageResponse<T>>

/**
 * Function type for fetching a page by cursor token.
 */
export type CursorPageFetcher<T> = (
  cursor: string,
  size?: number
) => Promise<CursorPageResponse<T>>

/**
 * Generic paginated result wrapper for Honcho SDK.
 * Provides async iteration and transformation capabilities.
 */
export class Page<T, TOriginal = T> implements AsyncIterable<T> {
  private _data: PageResponse<TOriginal>
  private _transformFunc?: (item: TOriginal) => T
  private _fetchNextPage?: NextPageFetcher<TOriginal>

  /**
   * Initialize a new Page.
   *
   * @param data - The page response data
   * @param transformFunc - Optional function to transform objects from the original type to type T.
   *                        If not provided, objects are passed through unchanged.
   * @param fetchNextPage - Optional function to fetch the next page of results.
   */
  constructor(
    data: PageResponse<TOriginal>,
    transformFunc?: (item: TOriginal) => T,
    fetchNextPage?: NextPageFetcher<TOriginal>
  ) {
    this._data = data
    this._transformFunc = transformFunc
    this._fetchNextPage = fetchNextPage
  }

  /**
   * Create a Page from raw response data.
   */
  static from<T>(
    data: PageResponse<T>,
    fetchNextPage?: NextPageFetcher<T>
  ): Page<T, T> {
    return new Page(data, undefined, fetchNextPage)
  }

  /**
   * Create a Page with a transformation function.
   */
  static fromWithTransform<T, TOriginal>(
    data: PageResponse<TOriginal>,
    transformFunc: (item: TOriginal) => T,
    fetchNextPage?: NextPageFetcher<TOriginal>
  ): Page<T, TOriginal> {
    return new Page(data, transformFunc, fetchNextPage)
  }

  /**
   * Async iterator for all transformed items across all pages.
   *
   * **Warning:** This iterator automatically fetches ALL subsequent pages as you iterate.
   * For large datasets, this may result in many API calls. If you only need
   * the current page, use the `items` property instead.
   */
  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    // Yield items from current page
    for (const item of this._data.items) {
      yield this._transformFunc
        ? this._transformFunc(item)
        : (item as unknown as T)
    }

    // Fetch and yield items from subsequent pages
    let currentPage: Page<T, TOriginal> | null = this
    while (currentPage.hasNextPage) {
      const nextPage = await currentPage.getNextPage()
      if (!nextPage) break
      currentPage = nextPage
      for (const item of nextPage._data.items) {
        yield nextPage._transformFunc
          ? nextPage._transformFunc(item)
          : (item as unknown as T)
      }
    }
  }

  /**
   * Get a transformed item by index on the current page.
   */
  get(index: number): T {
    const items = this._data.items || []
    if (index < 0 || index >= items.length) {
      throw new RangeError(
        `Index ${index} is out of bounds for page with ${items.length} items`
      )
    }
    const item = items[index]
    return this._transformFunc
      ? this._transformFunc(item)
      : (item as unknown as T)
  }

  /**
   * Get the number of items on the current page.
   */
  get length(): number {
    return this._data.items?.length ?? 0
  }

  /**
   * Get all transformed items on the current page.
   */
  get items(): T[] {
    const items = this._data.items || []
    return this._transformFunc
      ? items.map(this._transformFunc)
      : (items as unknown as T[])
  }

  /**
   * Get the total number of items across all pages.
   */
  get total(): number {
    return this._data.total
  }

  /**
   * Get the current page number (1-indexed).
   */
  get page(): number {
    return this._data.page
  }

  /**
   * Get the page size.
   */
  get size(): number {
    return this._data.size
  }

  /**
   * Get the total number of pages.
   */
  get pages(): number {
    return this._data.pages
  }

  /**
   * Check if there's a next page.
   */
  get hasNextPage(): boolean {
    return this._data.page < this._data.pages
  }

  /**
   * Fetch the next page of results.
   * Returns null if there are no more pages or if no fetch function is provided.
   */
  async getNextPage(): Promise<Page<T, TOriginal> | null> {
    if (!this.hasNextPage || !this._fetchNextPage) {
      return null
    }

    const nextPageData = await this._fetchNextPage(
      this._data.page + 1,
      this._data.size
    )

    return new Page(nextPageData, this._transformFunc, this._fetchNextPage)
  }

  /**
   * Collect all items from all pages into an array.
   */
  async toArray(): Promise<T[]> {
    const allItems: T[] = []
    for await (const item of this) {
      allItems.push(item)
    }
    return allItems
  }
}

/**
 * Cursor-paginated result wrapper. Uses opaque tokens (`nextPage`/`previousPage`)
 * instead of page numbers. Stable across concurrent mutations of the
 * underlying data, unlike offset pagination.
 *
 * **Warning:** The underlying server data may still mutate between page fetches.
 * Iterating across pages snapshots what the server returns per page, but pages
 * may be inconsistent with each other under concurrent processing. Use
 * `.items` to read just the current page if you need a stable view.
 */
export class CursorPage<T, TOriginal = T> implements AsyncIterable<T> {
  protected _data: CursorPageResponse<TOriginal>
  protected _transformFunc?: (item: TOriginal) => T
  protected _fetchNext?: CursorPageFetcher<TOriginal>
  protected _fetchPrevious?: CursorPageFetcher<TOriginal>

  constructor(
    data: CursorPageResponse<TOriginal>,
    transformFunc?: (item: TOriginal) => T,
    fetchNext?: CursorPageFetcher<TOriginal>,
    fetchPrevious?: CursorPageFetcher<TOriginal>
  ) {
    this._data = data
    this._transformFunc = transformFunc
    this._fetchNext = fetchNext
    this._fetchPrevious = fetchPrevious
  }

  /**
   * Async iterator across all pages, auto-following `nextPage` until exhausted.
   * Use `.items` for the current page only.
   */
  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    for (const item of this._data.items) {
      yield this._transformFunc
        ? this._transformFunc(item)
        : (item as unknown as T)
    }

    let currentPage: CursorPage<T, TOriginal> | null = this
    while (currentPage.hasNextPage) {
      const next: CursorPage<T, TOriginal> | null =
        await currentPage.getNextPage()
      if (!next) break
      currentPage = next
      for (const item of next._data.items) {
        yield next._transformFunc
          ? next._transformFunc(item)
          : (item as unknown as T)
      }
    }
  }

  get items(): T[] {
    const items = this._data.items || []
    return this._transformFunc
      ? items.map(this._transformFunc)
      : (items as unknown as T[])
  }

  get length(): number {
    return this._data.items?.length ?? 0
  }

  /** Total items across all pages, when the server populates it. */
  get total(): number | null {
    return this._data.total ?? null
  }

  /** Cursor token that re-fetches the current page. */
  get currentPage(): string | null {
    return this._data.current_page ?? null
  }

  /** Cursor token to re-fetch the current page from the last item. */
  get currentPageBackwards(): string | null {
    return this._data.current_page_backwards ?? null
  }

  /** Cursor token for the next page, or null at the end. */
  get nextPage(): string | null {
    return this._data.next_page ?? null
  }

  /** Cursor token for the previous page, or null at the start. */
  get previousPage(): string | null {
    return this._data.previous_page ?? null
  }

  get hasNextPage(): boolean {
    return this._data.next_page != null
  }

  get hasPreviousPage(): boolean {
    return this._data.previous_page != null
  }

  /** Fetch the next page; null if at end or no fetch callback. */
  async getNextPage(): Promise<CursorPage<T, TOriginal> | null> {
    if (!this._data.next_page || !this._fetchNext) return null
    const data = await this._fetchNext(this._data.next_page)
    return new CursorPage<T, TOriginal>(
      data,
      this._transformFunc,
      this._fetchNext,
      this._fetchPrevious
    )
  }

  /** Fetch the previous page; null if at start or no fetch callback. */
  async getPreviousPage(): Promise<CursorPage<T, TOriginal> | null> {
    if (!this._data.previous_page || !this._fetchPrevious) return null
    const data = await this._fetchPrevious(this._data.previous_page)
    return new CursorPage<T, TOriginal>(
      data,
      this._transformFunc,
      this._fetchNext,
      this._fetchPrevious
    )
  }

  /** Collect items from all pages into an array. */
  async toArray(): Promise<T[]> {
    const allItems: T[] = []
    for await (const item of this) {
      allItems.push(item)
    }
    return allItems
  }
}

/**
 * Cursor page for the /queue/work-units endpoint with envelope extras
 * (`representationBatchMaxTokens`, `flushEnabled`) carrying the server-side
 * deriver threshold configuration.
 */
export class QueueWorkUnitsPage<T, TOriginal = T> extends CursorPage<
  T,
  TOriginal
> {
  protected override _data: QueueWorkUnitsResponse & {
    items: TOriginal[]
  }

  constructor(
    data: QueueWorkUnitsResponse & { items: TOriginal[] },
    transformFunc?: (item: TOriginal) => T,
    fetchNext?: CursorPageFetcher<TOriginal>,
    fetchPrevious?: CursorPageFetcher<TOriginal>
  ) {
    super(data, transformFunc, fetchNext, fetchPrevious)
    this._data = data
  }

  /** DERIVER_REPRESENTATION_BATCH_MAX_TOKENS at the time of the request. */
  get representationBatchMaxTokens(): number {
    return this._data.representation_batch_max_tokens
  }

  /** True when the batch threshold gating is bypassed server-side. */
  get flushEnabled(): boolean {
    return this._data.flush_enabled
  }

  override async getNextPage(): Promise<QueueWorkUnitsPage<
    T,
    TOriginal
  > | null> {
    if (!this._data.next_page || !this._fetchNext) return null
    const data = (await this._fetchNext(
      this._data.next_page
    )) as QueueWorkUnitsResponse & { items: TOriginal[] }
    return new QueueWorkUnitsPage<T, TOriginal>(
      data,
      this._transformFunc,
      this._fetchNext,
      this._fetchPrevious
    )
  }

  override async getPreviousPage(): Promise<QueueWorkUnitsPage<
    T,
    TOriginal
  > | null> {
    if (!this._data.previous_page || !this._fetchPrevious) return null
    const data = (await this._fetchPrevious(
      this._data.previous_page
    )) as QueueWorkUnitsResponse & { items: TOriginal[] }
    return new QueueWorkUnitsPage<T, TOriginal>(
      data,
      this._transformFunc,
      this._fetchNext,
      this._fetchPrevious
    )
  }
}
