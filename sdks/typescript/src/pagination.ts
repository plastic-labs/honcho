import type { CursorPageResponse, PageResponse, TotalKind } from './types/api'

/**
 * Function type for fetching the next page of results.
 */
export type NextPageFetcher<T> = (
  page: number,
  size: number
) => Promise<PageResponse<T>>

/**
 * Function type that repeats a list request in cursor mode from the given cursor.
 */
export type CursorPageFetcher<T> = (
  cursor: string,
  size: number
) => Promise<CursorPageResponse<T>>

// The server's default PAGINATION_COUNT_LIMIT. A server that predates
// `total_kind` reports a total past it as the limit plus one or as a bound.
const DEFAULT_COUNT_LIMIT = 100_000

/**
 * Generic paginated result wrapper for Honcho SDK.
 * Provides async iteration and transformation capabilities.
 */
export class Page<T, TOriginal = T> implements AsyncIterable<T> {
  private _data: PageResponse<TOriginal>
  private _transformFunc?: (item: TOriginal) => T
  private _fetchNextPage?: NextPageFetcher<TOriginal>
  private _fetchByCursor?: CursorPageFetcher<TOriginal>

  /**
   * Initialize a new Page.
   *
   * @param data - The page response data
   * @param transformFunc - Optional function to transform objects from the original type to type T.
   *                        If not provided, objects are passed through unchanged.
   * @param fetchNextPage - Optional function to fetch the next page of results.
   * @param fetchByCursor - Optional function that repeats the list request in
   *                        cursor mode. Iteration uses it to read past this page.
   */
  constructor(
    data: PageResponse<TOriginal>,
    transformFunc?: (item: TOriginal) => T,
    fetchNextPage?: NextPageFetcher<TOriginal>,
    fetchByCursor?: CursorPageFetcher<TOriginal>
  ) {
    this._data = data
    this._transformFunc = transformFunc
    this._fetchNextPage = fetchNextPage
    this._fetchByCursor = fetchByCursor
  }

  /**
   * Create a Page from raw response data.
   */
  static from<T>(
    data: PageResponse<T>,
    fetchNextPage?: NextPageFetcher<T>,
    fetchByCursor?: CursorPageFetcher<T>
  ): Page<T, T> {
    return new Page(data, undefined, fetchNextPage, fetchByCursor)
  }

  /**
   * Create a Page with a transformation function.
   */
  static fromWithTransform<T, TOriginal>(
    data: PageResponse<TOriginal>,
    transformFunc: (item: TOriginal) => T,
    fetchNextPage?: NextPageFetcher<TOriginal>,
    fetchByCursor?: CursorPageFetcher<TOriginal>
  ): Page<T, TOriginal> {
    return new Page(data, transformFunc, fetchNextPage, fetchByCursor)
  }

  /**
   * Async iterator for all transformed items across all pages.
   *
   * When the server sends `next_cursor`, the pages after this one are read in
   * cursor mode: each request seeks past the last row read and runs no count,
   * and rows added or removed during the walk do not make it skip or repeat
   * rows. Servers that predate `next_cursor` are walked by page number.
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

    if (this._fetchByCursor && this._data.next_cursor !== undefined) {
      let cursor = this._data.next_cursor
      while (cursor) {
        const data = await this._fetchByCursor(cursor, this._data.size)
        if (data.next_page === undefined) {
          throw new Error(
            'The server answered a cursor-mode list request with an offset page'
          )
        }
        for (const item of data.items) {
          yield this._transformFunc
            ? this._transformFunc(item)
            : (item as unknown as T)
        }
        cursor = data.next_page
      }
      return
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
   * What `total` is: `exact`, `estimate`, `at_least` or `at_most` (see
   * `TotalKind`). Display anything but `exact` as approximate. For servers that
   * predate this field, totals past 100,000 (the server's default count limit)
   * are `at_least` and smaller ones `exact`.
   */
  get totalKind(): TotalKind {
    return (
      this._data.total_kind ??
      (this._data.total > DEFAULT_COUNT_LIMIT ? 'at_least' : 'exact')
    )
  }

  /**
   * The cursor that reads the rest of the list in cursor mode (null at the end
   * of the list, or from servers that predate cursor mode).
   */
  get nextCursor(): string | null {
    return this._data.next_cursor ?? null
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

    return new Page(
      nextPageData,
      this._transformFunc,
      this._fetchNextPage,
      this._fetchByCursor
    )
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
