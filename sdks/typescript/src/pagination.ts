import type { PageResponse } from './types/api'

/**
 * Function type for fetching the next page of results.
 */
export type NextPageFetcher<T> = (
  page: number,
  size: number
) => Promise<PageResponse<T>>

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
