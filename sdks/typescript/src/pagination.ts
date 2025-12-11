import type { Page as CorePage } from '@honcho-ai/core/pagination'

/**
 * Generic paginated result wrapper for Honcho SDK.
 * Provides async iteration and transformation capabilities while preserving
 * pagination functionality from the underlying core Page.
 */
// biome-ignore lint/suspicious/noExplicitAny: Generic type parameter with reasonable default for internal transform
export class Page<T, TOriginal = any> implements AsyncIterable<T> {
  private _originalPage: CorePage<TOriginal>
  private _transformFunc?: (item: TOriginal) => T

  /**
   * Initialize a new Page.
   *
   * @param originalPage - The original Page to wrap
   * @param transformFunc - Optional function to transform objects from the original type to type T.
   *                        If not provided, objects are passed through unchanged.
   */
  constructor(
    originalPage: CorePage<TOriginal>,
    transformFunc?: (item: TOriginal) => T
  ) {
    this._originalPage = originalPage
    this._transformFunc = transformFunc
  }

  /**
   * Async iterator for all transformed items across all pages.
   */
  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    for await (const item of this._originalPage) {
      yield this._transformFunc
        ? this._transformFunc(item)
        : (item as unknown as T)
    }
  }

  /**
   * Get a transformed item by index on the current page.
   */
  get(index: number): T {
    const items = this._originalPage.items || []
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
    const items = this._originalPage.items || []
    return items.length
  }

  /**
   * Get all transformed items on the current page.
   */
  get items(): T[] {
    const items = this._originalPage.items || []
    return this._transformFunc
      ? items.map(this._transformFunc)
      : (items as unknown as T[])
  }

  /**
   * Get the total number of items across all pages.
   */
  get total(): number | undefined {
    return this._originalPage?.total
  }

  /**
   * Get the current page number.
   */
  get page(): number | undefined {
    return this._originalPage?.page
  }

  /**
   * Get the page size.
   */
  get size(): number | undefined {
    return this._originalPage?.size
  }

  /**
   * Get the total number of pages.
   */
  get pages(): number | undefined {
    return this._originalPage?.pages
  }

  /**
   * Check if there's a next page.
   */
  get hasNextPage(): boolean {
    const hasNext = this._originalPage.hasNextPage
    if (typeof hasNext === 'function') {
      return hasNext()
    }
    return hasNext || false
  }

  /**
   * Fetch the next page of results.
   * Returns null if there are no more pages.
   */
  async getNextPage(): Promise<Page<T, TOriginal> | null> {
    if (!this._originalPage.getNextPage) {
      return null
    }

    const nextOriginalPage = await this._originalPage.getNextPage()
    if (!nextOriginalPage) {
      return null
    }

    return new Page(nextOriginalPage, this._transformFunc)
  }
}
