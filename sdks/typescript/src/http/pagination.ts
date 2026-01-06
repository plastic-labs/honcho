export interface PageData<T> {
  items: T[]
  total: number | null
  page: number
  size: number
  pages: number | null
}

/**
 * Paginated result wrapper with async iteration support.
 *
 * Supports transformation of items and on-demand page fetching.
 */
export class Page<T, U = T> implements AsyncIterable<U> {
  private _items: T[]
  private _total: number | null
  private _page: number
  private _size: number
  private _pages: number | null
  private _transformFunc?: (item: T) => U
  private _fetchNext?: () => Promise<Page<T, U>>

  constructor(
    data: PageData<T>,
    transformFunc?: (item: T) => U,
    fetchNext?: () => Promise<Page<T, U>>
  ) {
    this._items = data.items
    this._total = data.total
    this._page = data.page
    this._size = data.size
    this._pages = data.pages
    this._transformFunc = transformFunc
    this._fetchNext = fetchNext
  }

  /**
   * Get transformed items on the current page.
   */
  get items(): U[] {
    if (this._transformFunc) {
      return this._items.map(this._transformFunc)
    }
    return this._items as unknown as U[]
  }

  /**
   * Total number of items across all pages.
   */
  get total(): number | null {
    return this._total
  }

  /**
   * Current page number (1-indexed).
   */
  get page(): number {
    return this._page
  }

  /**
   * Number of items per page.
   */
  get size(): number {
    return this._size
  }

  /**
   * Total number of pages.
   */
  get pages(): number | null {
    return this._pages
  }

  /**
   * Number of items on the current page.
   */
  get length(): number {
    return this._items.length
  }

  /**
   * Get item by index on the current page.
   */
  get(index: number): U {
    const item = this._items[index]
    if (this._transformFunc) {
      return this._transformFunc(item)
    }
    return item as unknown as U
  }

  /**
   * Check if there's a next page.
   */
  get hasNextPage(): boolean {
    if (this._pages !== null) {
      return this._page < this._pages
    }
    // If we don't know total pages, check if we got a full page
    return this._items.length >= this._size
  }

  /**
   * Fetch the next page of results.
   */
  async getNextPage(): Promise<Page<T, U> | null> {
    if (!this.hasNextPage || !this._fetchNext) {
      return null
    }
    return this._fetchNext()
  }

  /**
   * Async iterate over all items across all pages.
   */
  async *[Symbol.asyncIterator](): AsyncIterator<U> {
    let current: Page<T, U> | null = this
    while (current !== null) {
      for (const item of current._items) {
        if (this._transformFunc) {
          yield this._transformFunc(item)
        } else {
          yield item as unknown as U
        }
      }
      current = await current.getNextPage()
    }
  }
}
