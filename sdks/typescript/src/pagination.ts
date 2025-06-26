/**
 * Generic paginated result wrapper for Honcho SDK.
 * Provides async iteration and transformation capabilities.
 */
export class Page<T> implements AsyncIterable<T> {
  private _originalPage: any;
  private _transformFunc?: (item: any) => T;

  /**
   * Initialize a new Page.
   */
  constructor(originalPage: any, transformFunc?: (item: any) => T) {
    this._originalPage = originalPage;
    this._transformFunc = transformFunc;
  }

  /**
   * Async iterator for the page's items.
   */
  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    // Handle different page structure formats
    const items = this._originalPage.items || this._originalPage.data || [];
    for (const item of items) {
      yield this._transformFunc ? this._transformFunc(item) : item;
    }
  }

  /**
   * Get an item by index.
   */
  async get(index: number): Promise<T> {
    if (!this._originalPage?.get || typeof this._originalPage.get !== 'function') {
      throw new Error('Original page does not support indexed access');
    }
    const item = await this._originalPage.get(index);
    return this._transformFunc ? this._transformFunc(item) : item;
  }

  /**
   * Get the size of the page.
   */
  get size(): number {
    return this._originalPage?.size ?? 0;
  }

  /**
   * Get the total number of items.
   */
  get total(): number {
    return this._originalPage?.total ?? 0;
  }

  /**
   * Get all data as a list.
   */
  async data(): Promise<T[]> {
    const items = this._originalPage.items || this._originalPage.data || [];
    const data = typeof items === 'function' ? await items() : items;
    return this._transformFunc ? data.map(this._transformFunc) : data;
  }

  /**
   * Check if there's a next page.
   */
  get hasNextPage(): boolean {
    return this._originalPage?.hasNextPage ?? false;
  }

  /**
   * Get the next page.
   */
  async nextPage(): Promise<Page<T> | null> {
    const nextPage = await this._originalPage.nextPage();
    if (!nextPage) return null;
    return new Page(nextPage, this._transformFunc);
  }
} 