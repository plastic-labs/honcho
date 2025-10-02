import { Page } from '../src/pagination'

describe('Page', () => {
  let mockOriginalPage: any
  let mockItems: any[]

  beforeEach(() => {
    mockItems = [
      { id: 'item1', name: 'Item 1' },
      { id: 'item2', name: 'Item 2' },
      { id: 'item3', name: 'Item 3' },
    ]

    mockOriginalPage = {
      items: mockItems,
      size: 3,
      total: 10,
      page: 1,
      pages: 4,
      hasNextPage: true,
      [Symbol.asyncIterator]: async function*() {
        for (const item of mockItems) {
          yield item
        }
      },
    } as any
  })

  describe('constructor', () => {
    it('should initialize with original page', () => {
      const page = new Page(mockOriginalPage)

      expect(page['_originalPage']).toBe(mockOriginalPage)
      expect(page['_transformFunc']).toBeUndefined()
    })

    it('should initialize with transform function', () => {
      const transformFunc = (item: any) => ({ ...item, transformed: true })
      const page = new Page(mockOriginalPage, transformFunc)

      expect(page['_originalPage']).toBe(mockOriginalPage)
      expect(page['_transformFunc']).toBe(transformFunc)
    })
  })

  describe('Symbol.asyncIterator', () => {
    it('should iterate through items without transform', async () => {
      const page = new Page(mockOriginalPage)
      const items: any[] = []

      for await (const item of page) {
        items.push(item)
      }

      expect(items).toEqual(mockItems)
    })

    it('should iterate through items with transform', async () => {
      const transformFunc = (item: any) => ({ ...item, transformed: true })
      const page = new Page(mockOriginalPage, transformFunc)
      const items: any[] = []

      for await (const item of page) {
        items.push(item)
      }

      expect(items).toEqual([
        { id: 'item1', name: 'Item 1', transformed: true },
        { id: 'item2', name: 'Item 2', transformed: true },
        { id: 'item3', name: 'Item 3', transformed: true },
      ])
    })

    it('should handle transform function that throws error', async () => {
      const errorTransform = () => {
        throw new Error('Transform error')
      }
      const page = new Page(mockOriginalPage, errorTransform)

      const iterate = async () => {
        for await (const item of page) {
          // This should throw
        }
      }

      await expect(iterate()).rejects.toThrow('Transform error')
    })
  })

  describe('get', () => {
    it('should get item by index without transform', () => {
      const page = new Page(mockOriginalPage)

      const item = page.get(1)

      expect(item).toEqual(mockItems[1])
    })

    it('should get item by index with transform', () => {
      const transformFunc = (item: any) => ({ ...item, transformed: true })
      const page = new Page(mockOriginalPage, transformFunc)

      const item = page.get(1)

      expect(item).toEqual({ ...mockItems[1], transformed: true })
    })

    it('should handle out of bounds index', () => {
      const page = new Page(mockOriginalPage)

      expect(() => page.get(999)).toThrow(
        'Index 999 is out of bounds for page with 3 items'
      )
    })
  })

  describe('length getter', () => {
    it('should return length of items array', () => {
      const page = new Page(mockOriginalPage)

      expect(page.length).toBe(3)
    })

    it('should handle empty items array', () => {
      const emptyPage = { ...mockOriginalPage, items: [] }
      const page = new Page(emptyPage)

      expect(page.length).toBe(0)
    })

    it('should handle undefined items', () => {
      const noItemsPage = { ...mockOriginalPage, items: undefined }
      const page = new Page(noItemsPage)

      expect(page.length).toBe(0)
    })
  })

  describe('items getter', () => {
    it('should return items array without transform', () => {
      const page = new Page(mockOriginalPage)

      const data = page.items

      expect(data).toEqual(mockItems)
    })

    it('should return items array with transform', () => {
      const transformFunc = (item: any) => ({ ...item, transformed: true })
      const page = new Page(mockOriginalPage, transformFunc)

      const data = page.items

      expect(data).toEqual([
        { id: 'item1', name: 'Item 1', transformed: true },
        { id: 'item2', name: 'Item 2', transformed: true },
        { id: 'item3', name: 'Item 3', transformed: true },
      ])
    })

    it('should handle transform function returning null', () => {
      const nullTransform = () => null
      const page = new Page(mockOriginalPage, nullTransform)

      const data = page.items

      expect(data).toEqual([null, null, null])
    })
  })

  describe('pagination metadata getters', () => {
    it('should return total from original page', () => {
      const page = new Page(mockOriginalPage)

      expect(page.total).toBe(10)
    })

    it('should return page number from original page', () => {
      const page = new Page(mockOriginalPage)

      expect(page.page).toBe(1)
    })

    it('should return size from original page', () => {
      const page = new Page(mockOriginalPage)

      expect(page.size).toBe(3)
    })

    it('should return pages from original page', () => {
      const page = new Page(mockOriginalPage)

      expect(page.pages).toBe(4)
    })

    it('should handle undefined metadata', () => {
      const minimalPage = { items: mockItems }
      const page = new Page(minimalPage as any)

      expect(page.total).toBeUndefined()
      expect(page.page).toBeUndefined()
      expect(page.size).toBeUndefined()
      expect(page.pages).toBeUndefined()
    })
  })

  describe('hasNextPage getter', () => {
    it('should return true when hasNextPage is true', () => {
      const page = new Page(mockOriginalPage)

      expect(page.hasNextPage).toBe(true)
    })

    it('should return false when hasNextPage is false', () => {
      const lastPage = { ...mockOriginalPage, hasNextPage: false }
      const page = new Page(lastPage)

      expect(page.hasNextPage).toBe(false)
    })

    it('should call hasNextPage function if it is a function', () => {
      const hasNextPageFn = jest.fn(() => true)
      const pageWithFn = { ...mockOriginalPage, hasNextPage: hasNextPageFn }
      const page = new Page(pageWithFn)

      const result = page.hasNextPage

      expect(result).toBe(true)
      expect(hasNextPageFn).toHaveBeenCalled()
    })

    it('should return false when hasNextPage is undefined', () => {
      const noNextPage = { ...mockOriginalPage }
      delete noNextPage.hasNextPage
      const page = new Page(noNextPage)

      expect(page.hasNextPage).toBe(false)
    })
  })

  describe('getNextPage', () => {
    it('should return next page with same transform function', async () => {
      const nextPageData = {
        items: [{ id: 'item4', name: 'Item 4' }],
        size: 1,
        total: 10,
        page: 2,
        pages: 4,
        hasNextPage: false,
        [Symbol.asyncIterator]: async function*() {
          for (const item of this.items) {
            yield item
          }
        },
      }
      const transformFunc = (item: any) => ({ ...item, transformed: true })
      mockOriginalPage.getNextPage = jest.fn().mockResolvedValue(nextPageData)
      const page = new Page(mockOriginalPage, transformFunc)

      const nextPage = await page.getNextPage()

      expect(nextPage).toBeInstanceOf(Page)
      expect(nextPage!['_transformFunc']).toBe(transformFunc)
      expect(mockOriginalPage.getNextPage).toHaveBeenCalled()
    })

    it('should return null when no next page', async () => {
      mockOriginalPage.getNextPage = jest.fn().mockResolvedValue(null)
      const page = new Page(mockOriginalPage)

      const nextPage = await page.getNextPage()

      expect(nextPage).toBeNull()
    })

    it('should return null when getNextPage returns undefined', async () => {
      mockOriginalPage.getNextPage = jest.fn().mockResolvedValue(undefined)
      const page = new Page(mockOriginalPage)

      const nextPage = await page.getNextPage()

      expect(nextPage).toBeNull()
    })

    it('should return null when getNextPage method does not exist', async () => {
      const pageWithoutNext = { ...mockOriginalPage }
      delete pageWithoutNext.getNextPage
      const page = new Page(pageWithoutNext)

      const nextPage = await page.getNextPage()

      expect(nextPage).toBeNull()
    })

    it('should propagate transform function to next page', async () => {
      const nextPageData = {
        items: [{ id: 'item4', name: 'Item 4' }],
        [Symbol.asyncIterator]: async function*() {
          for (const item of this.items) {
            yield item
          }
        },
      }
      const transformFunc = (item: any) => ({ ...item, count: 999 })
      mockOriginalPage.getNextPage = jest.fn().mockResolvedValue(nextPageData)
      const page = new Page(mockOriginalPage, transformFunc)

      const nextPage = await page.getNextPage()

      expect(nextPage).not.toBeNull()
      const transformedItem = nextPage!.get(0)
      expect(transformedItem).toEqual({
        id: 'item4',
        name: 'Item 4',
        count: 999,
      })
    })

    it('should handle error from getNextPage', async () => {
      mockOriginalPage.getNextPage = jest
        .fn()
        .mockRejectedValue(new Error('Failed to get next page'))
      const page = new Page(mockOriginalPage)

      await expect(page.getNextPage()).rejects.toThrow(
        'Failed to get next page'
      )
    })
  })
})
