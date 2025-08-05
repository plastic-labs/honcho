import { Page } from '../src/pagination';

describe('Page', () => {
  let mockOriginalPage: any;
  let mockItems: any[];

  beforeEach(() => {
    mockItems = [
      { id: 'item1', name: 'Item 1' },
      { id: 'item2', name: 'Item 2' },
      { id: 'item3', name: 'Item 3' },
    ];

    mockOriginalPage = {
      items: mockItems,
      data: mockItems,
      size: 3,
      total: 10,
      hasNextPage: true,
      get: jest.fn(),
      nextPage: jest.fn(),
    };
  });

  describe('constructor', () => {
    it('should initialize with original page', () => {
      const page = new Page(mockOriginalPage);

      expect(page['_originalPage']).toBe(mockOriginalPage);
      expect(page['_transformFunc']).toBeUndefined();
    });

    it('should initialize with transform function', () => {
      const transformFunc = (item: any) => ({ ...item, transformed: true });
      const page = new Page(mockOriginalPage, transformFunc);

      expect(page['_originalPage']).toBe(mockOriginalPage);
      expect(page['_transformFunc']).toBe(transformFunc);
    });
  });

  describe('Symbol.asyncIterator', () => {
    it('should iterate through items without transform', async () => {
      const page = new Page(mockOriginalPage);
      const items: any[] = [];

      for await (const item of page) {
        items.push(item);
      }

      expect(items).toEqual(mockItems);
    });

    it('should iterate through items with transform', async () => {
      const transformFunc = (item: any) => ({ ...item, transformed: true });
      const page = new Page(mockOriginalPage, transformFunc);
      const items: any[] = [];

      for await (const item of page) {
        items.push(item);
      }

      expect(items).toEqual([
        { id: 'item1', name: 'Item 1', transformed: true },
        { id: 'item2', name: 'Item 2', transformed: true },
        { id: 'item3', name: 'Item 3', transformed: true },
      ]);
    });

    it('should handle empty items array', async () => {
      const emptyPage = { items: [], data: [], size: 0, total: 0, hasNextPage: false };
      const page = new Page(emptyPage);
      const items: any[] = [];

      for await (const item of page) {
        items.push(item);
      }

      expect(items).toEqual([]);
    });

    it('should handle page with data field instead of items', async () => {
      const pageWithData = {
        data: mockItems,
        size: 3,
        total: 10,
        hasNextPage: true,
      };
      const page = new Page(pageWithData);
      const items: any[] = [];

      for await (const item of page) {
        items.push(item);
      }

      expect(items).toEqual(mockItems);
    });

    it('should handle page with neither items nor data', async () => {
      const pageWithoutItems = {
        size: 0,
        total: 0,
        hasNextPage: false,
      };
      const page = new Page(pageWithoutItems);
      const items: any[] = [];

      for await (const item of page) {
        items.push(item);
      }

      expect(items).toEqual([]);
    });
  });

  describe('get', () => {
    it('should get item by index without transform', async () => {
      mockOriginalPage.get.mockResolvedValue(mockItems[1]);
      const page = new Page(mockOriginalPage);

      const item = await page.get(1);

      expect(item).toEqual(mockItems[1]);
      expect(mockOriginalPage.get).toHaveBeenCalledWith(1);
    });

    it('should get item by index with transform', async () => {
      const transformFunc = (item: any) => ({ ...item, transformed: true });
      mockOriginalPage.get.mockResolvedValue(mockItems[1]);
      const page = new Page(mockOriginalPage, transformFunc);

      const item = await page.get(1);

      expect(item).toEqual({ ...mockItems[1], transformed: true });
      expect(mockOriginalPage.get).toHaveBeenCalledWith(1);
    });

    it('should handle out of bounds index', async () => {
      mockOriginalPage.get.mockResolvedValue(undefined);
      const page = new Page(mockOriginalPage);

      const item = await page.get(999);

      expect(item).toBeUndefined();
      expect(mockOriginalPage.get).toHaveBeenCalledWith(999);
    });

    it('should handle negative index', async () => {
      mockOriginalPage.get.mockResolvedValue(undefined);
      const page = new Page(mockOriginalPage);

      const item = await page.get(-1);

      expect(item).toBeUndefined();
      expect(mockOriginalPage.get).toHaveBeenCalledWith(-1);
    });
  });

  describe('size getter', () => {
    it('should return size from original page', () => {
      const page = new Page(mockOriginalPage);

      expect(page.size).toBe(3);
    });

    it('should handle missing size', () => {
      const pageWithoutSize = { items: mockItems };
      const page = new Page(pageWithoutSize);

      expect(page.size).toBeUndefined();
    });
  });

  describe('total getter', () => {
    it('should return total from original page', () => {
      const page = new Page(mockOriginalPage);

      expect(page.total).toBe(10);
    });

    it('should handle missing total', () => {
      const pageWithoutTotal = { items: mockItems };
      const page = new Page(pageWithoutTotal);

      expect(page.total).toBeUndefined();
    });
  });

  describe('data', () => {
    it('should return data array without transform', async () => {
      const page = new Page(mockOriginalPage);

      const data = await page.data();

      expect(data).toEqual(mockItems);
    });

    it('should return data array with transform', async () => {
      const transformFunc = (item: any) => ({ ...item, transformed: true });
      const page = new Page(mockOriginalPage, transformFunc);

      const data = await page.data();

      expect(data).toEqual([
        { id: 'item1', name: 'Item 1', transformed: true },
        { id: 'item2', name: 'Item 2', transformed: true },
        { id: 'item3', name: 'Item 3', transformed: true },
      ]);
    });

    it('should handle data as function', async () => {
      const mockDataFunction = jest.fn().mockResolvedValue(mockItems);
      const pageWithDataFunction = {
        data: mockDataFunction,
        size: 3,
        total: 10,
        hasNextPage: true,
      };
      const page = new Page(pageWithDataFunction);

      const data = await page.data();

      expect(data).toEqual(mockItems);
      expect(mockDataFunction).toHaveBeenCalled();
    });

    it('should handle items as function', async () => {
      const mockItemsFunction = jest.fn().mockResolvedValue(mockItems);
      const pageWithItemsFunction = {
        items: mockItemsFunction,
        size: 3,
        total: 10,
        hasNextPage: true,
      };
      const page = new Page(pageWithItemsFunction);

      const data = await page.data();

      expect(data).toEqual(mockItems);
      expect(mockItemsFunction).toHaveBeenCalled();
    });

    it('should handle empty data', async () => {
      const emptyPage = { items: [], size: 0, total: 0, hasNextPage: false };
      const page = new Page(emptyPage);

      const data = await page.data();

      expect(data).toEqual([]);
    });

    it('should prioritize items over data field', async () => {
      const pageWithBoth = {
        items: [{ id: 'from-items' }],
        data: [{ id: 'from-data' }],
        size: 1,
        total: 1,
        hasNextPage: false,
      };
      const page = new Page(pageWithBoth);

      const data = await page.data();

      expect(data).toEqual([{ id: 'from-items' }]);
    });
  });

  describe('hasNextPage getter', () => {
    it('should return hasNextPage from original page', () => {
      const page = new Page(mockOriginalPage);

      expect(page.hasNextPage).toBe(true);
    });

    it('should handle missing hasNextPage', () => {
      const pageWithoutHasNextPage = { items: mockItems };
      const page = new Page(pageWithoutHasNextPage);

      expect(page.hasNextPage).toBeUndefined();
    });

    it('should handle false hasNextPage', () => {
      const lastPage = { ...mockOriginalPage, hasNextPage: false };
      const page = new Page(lastPage);

      expect(page.hasNextPage).toBe(false);
    });
  });

  describe('nextPage', () => {
    it('should return next page with same transform function', async () => {
      const nextPageData = {
        items: [{ id: 'item4', name: 'Item 4' }],
        size: 1,
        total: 10,
        hasNextPage: false,
      };
      const transformFunc = (item: any) => ({ ...item, transformed: true });
      mockOriginalPage.nextPage.mockResolvedValue(nextPageData);
      const page = new Page(mockOriginalPage, transformFunc);

      const nextPage = await page.nextPage();

      expect(nextPage).toBeInstanceOf(Page);
      expect(nextPage!['_transformFunc']).toBe(transformFunc);
      expect(mockOriginalPage.nextPage).toHaveBeenCalled();
    });

    it('should return null when no next page', async () => {
      mockOriginalPage.nextPage.mockResolvedValue(null);
      const page = new Page(mockOriginalPage);

      const nextPage = await page.nextPage();

      expect(nextPage).toBeNull();
    });

    it('should handle next page returning undefined', async () => {
      mockOriginalPage.nextPage.mockResolvedValue(undefined);
      const page = new Page(mockOriginalPage);

      const nextPage = await page.nextPage();

      expect(nextPage).toBeNull();
    });

    it('should handle error from original page nextPage', async () => {
      mockOriginalPage.nextPage.mockRejectedValue(new Error('Failed to get next page'));
      const page = new Page(mockOriginalPage);

      await expect(page.nextPage()).rejects.toThrow();
    });
  });

  describe('edge cases and error handling', () => {
    it('should handle null original page', () => {
      const page = new Page(null);

      expect(page['_originalPage']).toBeNull();
      expect(page.size).toBeUndefined();
      expect(page.total).toBeUndefined();
      expect(page.hasNextPage).toBeUndefined();
    });

    it('should handle transform function that throws error', async () => {
      const errorTransform = () => {
        throw new Error('Transform error');
      };
      const page = new Page(mockOriginalPage, errorTransform);

      await expect(async () => {
        for await (const item of page) {
          // This should throw
        }
      }).rejects.toThrow();
    });

    it('should handle transform function returning null', async () => {
      const nullTransform = () => null;
      const page = new Page(mockOriginalPage, nullTransform);
      const items: any[] = [];

      for await (const item of page) {
        items.push(item);
      }

      expect(items).toEqual([null, null, null]);
    });

    it('should handle very large datasets', async () => {
      const largeItems = Array.from({ length: 10000 }, (_, i) => ({ id: i }));
      const largePage = {
        items: largeItems,
        size: 10000,
        total: 10000,
        hasNextPage: false,
      };
      const page = new Page(largePage);
      let count = 0;

      for await (const item of page) {
        count++;
        if (count > 10) break; // Don't actually iterate through all 10k items
      }

      expect(count).toBe(11);
    });

    it('should handle circular reference in items', async () => {
      const circularItem: any = { id: 'circular' };
      circularItem.self = circularItem;
      const pageWithCircular = {
        items: [circularItem],
        size: 1,
        total: 1,
        hasNextPage: false,
      };
      const page = new Page(pageWithCircular);
      const items: any[] = [];

      for await (const item of page) {
        items.push(item);
      }

      expect(items).toHaveLength(1);
      expect(items[0].id).toBe('circular');
      expect(items[0].self).toBe(items[0]);
    });

    it('should handle items with complex nested structures', async () => {
      const complexItems = [
        {
          id: 'complex1',
          nested: {
            deep: {
              value: 'deeply nested',
              array: [1, 2, { nested: 'array object' }],
            },
          },
          metadata: new Map([['key', 'value']]),
        },
      ];
      const complexPage = {
        items: complexItems,
        size: 1,
        total: 1,
        hasNextPage: false,
      };
      const page = new Page(complexPage);
      const items: any[] = [];

      for await (const item of page) {
        items.push(item);
      }

      expect(items).toEqual(complexItems);
      expect(items[0].nested.deep.array[2].nested).toBe('array object');
    });

    it('should handle async transform function', async () => {
      const asyncTransform = async (item: any) => {
        await new Promise(resolve => setTimeout(resolve, 1));
        return { ...item, asyncTransformed: true };
      };
      mockOriginalPage.get.mockResolvedValue(mockItems[0]);
      const page = new Page(mockOriginalPage, asyncTransform);

      const item = await page.get(0);

      expect(item).toEqual({ ...mockItems[0], asyncTransformed: true });
    });
  });
});
