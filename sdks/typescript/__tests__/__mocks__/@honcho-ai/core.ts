// Mock implementation of @honcho-ai/core for testing
export default class MockHonchoCore {
  public workspaces = {
    peers: {
      list: jest.fn(),
      chat: jest.fn(),
      sessions: {
        list: jest.fn(),
      },
      messages: {
        create: jest.fn(),
        list: jest.fn(),
      },
      getOrCreate: jest.fn(),
      update: jest.fn(),
      search: jest.fn(),
      workingRepresentation: jest.fn(),
    },
    sessions: {
      list: jest.fn(),
      peers: {
        add: jest.fn(),
        set: jest.fn(),
        remove: jest.fn(),
        list: jest.fn(),
      },
      messages: {
        create: jest.fn(),
        list: jest.fn(),
      },
      getOrCreate: jest.fn(),
      update: jest.fn(),
      getContext: jest.fn(),
      search: jest.fn(),
    },
    getOrCreate: jest.fn().mockResolvedValue({ id: 'test-workspace', metadata: {} }),
    update: jest.fn(),
    list: jest.fn(),
    search: jest.fn(),
  };

  constructor(options?: any) {
    // Mock constructor
  }
} 