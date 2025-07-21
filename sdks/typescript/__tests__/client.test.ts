import { Honcho } from '../src/client';
import { Peer } from '../src/peer';
import { Session } from '../src/session';
import { Page } from '../src/pagination';

// Mock the @honcho-ai/core module
jest.mock('@honcho-ai/core', () => {
  return jest.fn().mockImplementation(() => ({
    workspaces: {
      peers: {
        list: jest.fn(),
      },
      sessions: {
        list: jest.fn(),
      },
      getOrCreate: jest.fn().mockResolvedValue({ id: 'test-workspace', metadata: {} }),
      update: jest.fn(),
      list: jest.fn(),
      search: jest.fn(),
    },
  }));
});

describe('Honcho Client', () => {
  let honcho: Honcho;
  let mockClient: any;

  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();

    honcho = new Honcho({
      workspaceId: 'test-workspace',
      apiKey: 'test-key',
      environment: 'local',
    });

    mockClient = (honcho as any)._client;
  });

  describe('constructor', () => {
    it('should initialize with provided options', () => {
      const client = new Honcho({
        workspaceId: 'custom-workspace',
        apiKey: 'custom-key',
        environment: 'production',
        baseURL: 'https://custom-url.com',
        timeout: 5000,
        maxRetries: 3,
      });

      expect(client.workspaceId).toBe('custom-workspace');
    });

    it('should use environment variables as fallbacks', () => {
      process.env.HONCHO_WORKSPACE_ID = 'env-workspace';
      process.env.HONCHO_API_KEY = 'env-key';
      process.env.HONCHO_URL = 'https://env-url.com';

      const client = new Honcho({});

      expect(client.workspaceId).toBe('env-workspace');

      // Clean up environment variables
      delete process.env.HONCHO_WORKSPACE_ID;
      delete process.env.HONCHO_API_KEY;
      delete process.env.HONCHO_URL;
    });

    it('should use default workspace ID when none provided', () => {
      const client = new Honcho({});
      expect(client.workspaceId).toBe('default');
    });

    it('should handle all constructor options', () => {
      const client = new Honcho({
        workspaceId: 'test',
        apiKey: 'key',
        environment: 'local',
        baseURL: 'https://example.com',
        timeout: 10000,
        maxRetries: 5,
        defaultHeaders: { 'X-Custom': 'header' },
        defaultQuery: { param: 'value' },
      });

      expect(client.workspaceId).toBe('test');
    });
  });

  describe('peer', () => {
    it('should create a new Peer instance', () => {
      const peer = honcho.peer('test-peer');

      expect(peer).toBeInstanceOf(Peer);
      expect(peer.id).toBe('test-peer');
    });

    it('should throw error for empty peer ID', () => {
      expect(() => honcho.peer('')).toThrow('Peer ID must be a non-empty string');
    });

    it('should throw error for non-string peer ID', () => {
      expect(() => honcho.peer(null as any)).toThrow('Peer ID must be a non-empty string');
      expect(() => honcho.peer(undefined as any)).toThrow('Peer ID must be a non-empty string');
      expect(() => honcho.peer(123 as any)).toThrow('Peer ID must be a non-empty string');
    });
  });

  describe('getPeers', () => {
    it('should return a Page of Peer instances', async () => {
      const mockPeersData = {
        items: [
          { id: 'peer1', metadata: {} },
          { id: 'peer2', metadata: {} },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.list.mockResolvedValue(mockPeersData);

      const peersPage = await honcho.getPeers();

      expect(peersPage).toBeInstanceOf(Page);
      expect(mockClient.workspaces.peers.list).toHaveBeenCalledWith('test-workspace', { filter: undefined });
    });

    it('should handle empty peers list', async () => {
      const mockPeersData = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.list.mockResolvedValue(mockPeersData);

      const peersPage = await honcho.getPeers();

      expect(peersPage).toBeInstanceOf(Page);
      expect(mockClient.workspaces.peers.list).toHaveBeenCalledWith('test-workspace', { filter: undefined });
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.list.mockRejectedValue(new Error('API Error'));

      await expect(honcho.getPeers()).rejects.toThrow('API Error');
    });
  });

  describe('session', () => {
    it('should create a new Session instance', () => {
      const session = honcho.session('test-session');

      expect(session).toBeInstanceOf(Session);
      expect(session.id).toBe('test-session');
    });

    it('should throw error for empty session ID', () => {
      expect(() => honcho.session('')).toThrow('Session ID must be a non-empty string');
    });

    it('should throw error for non-string session ID', () => {
      expect(() => honcho.session(null as any)).toThrow('Session ID must be a non-empty string');
      expect(() => honcho.session(undefined as any)).toThrow('Session ID must be a non-empty string');
      expect(() => honcho.session(123 as any)).toThrow('Session ID must be a non-empty string');
    });
  });

  describe('getSessions', () => {
    it('should return a Page of Session instances', async () => {
      const mockSessionsData = {
        items: [
          { id: 'session1', metadata: {} },
          { id: 'session2', metadata: {} },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };
      mockClient.workspaces.sessions.list.mockResolvedValue(mockSessionsData);

      const sessionsPage = await honcho.getSessions();

      expect(sessionsPage).toBeInstanceOf(Page);
      expect(mockClient.workspaces.sessions.list).toHaveBeenCalledWith('test-workspace', { filter: undefined });
    });

    it('should handle empty sessions list', async () => {
      const mockSessionsData = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.sessions.list.mockResolvedValue(mockSessionsData);

      const sessionsPage = await honcho.getSessions();

      expect(sessionsPage).toBeInstanceOf(Page);
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.list.mockRejectedValue(new Error('API Error'));

      await expect(honcho.getSessions()).rejects.toThrow('API Error');
    });
  });

  describe('getMetadata', () => {
    it('should return workspace metadata', async () => {
      const mockWorkspace = {
        id: 'test-workspace',
        metadata: { key: 'value', setting: 'config' },
      };
      mockClient.workspaces.getOrCreate.mockResolvedValue(mockWorkspace);

      const metadata = await honcho.getMetadata();

      expect(metadata).toEqual({ key: 'value', setting: 'config' });
      expect(mockClient.workspaces.getOrCreate).toHaveBeenCalledWith({ id: 'test-workspace' });
    });

    it('should return empty object when no metadata exists', async () => {
      const mockWorkspace = {
        id: 'test-workspace',
        metadata: null,
      };
      mockClient.workspaces.getOrCreate.mockResolvedValue(mockWorkspace);

      const metadata = await honcho.getMetadata();

      expect(metadata).toEqual({});
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.getOrCreate.mockRejectedValue(new Error('Workspace not found'));

      await expect(honcho.getMetadata()).rejects.toThrow('Workspace not found');
    });
  });

  describe('setMetadata', () => {
    it('should update workspace metadata', async () => {
      const metadata = { newKey: 'newValue', updated: true };
      mockClient.workspaces.update.mockResolvedValue({});

      await honcho.setMetadata(metadata);

      expect(mockClient.workspaces.update).toHaveBeenCalledWith('test-workspace', { metadata });
    });

    it('should handle empty metadata object', async () => {
      mockClient.workspaces.update.mockResolvedValue({});

      await honcho.setMetadata({});

      expect(mockClient.workspaces.update).toHaveBeenCalledWith('test-workspace', { metadata: {} });
    });

    it('should handle complex metadata objects', async () => {
      const complexMetadata = {
        nested: { object: { with: 'values' } },
        array: [1, 2, 3],
        boolean: true,
        number: 42,
        string: 'test',
      };
      mockClient.workspaces.update.mockResolvedValue({});

      await honcho.setMetadata(complexMetadata);

      expect(mockClient.workspaces.update).toHaveBeenCalledWith('test-workspace', { metadata: complexMetadata });
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.update.mockRejectedValue(new Error('Update failed'));

      await expect(honcho.setMetadata({ key: 'value' })).rejects.toThrow('Update failed');
    });
  });

  describe('getWorkspaces', () => {
    it('should return array of workspace IDs', async () => {
      const mockWorkspacesPage = {
        [Symbol.asyncIterator]: async function* () {
          yield { id: 'workspace1' };
          yield { id: 'workspace2' };
          yield { id: 'workspace3' };
        },
      };
      mockClient.workspaces.list.mockResolvedValue(mockWorkspacesPage);

      const workspaces = await honcho.getWorkspaces();

      expect(workspaces).toEqual(['workspace1', 'workspace2', 'workspace3']);
      expect(mockClient.workspaces.list).toHaveBeenCalled();
    });

    it('should handle empty workspaces list', async () => {
      const mockWorkspacesPage = {
        [Symbol.asyncIterator]: async function* () {
          // Empty iterator
        },
      };
      mockClient.workspaces.list.mockResolvedValue(mockWorkspacesPage);

      const workspaces = await honcho.getWorkspaces();

      expect(workspaces).toEqual([]);
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.list.mockRejectedValue(new Error('Failed to list workspaces'));

      await expect(honcho.getWorkspaces()).rejects.toThrow('Failed to list workspaces');
    });
  });

  describe('search', () => {
    it('should search for messages and return Page', async () => {
      const mockSearchResults = {
        items: [
          { id: 'msg1', content: 'Hello world', peer_id: 'peer1' },
          { id: 'msg2', content: 'Hello there', peer_id: 'peer2' },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };
      mockClient.workspaces.search.mockResolvedValue(mockSearchResults);

      const results = await honcho.search('hello');

      expect(results).toBeInstanceOf(Page);
      expect(mockClient.workspaces.search).toHaveBeenCalledWith('test-workspace', { body: 'hello' });
    });

    it('should handle empty search results', async () => {
      const mockSearchResults = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.search.mockResolvedValue(mockSearchResults);

      const results = await honcho.search('nonexistent');

      expect(results).toBeInstanceOf(Page);
    });

    it('should throw error for empty query', async () => {
      await expect(honcho.search('')).rejects.toThrow('Search query must be a non-empty string');
      await expect(honcho.search('   ')).rejects.toThrow('Search query must be a non-empty string');
    });

    it('should throw error for non-string query', async () => {
      await expect(honcho.search(null as any)).rejects.toThrow('Search query must be a non-empty string');
      await expect(honcho.search(undefined as any)).rejects.toThrow('Search query must be a non-empty string');
      await expect(honcho.search(123 as any)).rejects.toThrow('Search query must be a non-empty string');
    });

    it('should handle complex search queries', async () => {
      const mockSearchResults = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.search.mockResolvedValue(mockSearchResults);

      const complexQuery = 'complex query with "quotes" and special characters!@#$%';
      await honcho.search(complexQuery);

      expect(mockClient.workspaces.search).toHaveBeenCalledWith('test-workspace', { body: complexQuery });
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.search.mockRejectedValue(new Error('Search failed'));

      await expect(honcho.search('test')).rejects.toThrow('Search failed');
    });
  });
});
