import { Honcho } from '../src/client';
import { Peer } from '../src/peer';
import { Session } from '../src/session';
import { Page } from '../src/pagination';
import type { Message } from '@honcho-ai/core/resources/workspaces/sessions/messages';

// Mock the @honcho-ai/core module
jest.mock('@honcho-ai/core', () => {
  return jest.fn().mockImplementation(() => ({
    workspaces: {
      peers: {
        list: jest.fn(),
        getOrCreate: jest.fn(),
      },
      sessions: {
        list: jest.fn(),
        getOrCreate: jest.fn(),
      },
      getOrCreate: jest.fn().mockResolvedValue({ id: 'test-workspace', metadata: {} }),
      update: jest.fn(),
      list: jest.fn(),
      search: jest.fn(),
      deriverStatus: jest.fn(),
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
    it('should create a new Peer instance', async () => {
      const peer = await honcho.peer('test-peer');

      expect(peer).toBeInstanceOf(Peer);
      expect(peer.id).toBe('test-peer');
    });

    it('should create peer with metadata and config', async () => {
      const metadata = { name: 'Test Peer' };
      const config = { observe_me: false };

      mockClient.workspaces.peers.getOrCreate.mockResolvedValue({
        id: 'test-peer',
        metadata: metadata,
        configuration: config,
      });

      await honcho.peer('test-peer', { metadata, config });

      expect(mockClient.workspaces.peers.getOrCreate).toHaveBeenCalledWith(
        'test-workspace',
        { id: 'test-peer', metadata: metadata, configuration: config }
      );
    });

    it('should throw error for empty peer ID', async () => {
      await expect(honcho.peer('')).rejects.toThrow();
    });

    it('should throw error for non-string peer ID', async () => {
      await expect(honcho.peer(null as any)).rejects.toThrow();
      await expect(honcho.peer(undefined as any)).rejects.toThrow();
      await expect(honcho.peer(123 as any)).rejects.toThrow();
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
      expect(mockClient.workspaces.peers.list).toHaveBeenCalledWith('test-workspace', { filters: undefined });
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
      expect(mockClient.workspaces.peers.list).toHaveBeenCalledWith('test-workspace', { filters: undefined });
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.list.mockRejectedValue(new Error('API Error'));

      await expect(honcho.getPeers()).rejects.toThrow();
    });
  });

  describe('session', () => {
    it('should create a new Session instance', async () => {
      const session = await honcho.session('test-session');

      expect(session).toBeInstanceOf(Session);
      expect(session.id).toBe('test-session');
    });

    it('should create session with metadata and config', async () => {
      const metadata = { name: 'Test Session' };
      const config = { anonymous: true };

      mockClient.workspaces.sessions.getOrCreate.mockResolvedValue({
        id: 'test-session',
        metadata: metadata,
        configuration: config,
      });

      await honcho.session('test-session', { metadata, config });

      expect(mockClient.workspaces.sessions.getOrCreate).toHaveBeenCalledWith(
        'test-workspace',
        { id: 'test-session', metadata: metadata, configuration: config }
      );
    });

    it('should throw error for empty session ID', async () => {
      await expect(honcho.session('')).rejects.toThrow();
    });

    it('should throw error for non-string session ID', async () => {
      await expect(honcho.session(null as any)).rejects.toThrow();
      await expect(honcho.session(undefined as any)).rejects.toThrow();
      await expect(honcho.session(123 as any)).rejects.toThrow();
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
      expect(mockClient.workspaces.sessions.list).toHaveBeenCalledWith('test-workspace', { filters: undefined });
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

      await expect(honcho.getSessions()).rejects.toThrow();
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

      await expect(honcho.getMetadata()).rejects.toThrow();
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

      await expect(honcho.setMetadata({ key: 'value' })).rejects.toThrow();
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

      await expect(honcho.getWorkspaces()).rejects.toThrow();
    });
  });

  describe('search', () => {
    it('should search for messages and return Page', async () => {
      const mockSearchResults = [
        { id: 'msg1', content: 'Hello world', peer_id: 'peer1' },
        { id: 'msg2', content: 'Hello there', peer_id: 'peer2' },
      ];
      mockClient.workspaces.search.mockResolvedValue(mockSearchResults);

      const results = await honcho.search('hello');

      expect(Array.isArray(results)).toBe(true);
      expect(mockClient.workspaces.search).toHaveBeenCalledWith('test-workspace', { query: 'hello', limit: undefined });
    });

    it('should handle empty search results', async () => {
      const mockSearchResults: any[] = [];
      mockClient.workspaces.search.mockResolvedValue(mockSearchResults);

      const results = await honcho.search('nonexistent');

      expect(Array.isArray(results)).toBe(true);
    });

    it('should throw error for empty query', async () => {
      await expect(honcho.search('')).rejects.toThrow();
      await expect(honcho.search('   ')).rejects.toThrow();
    });

    it('should throw error for non-string query', async () => {
      await expect(honcho.search(null as any)).rejects.toThrow();
      await expect(honcho.search(undefined as any)).rejects.toThrow();
      await expect(honcho.search(123 as any)).rejects.toThrow();
    });

    it('should handle complex search queries', async () => {
      const mockSearchResults: any[] = [];
      mockClient.workspaces.search.mockResolvedValue(mockSearchResults);

      const complexQuery = 'complex query with "quotes" and special characters!@#$%';
      await honcho.search(complexQuery);

      expect(mockClient.workspaces.search).toHaveBeenCalledWith('test-workspace', { query: complexQuery, limit: undefined });
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.search.mockRejectedValue(new Error('Search failed'));

      await expect(honcho.search('test')).rejects.toThrow();
    });
  });

  describe('getDeriverStatus', () => {
    it('should return deriver status without options', async () => {
      const mockStatus = {
        total_work_units: 10,
        completed_work_units: 5,
        in_progress_work_units: 3,
        pending_work_units: 2,
        sessions: { 'session1': { status: 'active' } },
      };
      mockClient.workspaces.deriverStatus.mockResolvedValue(mockStatus);

      const status = await honcho.getDeriverStatus();

      expect(status).toEqual({
        totalWorkUnits: 10,
        completedWorkUnits: 5,
        inProgressWorkUnits: 3,
        pendingWorkUnits: 2,
        sessions: { 'session1': { status: 'active' } },
      });
      expect(mockClient.workspaces.deriverStatus).toHaveBeenCalledWith('test-workspace', {});
    });

    it('should return deriver status with options', async () => {
      const mockStatus = {
        total_work_units: 5,
        completed_work_units: 3,
        in_progress_work_units: 1,
        pending_work_units: 1,
      };
      mockClient.workspaces.deriverStatus.mockResolvedValue(mockStatus);

      const status = await honcho.getDeriverStatus({
        observer: 'observer1',
        sender: 'sender1',
        session: 'session1',
      });

      expect(status).toEqual({
        totalWorkUnits: 5,
        completedWorkUnits: 3,
        inProgressWorkUnits: 1,
        pendingWorkUnits: 1,
        sessions: undefined,
      });
      expect(mockClient.workspaces.deriverStatus).toHaveBeenCalledWith('test-workspace', {
        observer_id: 'observer1',
        sender_id: 'sender1',
        session_id: 'session1',
      });
    });
  });

  describe('pollDeriverStatus', () => {
    it('should poll until processing is complete', async () => {
      const mockStatusComplete = {
        total_work_units: 5,
        completed_work_units: 5,
        in_progress_work_units: 0,
        pending_work_units: 0,
      };
      mockClient.workspaces.deriverStatus.mockResolvedValue(mockStatusComplete);

      const status = await honcho.pollDeriverStatus();

      expect(status).toEqual({
        totalWorkUnits: 5,
        completedWorkUnits: 5,
        inProgressWorkUnits: 0,
        pendingWorkUnits: 0,
        sessions: undefined,
      });
    });

    it('should timeout if processing takes too long', async () => {
      const mockStatusPending = {
        total_work_units: 5,
        completed_work_units: 2,
        in_progress_work_units: 2,
        pending_work_units: 1,
      };
      mockClient.workspaces.deriverStatus.mockResolvedValue(mockStatusPending);

      await expect(honcho.pollDeriverStatus({ timeoutMs: 100 })).rejects.toThrow();
    });
  });

  describe('updateMessage', () => {
    beforeEach(() => {
      mockClient.workspaces.sessions = {
        messages: {
          update: jest.fn(),
        },
      };
    });

    it('should update message metadata using Message object', async () => {
      const mockMessage: Message = {
        id: 'msg-123',
        session_id: 'session-456',
        content: 'Test message',
        peer_id: 'peer-789',
        created_at: '2024-01-01T00:00:00Z',
        token_count: 10,
        workspace_id: 'test-workspace',
      };
      const metadata = { updated: true, importance: 'high' };
      const mockUpdatedMessage = { ...mockMessage, metadata };

      mockClient.workspaces.sessions.messages.update.mockResolvedValue(mockUpdatedMessage);

      const result = await honcho.updateMessage(mockMessage, metadata);

      expect(result).toEqual(mockUpdatedMessage);
      expect(mockClient.workspaces.sessions.messages.update).toHaveBeenCalledWith(
        'test-workspace',
        'session-456',
        'msg-123',
        { metadata }
      );
    });

    it('should update message metadata using message ID and session ID', async () => {
      const messageId = 'msg-123';
      const sessionId = 'session-456';
      const metadata = { updated: true, importance: 'high' };
      const mockUpdatedMessage = {
        id: messageId,
        session_id: sessionId,
        content: 'Test message',
        peer_id: 'peer-789',
        metadata,
      };

      mockClient.workspaces.sessions.messages.update.mockResolvedValue(mockUpdatedMessage);

      const result = await honcho.updateMessage(messageId, metadata, sessionId);

      expect(result).toEqual(mockUpdatedMessage);
      expect(mockClient.workspaces.sessions.messages.update).toHaveBeenCalledWith(
        'test-workspace',
        sessionId,
        messageId,
        { metadata }
      );
    });

    it('should throw error when message is string ID but session ID is not provided', async () => {
      const messageId = 'msg-123';
      const metadata = { updated: true };

      await expect(honcho.updateMessage(messageId, metadata)).rejects.toThrow(
        'session is required when message is a string ID'
      );
    });

    it('should handle API errors', async () => {
      const mockMessage: Message = {
        id: 'msg-123',
        session_id: 'session-456',
        content: 'Test message',
        peer_id: 'peer-789',
        created_at: '2024-01-01T00:00:00Z',
        token_count: 10,
        workspace_id: 'test-workspace',
      };
      const metadata = { updated: true };

      mockClient.workspaces.sessions.messages.update.mockRejectedValue(
        new Error('Update failed')
      );

      await expect(honcho.updateMessage(mockMessage, metadata)).rejects.toThrow('Update failed');
    });
  });
});
