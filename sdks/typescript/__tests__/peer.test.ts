import { Peer } from '../src/peer';
import { Session } from '../src/session';
import { Page } from '../src/pagination';
import { Honcho } from '../src/client';

// Mock the @honcho-ai/core module
jest.mock('@honcho-ai/core', () => {
  return jest.fn().mockImplementation(() => ({
    workspaces: {
      peers: {
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
      },
      getOrCreate: jest.fn().mockResolvedValue({ id: 'test-workspace', metadata: {} }),
      update: jest.fn(),
      list: jest.fn(),
      search: jest.fn(),
    },
  }));
});

describe('Peer', () => {
  let honcho: Honcho;
  let peer: Peer;
  let mockClient: any;

  beforeEach(() => {
    jest.clearAllMocks();

    honcho = new Honcho({
      workspaceId: 'test-workspace',
      apiKey: 'test-key',
      environment: 'local',
    });

    peer = new Peer('test-peer', 'test-workspace', (honcho as any)._client);
    mockClient = (honcho as any)._client;
  });

  describe('constructor', () => {
    it('should initialize with correct properties', () => {
      const newPeer = new Peer('peer-id', 'test-workspace', mockClient);

      expect(newPeer.id).toBe('peer-id');
      expect(newPeer.workspaceId).toBe('test-workspace');
      expect(newPeer['_client']).toBe(mockClient);
    });
  });

  describe('chat', () => {
    it('should query peer representation and return response', async () => {
      const mockResponse = { content: 'Hello, I am a peer response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      const result = await peer.chat('Hello');

      expect(result).toBe('Hello, I am a peer response');
      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: undefined, target: undefined, session_id: undefined }
      );
    });

    it('should return null for None content', async () => {
      const mockResponse = { content: 'None' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      const result = await peer.chat('Hello');

      expect(result).toBeNull();
    });

    it('should return null for empty content', async () => {
      const mockResponse = { content: null };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      const result = await peer.chat('Hello');

      expect(result).toBeNull();
    });

    it('should handle chat with streaming option', async () => {
      const mockResponse = { content: 'Streamed response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', true);

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: true, target: undefined, session_id: undefined }
      );
    });

    it('should handle chat with target peer', async () => {
      const targetPeer = new Peer('target-peer', 'test-workspace', mockClient);
      const mockResponse = { content: 'Targeted response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', undefined, targetPeer);

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: undefined, target: 'target-peer', session_id: undefined }
      );
    });

    it('should handle chat with target as string', async () => {
      const mockResponse = { content: 'Targeted response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', undefined, 'string-target');

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: undefined, target: 'string-target', session_id: undefined }
      );
    });

    it('should handle chat with session ID', async () => {
      const mockResponse = { content: 'Session-specific response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', undefined, undefined, 'session-123');

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: undefined, target: undefined, session_id: 'session-123' }
      );
    });

    it('should handle all options together', async () => {
      const targetPeer = new Peer('target-peer', 'test-workspace', mockClient);
      const mockResponse = { content: 'Full options response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', true, targetPeer, 'session-456');

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: true, target: 'target-peer', session_id: 'session-456' }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.chat.mockRejectedValue(new Error('Chat failed'));

      await expect(peer.chat('Hello')).rejects.toThrow();
    });
  });

  describe('getSessions', () => {
    it('should return Page of Session instances', async () => {
      const mockSessionsData = {
        items: [
          { id: 'session1', metadata: {} },
          { id: 'session2', metadata: {} },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.sessions.list.mockResolvedValue(mockSessionsData);

      const sessionsPage = await peer.getSessions();

      expect(sessionsPage).toBeInstanceOf(Page);
      expect(mockClient.workspaces.peers.sessions.list).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { filter: undefined }
      );
    });

    it('should handle empty sessions list', async () => {
      const mockSessionsData = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.sessions.list.mockResolvedValue(mockSessionsData);

      const sessionsPage = await peer.getSessions();

      expect(sessionsPage).toBeInstanceOf(Page);
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.sessions.list.mockRejectedValue(new Error('Failed to get sessions'));

      await expect(peer.getSessions()).rejects.toThrow();
    });
  });

  describe('message', () => {
    it('should create message object without metadata', () => {
      const message = peer.message('Test content');

      expect(message).toEqual({
        peer_id: 'test-peer',
        content: 'Test content',
        metadata: undefined,
      });
    });

    it('should create message object with metadata', () => {
      const metadata = { importance: 'high', category: 'greeting' };
      const message = peer.message('Hello there', { metadata });

      expect(message).toEqual({
        peer_id: 'test-peer',
        content: 'Hello there',
        metadata: { importance: 'high', category: 'greeting' },
      });
    });

    it('should handle empty content', () => {
      const message = peer.message('');

      expect(message).toEqual({
        peer_id: 'test-peer',
        content: '',
        metadata: undefined,
      });
    });
  });

  describe('getMetadata', () => {
    it('should return peer metadata', async () => {
      const mockPeer = {
        id: 'test-peer',
        metadata: { name: 'Test Peer', role: 'assistant' },
      };
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

      const metadata = await peer.getMetadata();

      expect(metadata).toEqual({ name: 'Test Peer', role: 'assistant' });
      expect(mockClient.workspaces.peers.getOrCreate).toHaveBeenCalledWith(
        'test-workspace',
        { id: 'test-peer' }
      );
    });

    it('should return empty object when no metadata exists', async () => {
      const mockPeer = {
        id: 'test-peer',
        metadata: null,
      };
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

      const metadata = await peer.getMetadata();

      expect(metadata).toEqual({});
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.getOrCreate.mockRejectedValue(new Error('Peer not found'));

      await expect(peer.getMetadata()).rejects.toThrow();
    });
  });

  describe('setMetadata', () => {
    it('should update peer metadata', async () => {
      const metadata = { name: 'Updated Peer', status: 'active' };
      mockClient.workspaces.peers.update.mockResolvedValue({});

      await peer.setMetadata(metadata);

      expect(mockClient.workspaces.peers.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { metadata }
      );
    });

    it('should handle empty metadata', async () => {
      mockClient.workspaces.peers.update.mockResolvedValue({});

      await peer.setMetadata({});

      expect(mockClient.workspaces.peers.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { metadata: {} }
      );
    });

    it('should handle complex metadata objects', async () => {
      const complexMetadata = {
        profile: { name: 'Complex Peer', age: 25 },
        settings: { theme: 'dark', notifications: true },
        tags: ['ai', 'assistant', 'helpful'],
      };
      mockClient.workspaces.peers.update.mockResolvedValue({});

      await peer.setMetadata(complexMetadata);

      expect(mockClient.workspaces.peers.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { metadata: complexMetadata }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.update.mockRejectedValue(new Error('Update failed'));

      await expect(peer.setMetadata({ key: 'value' })).rejects.toThrow();
    });
  });

  describe('getPeerConfig', () => {
    it('should return peer configuration', async () => {
      const mockPeer = {
        id: 'test-peer',
        configuration: { observe_me: true, observe_others: false },
      };
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

      const config = await peer.getPeerConfig();

      expect(config).toEqual({ observe_me: true, observe_others: false });
      expect(mockClient.workspaces.peers.getOrCreate).toHaveBeenCalledWith(
        'test-workspace',
        { id: 'test-peer' }
      );
    });

    it('should return empty object when no configuration exists', async () => {
      const mockPeer = {
        id: 'test-peer',
        configuration: null,
      };
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

      const config = await peer.getPeerConfig();

      expect(config).toEqual({});
    });
  });

  describe('setPeerConfig', () => {
    it('should update peer configuration', async () => {
      const config = { observe_me: false, observe_others: true };
      mockClient.workspaces.peers.update.mockResolvedValue({});

      await peer.setPeerConfig(config);

      expect(mockClient.workspaces.peers.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { configuration: config }
      );
    });
  });

  describe('search', () => {
    it('should search peer messages and return array', async () => {
      const mockSearchResults = [
        { id: 'msg1', content: 'Hello world', peer_id: 'test-peer' },
        { id: 'msg2', content: 'Hello there', peer_id: 'test-peer' },
      ];
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const results = await peer.search('hello');

      expect(Array.isArray(results)).toBe(true);
      expect(mockClient.workspaces.peers.search).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'hello' }
      );
    });

    it('should handle empty search results', async () => {
      const mockSearchResults: any[] = [];
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const results = await peer.search('nonexistent');

      expect(Array.isArray(results)).toBe(true);
    });

    it('should throw error for empty query', async () => {
      await expect(peer.search('')).rejects.toThrow();
      await expect(peer.search('   ')).rejects.toThrow();
    });

    it('should throw error for non-string query', async () => {
      await expect(peer.search(null as any)).rejects.toThrow();
      await expect(peer.search(undefined as any)).rejects.toThrow();
      await expect(peer.search(123 as any)).rejects.toThrow();
    });

    it('should handle complex search queries', async () => {
      const mockSearchResults: any[] = [];
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const complexQuery = 'complex query with "quotes" and special characters!@#$%';
      await peer.search(complexQuery);

      expect(mockClient.workspaces.peers.search).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: complexQuery }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.search.mockRejectedValue(new Error('Search failed'));

      await expect(peer.search('test')).rejects.toThrow();
    });
  });
});
